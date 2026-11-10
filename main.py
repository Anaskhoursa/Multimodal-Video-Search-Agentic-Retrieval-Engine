import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import uuid
import glob
import json
import shutil
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Query
from qdrant_client.models import PointStruct, SparseVector
from pydantic import BaseModel, Field

from core.ingestion import VideoIngestionPipeline
from core.embedder import VisionEmbedder
from core.vector_db import VectorDBManager
from core.audio_processor import AudioProcessor
from core.detector import ObjectDetector
from core.ocr import OCREngine
from core.retriever import MultimodalRetriever
from core.agent import AgenticQueryEngine, QueryPlan
from core.scene import SceneRecognizer
# Use logging instead of print — bypasses Windows stdout buffering
from core.chat_engine import AgenticVideoChatEngine
from typing import Dict, List
from fastembed import SparseTextEmbedding

bm25_model = SparseTextEmbedding(
    model_name="Qdrant/bm25"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# Global placeholders
pipeline = VideoIngestionPipeline(base_storage_dir="./storage")
db_manager = VectorDBManager(host="localhost", port=6333)
embedder = None
audio_processor = None
detector = None
ocr_engine = None
agent_engine = None
action_engine = None
chat_engine = None
chat_sessions: Dict[str, Dict] = {}   # in-memory session store keyed by session_id

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chat_engine, embedder, audio_processor, detector, ocr_engine, retriever, agent_engine, action_engine
    log.info("--- Starting Server Lifespan Initialization ---")

    log.info("Initializing Qdrant collections...")
    db_manager.create_collections_if_not_exists()

    log.info("Loading CLIP Vision Embedder...")
    embedder = VisionEmbedder()
    log.info("CLIP loaded.")

    log.info("Loading Faster-Whisper & Text Embedder...")
    audio_processor = AudioProcessor(whisper_model_size="base", device="cuda")
    log.info("Audio processor loaded.")
    
    print("Loading YOLOv10 Object Detector...")
    detector = ObjectDetector(model_version="yolov10n", device="cuda")
    
    log.info("Initializing Phase 4 Agentic Engine...")
    agent_engine = AgenticQueryEngine(ollama_url="http://localhost:11434")
    
    print("Loading EasyOCR Text Reader Engine...")
    ocr_engine = OCREngine(languages=["en"], device="cuda")
    
    print("Loading Action Engine...")
    action_engine = SceneRecognizer()
    # Initialize retriever here so it warms up on cuda
    log.info("Loading Multimodal Retriever...")
    retriever = MultimodalRetriever()
    log.info("Retriever loaded.")
    log.info("Loading Chat Engine...")
    chat_engine = AgenticVideoChatEngine(
        retriever=retriever,
        agent_engine=agent_engine,
        ollama_url="http://localhost:11434/api/chat",
        ollama_model="llama3",
        context_window_seconds=20.0,
        follow_up_window_seconds=45.0,
        confidence_gap_ratio=0.4,
        top_k=5
    )
    log.info("Chat engine ready.")
    log.info("--- All models loaded. Server ready! ---")
    yield
    log.info("Shutting down...")


app = FastAPI(title="Multimodal Agentic Video Engine", lifespan=lifespan)
retriever = None



class ChatRequest(BaseModel):
    video_id:   str  = Field(..., description="The video to chat about")
    message:    str  = Field(..., description="User's message")
    session_id: str  = Field(..., description="Unique session ID — generate once per conversation on the client")
    history:    List[Dict[str, str]] = Field(default=[], description="Full chat history [{role, content}]")

import os
import glob
import json
import uuid
import subprocess
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import detect_silence


def get_audio_silence_timestamps(
    audio_path: Path, 
    video_duration: float, 
    max_chunk_sec: float = 45.0, 
    min_silence_len: int = 700, 
    silence_thresh: int = -35
) -> list[tuple[float, float]]:
    """
    Detects audio pauses to slice static/lecture videos into clean segments.
    Falls back to fixed-window cuts if no speech pauses are detected.
    """
    try:
        sound = AudioSegment.from_file(audio_path)
        # Detect silence intervals in milliseconds -> converts to seconds
        silences = detect_silence(sound, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
        silence_cuts = [s[0] / 1000.0 for s in silences]
    except Exception as e:
        print(f"Warning: Audio silence detection failed ({e}). Defaulting to fixed windows.")
        silence_cuts = []

    chunks = []
    current_start = 0.0

    while current_start < video_duration:
        target_end = current_start + max_chunk_sec
        if target_end >= video_duration:
            chunks.append((current_start, video_duration))
            break

        # Find a natural audio silence pause near the target split point
        best_cut = None
        for cut in silence_cuts:
            if current_start + 15.0 <= cut <= target_end:
                best_cut = cut

        # If a pause exists, cut at the pause; otherwise, enforce the hard max_chunk_sec limit
        cut_point = best_cut if best_cut else target_end
        chunks.append((current_start, cut_point))
        current_start = cut_point

    return chunks


def run_background_ingestion(video_id: str, raw_file_name: str, fps: float = 1.0):
    global audio_processor
    raw_video_path = pipeline.raw_dir / raw_file_name
    workspace = pipeline.create_video_workspace(video_id)

    # Phase 1 — Ingestion
    pipeline.extract_audio(raw_video_path, workspace["audio"] / "audio.wav")
    pipeline.chunk_video(raw_video_path, workspace["chunks"], workspace["frames"])

    # --- Silence-based Sub-Chunking Post-Processor for Long Scenes ---
    MAX_CHUNK_SEC = 45.0
    existing_chunks = sorted(glob.glob(os.path.join(workspace["chunks"], "scene_*.mp4")))

    for chunk_path_str in existing_chunks:
        chunk_path = Path(chunk_path_str)
        chunk_name = chunk_path.name

        try:
            time_parts = chunk_name.replace("scene_", "").replace(".mp4", "").split("_to_")
            start_time = float(time_parts[0])
            end_time = float(time_parts[1])
        except (ValueError, IndexError):
            continue

        chunk_duration = end_time - start_time

        # If the chunk exceeds MAX_CHUNK_SEC, sub-split it using audio silence
        if chunk_duration > MAX_CHUNK_SEC:
            print(f"[{video_id}] Chunk '{chunk_name}' ({chunk_duration:.2f}s) exceeds threshold. Slicing with audio silence...")

            sub_splits = get_audio_silence_timestamps(
                audio_path=chunk_path,
                video_duration=chunk_duration,
                max_chunk_sec=MAX_CHUNK_SEC
            )

            # Re-slice into sub-chunks if more than 1 split occurred
            if len(sub_splits) > 1:
                for sub_start_rel, sub_end_rel in sub_splits:
                    abs_start = round(start_time + sub_start_rel, 2)
                    abs_end = round(start_time + sub_end_rel, 2)

                    sub_chunk_name = f"scene_{abs_start:.2f}_to_{abs_end:.2f}.mp4"
                    sub_chunk_path = Path(workspace["chunks"]) / sub_chunk_name

                    cmd = [
                        "ffmpeg", "-y",
                        "-ss", str(sub_start_rel),
                        "-i", str(chunk_path),
                        "-t", str(sub_end_rel - sub_start_rel),
                        "-c", "copy",
                        str(sub_chunk_path)
                    ]
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # Remove original long chunk so Phase 3 only processes sub-chunks
                if chunk_path.exists():
                    chunk_path.unlink()

    # 2. Multi-Modal Vision Processing Layer (Phase 2: CLIP + YOLO + OCR)
    print(f"[{video_id}] Processing video frames through Vision Pipeline...")
    frame_files = sorted(glob.glob(os.path.join(workspace["frames"], "*.jpg")))
    
    qdrant_points = []
    qdrant_ocr_points = []
    
    for index, frame_path in enumerate(frame_files):
        # Frame path example: "C:/.../frames/chunk_0.00_3.29_frame0.jpg"
        filename = os.path.basename(frame_path)
        
        try:
            # 1. Strip the extension and split by underscores
            # ["chunk", "0.00", "3.29", "frame0"]
            parts = filename.replace(".jpg", "").split("_")
            
            # 2. Extract the start time (index 1) and end time (index 2)
            start_time = float(parts[1])
            end_time = float(parts[2])
            
            # Use the scene start time as the anchor timestamp for indexing
            timestamp_seconds = start_time
            
        except (ValueError, IndexError) as e:
            # Fallback if filename parsing fails for an unexpected file
            print(f"Warning: Could not parse timestamp from filename '{filename}'. Using index fallback. Error: {e}")
            timestamp_seconds = float(index) # Fallback to index approximatio
        
        # A. Detect Physical Objects
        detections = detector.detect_objects(frame_path, confidence_threshold=0.3)
        detected_labels = list(set([det["object"] for det in detections]))
        detected_labels_bm25 = next(bm25_model.embed([", ".join(detected_labels)]))
        # B. Read On-Screen Text (OCR)
        on_screen_text = ocr_engine.extract_text(frame_path)
        ocr_embedding = None
        
        if on_screen_text:
            print(f"   -> OCR Found text in frame_{index:04d}: \"{on_screen_text[:30]}...\"")
            # Reuse our text embedding engine to vectorize the text found on screen
            ocr_embedding = audio_processor.embedder.encode(on_screen_text).tolist()
            ocr_bm25_embedding = next(bm25_model.embed([on_screen_text]))
            # Queue for video_ocr collection upload
            qdrant_ocr_points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": ocr_embedding,
                    "bm25": SparseVector(
                        indices=ocr_bm25_embedding.indices,
                        values=ocr_bm25_embedding.values
                    )
                },
                payload={
                    "video_id": video_id,
                    "frame_name": os.path.basename(frame_path),
                    "timestamp": timestamp_seconds,
                    "text": on_screen_text,
                    "type": "ocr_text"
                }
            ))
        
        # C. Generate CLIP Feature Vector
        embedding = embedder.generate_frame_embedding(frame_path)
        
        if embedding:
            point_id = str(uuid.uuid4())
            point = PointStruct(
                id=point_id,
                vector={
                    "clip": embedding,
                    "bm25": SparseVector(indices= detected_labels_bm25.indices, values=detected_labels_bm25.values)
                },
                payload={
                    "video_id": video_id,
                    "frame_name": os.path.basename(frame_path),
                    "timestamp": timestamp_seconds,
                    "type": "visual_frame",
                    "detected_objects": ", ".join(detected_labels),
                    "ocr_text": on_screen_text,                 # Injected text data mapping
                    "raw_detections": detections
                }
            )
            qdrant_points.append(point)
            
    if qdrant_points:
        db_manager.upsert_frame_embeddings(qdrant_points)
        print(f"[{video_id}] Indexed {len(qdrant_points)} frames with Object + OCR tags into Qdrant.")

    if qdrant_ocr_points:
        db_manager.upsert_ocr_embeddings(qdrant_ocr_points)
        print(f"[{video_id}] Indexed {len(qdrant_ocr_points)} frames with explicit OCR text embeddings.")
    
    # Phase 3 — Audio transcription
    log.info(f"[{video_id}] Transcribing audio chunks...")
    chunk_files = sorted(glob.glob(os.path.join(workspace["chunks"], "scene_*.mp4")))

    transcript_cache = []
    transcript_points = []
    
    qdrant_semantics_points = []

    for chunk_path in chunk_files:
        chunk_name = os.path.basename(chunk_path)

        try:
            time_parts = chunk_name.replace("scene_", "").replace(".mp4", "").split("_to_")
            start_time = float(time_parts[0])
            end_time = float(time_parts[1])
        except ValueError:
            start_time, end_time = 0.0, 0.0

        temp_audio_path = Path(chunk_path).with_suffix(".wav")

        if audio_processor.extract_audio_from_chunk(Path(chunk_path), temp_audio_path):
            result = audio_processor.transcribe_and_embed(temp_audio_path)
            sparse = next(bm25_model.embed([result["text"]]))
            if temp_audio_path.exists():
                temp_audio_path.unlink()

            if result["text"]:
                log.info(f"   [{chunk_name}]: \"{result['text'][:60]}...\"")

                transcript_points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector={"dense":result["embedding"], "bm25": SparseVector(indices = sparse.indices, values=sparse.values)},
                    payload={
                        "video_id": video_id,
                        "chunk_name": chunk_name,
                        "start_time": start_time,
                        "end_time": end_time,
                        "text": result["text"],
                        "type": "audio_transcript"
                    }
                ))

                transcript_cache.append({
                    "chunk_name": chunk_name,
                    "start_time": start_time,
                    "end_time": end_time,
                    "text": result["text"]
                })

        actions = action_engine.recognize_actions(Path(chunk_path))
        if actions:
            actions = ", ".join(actions)
            actions_dense = audio_processor.embedder.encode(actions).tolist()
            actions_bm25 = next(bm25_model.embed([actions]))
            qdrant_semantics_points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense":actions_dense, "bm25": SparseVector(indices = actions_bm25.indices, values=actions_bm25.values)},
                payload={
                    "video_id": video_id,
                    "chunk_name": chunk_name,
                    "start_time": start_time,
                    "end_time": end_time,
                    "semantic_text": actions,

                }
            ))
            
    if transcript_points:
        db_manager.upsert_transcript_embeddings(transcript_points)
        log.info(f"[{video_id}] Indexed {len(transcript_points)} transcript vectors.")
        
    if qdrant_semantics_points:
        db_manager.upsert_scene_semantics_embeddings(qdrant_semantics_points)
        log.info(f"[{video_id}] Indexed {len(qdrant_semantics_points)} semantic vectors.")

    transcript_path = Path(workspace["audio"]).parent / "transcripts" / "transcript.json"
    transcript_path.parent.mkdir(exist_ok=True)
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump({"video_id": video_id, "segments": transcript_cache}, f, indent=2, ensure_ascii=False)

    log.info(f"[{video_id}] Ingestion complete.")

@app.post("/ingest")
async def ingest_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Unsupported file format.")

    video_id = str(uuid.uuid4())
    file_extension = file.filename.split(".")[-1]
    saved_filename = f"{video_id}.{file_extension}"
    target_destination = pipeline.raw_dir / saved_filename

    with target_destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(run_background_ingestion, video_id, saved_filename)

    return {
        "status": "queued",
        "video_id": video_id,
        "message": "Video uploaded. Pipelines running in background."
    }


from pydantic import BaseModel, Field

# 1. Define the incoming request body data shape
class AgentSearchRequest(BaseModel):
    q: str = Field(..., description="The natural language query intended for the agent")
    limit: int = Field(default=5, description="Results to return per channel matching plan")

# 2. Change the method to POST and accept the schema as an argument
# @app.post("/api/v4/agent/search")
# async def agentic_search_endpoint(body: AgentSearchRequest):
#     if not body.q.strip():
#         raise HTTPException(status_code=400, detail="Query string cannot be empty.")
        
#     try:
#         # Step 1: LLM planning phase
#         plan: QueryPlan = await agent_engine.plan_search(body.q)
        
#         # Step 2: Dynamically shift weights based on targeted channels
#         weights = {
#             "frames": 1.2 if "frames" in plan.target_channels else 0.0,
#             "transcripts": 1.0 if "transcripts" in plan.target_channels else 0.0,
#             "ocr": 1.0 if "ocr" in plan.target_channels else 0.0
#         }
        
#         # Step 3: Retrieve across collections
#         raw_results = retriever.search(
#             query=plan.optimized_query, 
#             limit=body.limit, 
#             weights=weights
#         )
        
#         # Step 4: Apply structural temporal filters if discovered by the agent plan
#         fused_results = []
#         for match in raw_results:
#             timestamp = match.get("payload", {}).get("timestamp", match.get("payload", {}).get("start_time", 0.0))
#             if plan.temporal_filter:
#                 start_lim = plan.temporal_filter.get("start", 0.0)
#                 end_lim = plan.temporal_filter.get("end", float("inf"))
#                 if not (start_lim <= timestamp <= end_lim):
#                     continue # Skip entry out of structural scope
#             fused_results.append(match)
            
#         return {
#             "agent_execution_plan": plan.dict(),
#             "results_fused": fused_results[:body.limit]
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Agentic query engine error: {str(e)}")

@app.post("/api/v1/chat")
async def chat_with_video(body: ChatRequest):
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if not body.video_id.strip():
        raise HTTPException(status_code=400, detail="video_id cannot be empty.")

    try:
        # Restore session state for this conversation
        session = chat_sessions.get(body.session_id, {})

        result = await chat_engine.execute_chat(
            video_id=body.video_id,
            message=body.message,
            history=body.history,
            session=session
        )

        # Persist updated session state
        chat_sessions[body.session_id] = result["session"]

        return {
            "session_id":       body.session_id,
            "response":         result["response"],
            "needs_user_input": result["needs_user_input"],
            "options":          result.get("options", []),
            "query_plan":       result.get("query_plan"),
            "session_state": {
                "conversation_mode": result["session"].get("conversation_mode"),
                "active_chunk_ts":   result["session"].get("active_chunk", {}).get("timestamp") if result["session"].get("active_chunk") else None,
            }
        }

    except Exception as e:
        import traceback
        log.error(f"Chat failed:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Chat engine error: {str(e)}")


@app.delete("/api/v1/chat/session/{session_id}")
async def clear_chat_session(session_id: str):
    """Clears session state — call this when the user closes the chat or switches videos."""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    return {"status": "cleared", "session_id": session_id}


# Critical on Windows — prevents infinite spawn loop
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,   # Never use reload=True with GPU models on Windows
        workers=1       # Single worker — models are loaded once in lifespan
    )