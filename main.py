"""FastAPI application entry point for multimodal video ingestion and Q&A."""
import json
import logging
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from qdrant_client.models import PointStruct, SparseVector
from fastembed import SparseTextEmbedding

from config.config import get_settings
from core.agent import AgenticQueryEngine
from core.audio_processor import AudioProcessor
from core.chat_engine import AgenticVideoChatEngine
from core.detector import ObjectDetector
from core.embedder import VisionEmbedder
from core.ingestion import VideoIngestionPipeline
from core.ocr import OCREngine
from core.retriever import MultimodalRetriever
from core.scene import SceneRecognizer
from core.vector_db import VectorDBManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)
settings = get_settings()
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

pipeline = db_manager = embedder = audio_processor = detector = None
ocr_engine = action_engine = retriever = chat_engine = sparse_model = None
ingestion_jobs: dict[str, dict[str, Any]] = {}
chat_sessions: dict[str, dict[str, Any]] = {}


class ChatRequest(BaseModel):
    video_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=8000)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), min_length=1, max_length=128)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=20)


def _sparse(text: str) -> SparseVector:
    vector = next(sparse_model.embed([text or "video"]))
    return SparseVector(indices=vector.indices.tolist(), values=vector.values.tolist())


def _timestamp_from_name(name: str) -> tuple[float, float]:
    match = re.search(r"scene_([0-9.]+)_to_([0-9.]+)", name)
    if not match:
        raise ValueError(f"Cannot parse timestamp range from {name}")
    return float(match.group(1)), float(match.group(2))


def run_background_ingestion(video_id: str, raw_file_name: str) -> None:
    raw_path = pipeline.raw_dir / raw_file_name
    ingestion_jobs[video_id] = {"status": "processing", "video_id": video_id}
    try:
        workspace = pipeline.create_video_workspace(video_id)
        pipeline.chunk_video(raw_path, workspace["chunks"], workspace["frames"])
        frame_points: list[PointStruct] = []
        ocr_points: list[PointStruct] = []
        frame_files = sorted(workspace["frames"].glob("*.jpg"))
        log.info("[%s] Indexing %d sampled frames", video_id, len(frame_files))
        for frame_path in frame_files:
            try:
                start_time, end_time = _timestamp_from_name(frame_path.name)
                frame_index_match = re.search(r"_frame(\d+)\.jpg$", frame_path.name)
                frame_index = int(frame_index_match.group(1)) if frame_index_match else 1
                timestamp = min(end_time, start_time + (frame_index - 1) * 2.0)
                detections = detector.detect_objects(frame_path)
                labels = sorted({item["object"] for item in detections})
                object_text = ", ".join(labels)
                ocr_text = ocr_engine.extract_text(str(frame_path))
                clip_vector = embedder.generate_frame_embedding(str(frame_path))
                if not clip_vector:
                    continue
                frame_points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector={"clip": clip_vector, "bm25": _sparse(object_text)},
                    payload={
                        "video_id": video_id, "frame_name": frame_path.name,
                        "timestamp": timestamp, "type": "visual_frame",
                        "detected_objects": object_text, "ocr_text": ocr_text,
                        "raw_detections": detections,
                    },
                ))
                if ocr_text.strip():
                    dense = audio_processor.embedder.encode(ocr_text, normalize_embeddings=True).tolist()
                    ocr_points.append(PointStruct(
                        id=str(uuid.uuid4()),
                        vector={"dense": dense, "bm25": _sparse(ocr_text)},
                        payload={"video_id": video_id, "frame_name": frame_path.name,
                                 "timestamp": timestamp, "text": ocr_text, "type": "ocr_text"},
                    ))
            except Exception:
                log.exception("[%s] Frame processing failed: %s", video_id, frame_path.name)

        db_manager.upsert_frame_embeddings(frame_points)
        db_manager.upsert_ocr_embeddings(ocr_points)

        transcript_points: list[PointStruct] = []
        semantic_points: list[PointStruct] = []
        transcript_cache: list[dict] = []
        for chunk_path in sorted(workspace["chunks"].glob("scene_*.mp4")):
            try:
                start_time, end_time = _timestamp_from_name(chunk_path.name)
                audio_path = chunk_path.with_suffix(".wav")
                if audio_processor.extract_audio_from_chunk(chunk_path, audio_path):
                    result = audio_processor.transcribe_and_embed(audio_path)
                    if result["text"]:
                        transcript_points.append(PointStruct(
                            id=str(uuid.uuid4()),
                            vector={"dense": result["embedding"], "bm25": _sparse(result["text"])},
                            payload={
                                "video_id": video_id, "chunk_name": chunk_path.name,
                                "start_time": start_time, "end_time": end_time,
                                "text": result["text"], "type": "audio_transcript",
                            },
                        ))
                        transcript_cache.append({
                            "chunk_name": chunk_path.name, "start_time": start_time,
                            "end_time": end_time, "text": result["text"],
                        })
                audio_path.unlink(missing_ok=True)
                if action_engine is not None:
                    try:
                        action_text = ", ".join(action_engine.recognize_actions(chunk_path))
                        if action_text:
                            dense = audio_processor.embedder.encode(
                                action_text, normalize_embeddings=True
                            ).tolist()
                            semantic_points.append(PointStruct(
                                id=str(uuid.uuid4()),
                                vector={"dense": dense, "bm25": _sparse(action_text)},
                                payload={
                                    "video_id": video_id, "chunk_name": chunk_path.name,
                                    "start_time": start_time, "end_time": end_time,
                                    "semantic_text": action_text, "type": "action_labels",
                                },
                            ))
                    except Exception:
                        log.exception("[%s] Action recognition failed: %s", video_id, chunk_path.name)
            except Exception:
                log.exception("[%s] Chunk processing failed: %s", video_id, chunk_path.name)

        db_manager.upsert_transcript_embeddings(transcript_points)
        db_manager.upsert_scene_semantics_embeddings(semantic_points)
        transcript_path = workspace["audio"] / "transcripts" / "transcript.json"
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text(
            json.dumps({"video_id": video_id, "segments": transcript_cache}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        ingestion_jobs[video_id] = {
            "status": "completed", "video_id": video_id,
            "frames_indexed": len(frame_points), "ocr_records": len(ocr_points),
            "transcript_chunks": len(transcript_points), "action_records": len(semantic_points),
        }
        log.info("[%s] Ingestion completed", video_id)
    except Exception as exc:
        log.exception("[%s] Ingestion failed", video_id)
        ingestion_jobs[video_id] = {"status": "failed", "video_id": video_id, "error": str(exc)[:1000]}
    finally:
        raw_path.unlink(missing_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, db_manager, embedder, audio_processor, detector
    global ocr_engine, action_engine, retriever, chat_engine, sparse_model

    pipeline = VideoIngestionPipeline(settings.storage_dir, settings.max_video_chunk_seconds)
    db_manager = VectorDBManager(url=settings.qdrant_url)
    log.info("Connecting to Qdrant")
    db_manager.create_collections_if_not_exists()
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    embedder = VisionEmbedder(model_name=settings.clip_model)
    audio_processor = AudioProcessor(
        whisper_model_size=settings.whisper_model, device=settings.device,
        text_model=settings.text_embedding_model,
    )
    detector = ObjectDetector(model_version=settings.yolo_model, device=settings.device)
    ocr_engine = OCREngine(languages=["en"], device=settings.device)
    action_engine = None
    if settings.enable_scene_recognition:
        try:
            action_engine = SceneRecognizer()
        except Exception:
            log.exception("Action model unavailable; continuing without action labels")
    retriever = MultimodalRetriever(
        qdrant_url=settings.qdrant_url, device=settings.device,
        text_model=settings.text_embedding_model, clip_model=settings.clip_model,
        text_embedder=audio_processor.embedder, clip_embedder=embedder,
        sparse_model=sparse_model,
    )
    agent_engine = AgenticQueryEngine(ollama_url=settings.ollama_url, ollama_model=settings.ollama_model)
    chat_engine = AgenticVideoChatEngine(
        retriever=retriever, agent_engine=agent_engine,
        ollama_url=settings.ollama_url, ollama_model=settings.ollama_model,
        top_k=settings.retrieval_limit,
    )
    log.info("Application startup complete")
    yield
    if retriever is not None:
        retriever.close()
    if db_manager is not None:
        db_manager.close()
    log.info("Application shutdown complete")


app = FastAPI(
    title="Multimodal Agentic Video Search Engine",
    description="Upload videos, index visual/audio/text evidence, and ask timestamp-grounded questions.",
    version="2.0.0", lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "multimodal-video-search"}


@app.post("/ingest", status_code=202)
async def ingest_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    video_id = str(uuid.uuid4())
    destination = pipeline.raw_dir / f"{video_id}{suffix}"
    total = 0
    try:
        with destination.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > settings.max_upload_bytes:
                    raise HTTPException(status_code=413, detail="Uploaded video exceeds MAX_UPLOAD_BYTES.")
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    if total == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    ingestion_jobs[video_id] = {"status": "queued", "video_id": video_id}
    background_tasks.add_task(run_background_ingestion, video_id, destination.name)
    return {"status": "queued", "video_id": video_id, "status_url": f"/ingest/{video_id}"}


@app.get("/ingest/{video_id}")
async def get_ingestion_status(video_id: str):
    status = ingestion_jobs.get(video_id)
    if status is None:
        raise HTTPException(status_code=404, detail="No ingestion job found in this server process.")
    return status


@app.post("/api/v1/chat")
async def chat_with_video(body: ChatRequest):
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    status = ingestion_jobs.get(body.video_id)
    if status and status.get("status") in {"queued", "processing"}:
        raise HTTPException(status_code=409, detail="Video ingestion is still running.")
    if status and status.get("status") == "failed":
        raise HTTPException(status_code=409, detail="Video ingestion failed; inspect GET /ingest/{video_id}.")
    try:
        session = chat_sessions.get(body.session_id, {})
        if session.get("video_id") not in (None, body.video_id):
            session = {}
        result = await chat_engine.execute_chat(
            video_id=body.video_id, message=body.message,
            history=body.history, session=session,
        )
        chat_sessions[body.session_id] = {**result["session"], "video_id": body.video_id}
        return {
            "session_id": body.session_id, "response": result["response"],
            "needs_user_input": result["needs_user_input"], "options": result.get("options", []),
            "query_plan": result.get("query_plan"),
            "session_state": {
                "conversation_mode": result["session"].get("conversation_mode"),
                "active_chunk_ts": (result["session"].get("active_chunk") or {}).get("timestamp"),
            },
        }
    except Exception as exc:
        log.exception("Chat request failed")
        raise HTTPException(status_code=500, detail="Chat processing failed. Check server logs.") from exc


@app.delete("/api/v1/chat/session/{session_id}")
async def clear_chat_session(session_id: str):
    chat_sessions.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.app_host, port=settings.app_port, workers=1)
