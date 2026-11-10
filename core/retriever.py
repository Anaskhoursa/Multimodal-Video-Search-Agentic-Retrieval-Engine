import os
import torch
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client import models
import logging
from fastembed import SparseTextEmbedding
from qdrant_client.models import Filter, FieldCondition, MatchValue
bm25_model = SparseTextEmbedding(
    model_name="Qdrant/bm25"
)
log = logging.getLogger(__name__)

class MultimodalRetriever:
    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        print("Initializing Retrieval Embedders on:", self.device)
        # Load CLIP for visual search matching
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True, weights_only=False).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Load Text Embedder for speech & OCR matching
        self.text_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Generates a 384-dim dense vector for transcript and OCR queries."""
        return self.text_model.encode(text).tolist()

    def _get_clip_text_embedding(self, text: str) -> List[float]:
        """Generates a 512-dim dense vector using CLIP for frame matching."""
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
            # Normalize the embedding
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features[0].cpu().numpy().tolist()


    def get_temporal_context(
        self,
        video_id: str,
        center_timestamp: float,
        window_seconds: float = 20.0
    ) -> List[Dict]:
        """
        Fetches all indexed events within ±window_seconds of center_timestamp
        across all four collections, sorted chronologically.

        Returns a unified list of dicts with keys:
            timestamp, type, text (transcript/ocr), objects (visual), ocr_text (visual)
        """
        window_start = max(0.0, center_timestamp - window_seconds)
        window_end   = center_timestamp + window_seconds

        context_entries = []

        # ── video_frames ─────────────────────────────────────────────────────
        try:
            frame_points, _ = self.client.scroll(
                collection_name="video_frames",
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="video_id",
                            match=models.MatchValue(value=video_id)
                        ),
                        models.FieldCondition(
                            key="timestamp",
                            range=models.Range(gte=window_start, lte=window_end)
                        ),
                    ]
                ),
                limit=200,
                with_payload=True,
                with_vectors=False,
            )
            for pt in frame_points:
                p = pt.payload
                context_entries.append({
                    "timestamp": float(p.get("timestamp", 0.0)),
                    "type":      "visual",
                    "objects":   [o.strip() for o in p.get("detected_objects", "").split(",") if o.strip()],
                    "ocr_text":  p.get("ocr_text") or "",
                })
        except Exception as e:
            log.warning(f"get_temporal_context: video_frames scroll failed: {e}")

        # ── video_transcripts ────────────────────────────────────────────────
        try:
            transcript_points, _ = self.client.scroll(
                collection_name="video_transcripts",
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="video_id",
                            match=models.MatchValue(value=video_id)
                        ),
                        models.FieldCondition(
                            key="start_time",                    # transcripts use start_time
                            range=models.Range(gte=window_start, lte=window_end)
                        ),
                    ]
                ),
                limit=100,
                with_payload=True,
                with_vectors=False,
            )
            for pt in transcript_points:
                p = pt.payload
                text = p.get("text", "").strip()
                if text:
                    context_entries.append({
                        "timestamp": float(p.get("start_time", 0.0)),
                        "type":      "transcript",
                        "text":      text,
                    })
        except Exception as e:
            log.warning(f"get_temporal_context: video_transcripts scroll failed: {e}")

        # ── video_ocr ────────────────────────────────────────────────────────
        try:
            ocr_points, _ = self.client.scroll(
                collection_name="video_ocr",
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="video_id",
                            match=models.MatchValue(value=video_id)
                        ),
                        models.FieldCondition(
                            key="timestamp",
                            range=models.Range(gte=window_start, lte=window_end)
                        ),
                    ]
                ),
                limit=100,
                with_payload=True,
                with_vectors=False,
            )
            for pt in ocr_points:
                p = pt.payload
                text = p.get("text", "").strip()
                if text:
                    context_entries.append({
                        "timestamp": float(p.get("timestamp", 0.0)),
                        "type":      "ocr",
                        "text":      text,
                    })
        except Exception as e:
            log.warning(f"get_temporal_context: video_ocr scroll failed: {e}")

        # ── video_scene_semantics ────────────────────────────────────────────
        try:
            semantics_points, _ = self.client.scroll(
                collection_name="video_scene_semantics",
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="video_id",
                            match=models.MatchValue(value=video_id)
                        ),
                        models.FieldCondition(
                            key="start_time",                    # semantics use start_time
                            range=models.Range(gte=window_start, lte=window_end)
                        ),
                    ]
                ),
                limit=100,
                with_payload=True,
                with_vectors=False,
            )
            for pt in semantics_points:
                p = pt.payload
                text = p.get("semantic_text", "").strip()
                if text:
                    context_entries.append({
                        "timestamp": float(p.get("start_time", 0.0)),
                        "type":      "action",
                        "text":      text,
                    })
        except Exception as e:
            log.warning(f"get_temporal_context: video_scene_semantics scroll failed: {e}")

        # ── Deduplicate and sort ─────────────────────────────────────────────
        seen = set()
        unique = []
        for entry in sorted(context_entries, key=lambda x: x["timestamp"]):
            # Deduplicate on (timestamp, type, text snippet)
            key = (
                round(entry["timestamp"], 1),
                entry["type"],
                entry.get("text", "")[:60]
            )
            if key not in seen:
                seen.add(key)
                unique.append(entry)

        log.info(
            f"get_temporal_context: video={video_id} center={center_timestamp:.1f}s "
            f"window=±{window_seconds}s → {len(unique)} entries"
        )
        return unique

    def rrf_fusion(
        self,
        ranked_lists: Dict[str, List[Any]],
        k: int = 60,
        weights: Dict[str, float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generic Reciprocal Rank Fusion.

        Parameters
        ----------
        ranked_lists:
            {
                "frames": [...],
                "transcript_dense": [...],
                "transcript_bm25": [...],
                ...
            }

        weights:
            Optional weight for each ranked list.

        Returns
        -------
        List of fused results sorted by RRF score.
        """

        if weights is None:
            weights = {name: 1.0 for name in ranked_lists}

        fused = {}

        for stream_name, results in ranked_lists.items():

            stream_weight = weights.get(stream_name, 1.0)

            for rank, hit in enumerate(results, start=1):

                # -------------------------------
                # Raw Qdrant search result
                # -------------------------------
                if hasattr(hit, "payload"):

                    payload = hit.payload

                    timestamp = (
                        payload.get("timestamp")
                        or payload.get("start_time")
                        or 0.0
                    )

                    key = (
                        payload["video_id"],
                        round(float(timestamp), 1),
                    )

                    if key not in fused:
                        fused[key] = {
                            "video_id": payload["video_id"],
                            "timestamp": round(float(timestamp), 1),
                            "rrf_score": 0.0,
                            "matches": [],
                        }

                    fused[key]["rrf_score"] += stream_weight / (k + rank)

                    fused[key]["matches"].append({
                        "stream": stream_name,
                        "rank": rank,
                        "raw_score": hit.score,
                    })

                # -------------------------------
                # Already-fused result
                # -------------------------------
                else:

                    key = (
                        hit["video_id"],
                        round(float(hit["timestamp"]), 1),
                    )

                    if key not in fused:
                        fused[key] = {
                            "video_id": hit["video_id"],
                            "timestamp": hit["timestamp"],
                            "rrf_score": 0.0,
                            "matches": [],
                        }

                    fused[key]["rrf_score"] += stream_weight / (k + rank)

                    fused[key]["matches"].extend(hit.get("matches", []))

        return sorted(
            fused.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )
    def search(self, video_id: Any, query: str, limit: int = 5, weights: Dict[str, float] = None) -> List[Dict[str, Any]]:
        if weights is None:
            weights = {"frames": 1.2, "transcripts": 1.0, "ocr": 1.0}
        # 1. Generate embeddings for query
        dense_text_embedding = self._get_text_embedding(query)
        dense_clip_embedding = self._get_clip_text_embedding(query)
        bm25_query = next(bm25_model.embed([query]))
        # 2. Query Qdrant collections independently
        # Query Visual Frames
        frame_dense = self.client.search(
            collection_name="video_frames",
            query_vector=("clip", dense_clip_embedding),
            limit=limit * 2,
            query_filter=Filter(
                must=[
                    FieldCondition(key="video_id", match=MatchValue(value=video_id))
                ]
            )
        )
        frame_sparse = self.client.search(
            collection_name="video_frames",
            query_vector=models.NamedSparseVector(
                name="bm25",
                vector=models.SparseVector(
                    indices=bm25_query.indices,
                    values=bm25_query.values,
                ),
            ),
            limit=20,
            query_filter=Filter(
                must=[
                    FieldCondition(key="video_id", match=MatchValue(value=video_id))
                ]
            )
            
        )
        
        frames_results = self.rrf_fusion(
            ranked_lists={
                "frames_clip": frame_dense,
                "frames_bm25": frame_sparse,
            },
            weights={
                "frames_clip": 1.2,
                "frames_bm25": 1.0,
            },
        )

        # Query Spoken Transcripts
        transcripts_dense = self.client.search(
            collection_name="video_transcripts",
            query_vector=("dense", dense_text_embedding),
            limit=limit * 2,
            query_filter=Filter(
                must=[
                    FieldCondition(key="video_id", match=MatchValue(value=video_id))
                ]
            )
        )
        transcripts_bm25 = self.client.search(
            collection_name="video_transcripts",
            query_vector=models.NamedSparseVector(
                name="bm25",
                vector=models.SparseVector(
                    indices=bm25_query.indices,
                    values=bm25_query.values,
                ),
            ),
            limit=limit * 2,
            query_filter=Filter(
                must=[
                    FieldCondition(key="video_id", match=MatchValue(value=video_id))
                ]
            )
        )
        transcripts_results = self.rrf_fusion(
            ranked_lists={
                "transcripts_dense": transcripts_dense,
                "transcripts_bm25": transcripts_bm25,
            },
            weights={
                "transcripts_dense": 1.0,
                "transcripts_bm25": 1.2,
            },
        )
        # Query On-Screen OCR
        ocr_dense = self.client.search(
            collection_name="video_ocr",
            query_vector=("dense", dense_text_embedding),
            limit=limit * 2,
            query_filter=Filter(
                must=[
                    FieldCondition(key="video_id", match=MatchValue(value=video_id))
                ]
            )
        )
        ocr_bm25 = self.client.search(
            collection_name="video_ocr",
            query_vector=models.NamedSparseVector(
                name="bm25",
                vector=models.SparseVector(
                    indices=bm25_query.indices,
                    values=bm25_query.values,
                ),
            ),
            limit=limit * 2,
            query_filter=Filter(
                must=[
                    FieldCondition(key="video_id", match=MatchValue(value=video_id))
                ]
            )            
        )

        ocr_results = self.rrf_fusion(
            ranked_lists={
                "ocr_dense": ocr_dense,
                "ocr_bm25": ocr_bm25,
            },
            weights={
                "ocr_dense": 1.0,
                "ocr_bm25": 1.2,
            }
        )
        
                # Query On-Screen OCR
        actions_dense = self.client.search(
            collection_name="video_scene_semantics",
            query_vector=("dense", dense_text_embedding),
            limit=limit * 2,
            query_filter=Filter(
                must=[
                    FieldCondition(key="video_id", match=MatchValue(value=video_id))
                ]
            )            
        )
        actions_bm25 = self.client.search(
            collection_name="video_scene_semantics",
            query_vector=models.NamedSparseVector(
                name="bm25",
                vector=models.SparseVector(
                    indices=bm25_query.indices,
                    values=bm25_query.values,
                ),
            ),
            limit=limit * 2,
             query_filter=Filter(
                must=[
                    FieldCondition(key="video_id", match=MatchValue(value=video_id))
                ]
            )           
        )

        actions_results = self.rrf_fusion(
            ranked_lists={
                "actions_dense": actions_dense,
                "actions_bm25": actions_bm25,
            },
            weights={
                "actions_dense": 1.0,
                "actions_bm25": 1.2,
            }
        )
        
        final_results = self.rrf_fusion(
            ranked_lists={
                "frames": frames_results,
                "transcripts": transcripts_results,
                "ocr": ocr_results,
                "actions": actions_results
            },
            weights={
                "frames": 1.2,
                "transcripts": 1.0,
                "ocr": 1.0,
                "actions": 1.4
            },
        )

        return final_results[:limit]