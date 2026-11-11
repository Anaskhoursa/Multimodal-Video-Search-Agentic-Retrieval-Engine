"""Multimodal dense+sparse retrieval with video-scoped filtering and RRF."""
import logging
from typing import Any
from config.config import get_settings

log = logging.getLogger(__name__)


class MultimodalRetriever:
    def __init__(self, qdrant_url: str | None = None, device: str | None = None,
                 text_model: str | None = None, clip_model: str | None = None,
                 text_embedder=None, clip_embedder=None, sparse_model=None):
        import torch
        from fastembed import SparseTextEmbedding
        from sentence_transformers import SentenceTransformer
        from transformers import CLIPModel, CLIPProcessor
        from qdrant_client import QdrantClient, models

        settings = get_settings()
        requested = device or settings.device
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA requested but unavailable; falling back to CPU")
            self.device = "cpu"
        self.models = models
        self.client = QdrantClient(url=qdrant_url or settings.qdrant_url)
        self.text_model = text_embedder or SentenceTransformer(
            text_model or settings.text_embedding_model, device=self.device
        )
        if clip_embedder is not None:
            self.clip_model = clip_embedder.model
            self.clip_processor = clip_embedder.processor
        else:
            clip_name = clip_model or settings.clip_model
            self.clip_model = CLIPModel.from_pretrained(clip_name).to(self.device).eval()
            self.clip_processor = CLIPProcessor.from_pretrained(clip_name)
        self.sparse_model = sparse_model or SparseTextEmbedding(model_name="Qdrant/bm25")

    def _text_vector(self, text: str) -> list[float]:
        return self.text_model.encode(text, normalize_embeddings=True).tolist()

    def _clip_vector(self, text: str) -> list[float]:
        import torch
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.inference_mode():
            features = torch.nn.functional.normalize(self.clip_model.get_text_features(**inputs), dim=-1)
        return features[0].detach().cpu().tolist()

    def _sparse_vector(self, text: str):
        embedded = next(self.sparse_model.embed([text or "video"]))
        return self.models.SparseVector(indices=embedded.indices.tolist(), values=embedded.values.tolist())

    @staticmethod
    def rrf_fusion(ranked_lists: dict[str, list[Any]], k: int = 60,
                   weights: dict[str, float] | None = None) -> list[dict]:
        weights = weights or {}
        fused: dict[tuple[str, float], dict] = {}
        for stream, results in ranked_lists.items():
            weight = weights.get(stream, 1.0)
            for rank, hit in enumerate(results, start=1):
                if isinstance(hit, dict):
                    video_id = hit.get("video_id")
                    timestamp = round(float(hit.get("timestamp", 0.0)), 1)
                    if not video_id:
                        continue
                    key = (video_id, timestamp)
                    row = fused.setdefault(key, {
                        "video_id": video_id, "timestamp": timestamp,
                        "rrf_score": 0.0, "matches": [], "evidence": [],
                    })
                    row["rrf_score"] += weight / (k + rank)
                    row["matches"].extend(hit.get("matches", []))
                    row["evidence"].extend(hit.get("evidence", []))
                    continue
                payload = getattr(hit, "payload", None) or {}
                video_id = payload.get("video_id")
                if not video_id:
                    continue
                timestamp = payload.get("timestamp")
                if timestamp is None:
                    timestamp = payload.get("start_time", 0.0)
                timestamp = round(float(timestamp or 0.0), 1)
                key = (video_id, timestamp)
                row = fused.setdefault(key, {
                    "video_id": video_id, "timestamp": timestamp,
                    "rrf_score": 0.0, "matches": [], "evidence": [],
                })
                row["rrf_score"] += weight / (k + rank)
                row["matches"].append({
                    "stream": stream, "rank": rank,
                    "raw_score": float(getattr(hit, "score", 0.0)),
                })
                text = payload.get("text") or payload.get("semantic_text") or payload.get("detected_objects") or payload.get("ocr_text") or ""
                if text:
                    row["evidence"].append({"stream": stream, "text": str(text)[:1200]})
        return sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)

    def _search_collection(self, collection: str, dense_name: str, dense_vector: list[float],
                           sparse, video_id: str, limit: int,
                           dense_weight: float, sparse_weight: float) -> list[dict]:
        models = self.models
        query_filter = models.Filter(must=[
            models.FieldCondition(key="video_id", match=models.MatchValue(value=video_id))
        ])
        ranked: dict[str, list] = {}
        try:
            ranked[f"{collection}_dense"] = self.client.search(
                collection_name=collection, query_vector=(dense_name, dense_vector),
                query_filter=query_filter, limit=limit, with_payload=True,
            )
        except Exception as exc:
            log.warning("Dense search failed for %s: %s", collection, exc)
            ranked[f"{collection}_dense"] = []
        try:
            ranked[f"{collection}_bm25"] = self.client.search(
                collection_name=collection,
                query_vector=models.NamedSparseVector(name="bm25", vector=sparse),
                query_filter=query_filter, limit=limit, with_payload=True,
            )
        except Exception as exc:
            log.warning("Sparse search failed for %s: %s", collection, exc)
            ranked[f"{collection}_bm25"] = []
        return self.rrf_fusion(ranked, weights={
            f"{collection}_dense": dense_weight,
            f"{collection}_bm25": sparse_weight,
        })

    def search(self, video_id: str, query: str, limit: int = 5,
               weights: dict[str, float] | None = None) -> list[dict]:
        if not video_id or not query or not query.strip():
            return []
        limit = max(1, min(int(limit), 100))
        dense_text = self._text_vector(query)
        dense_clip = self._clip_vector(query)
        sparse = self._sparse_vector(query)
        channel_weights = weights or {"frames": 1.2, "transcripts": 1.0, "ocr": 1.0, "actions": 1.4}
        channel_specs = {
            "frames": ("video_frames", "clip", dense_clip, 1.2, 1.0),
            "transcripts": ("video_transcripts", "dense", dense_text, 1.0, 1.2),
            "ocr": ("video_ocr", "dense", dense_text, 1.0, 1.2),
            "actions": ("video_scene_semantics", "dense", dense_text, 1.0, 1.2),
        }
        channel_results = {}
        for channel, (collection, vector_name, dense, dense_weight, sparse_weight) in channel_specs.items():
            if channel_weights.get(channel, 1.0) <= 0:
                channel_results[channel] = []
                continue
            channel_results[channel] = self._search_collection(
                collection, vector_name, dense, sparse, video_id, limit * 2,
                dense_weight, sparse_weight,
            )
        return self.rrf_fusion(channel_results, weights=channel_weights)[:limit]

    def get_temporal_context(self, video_id: str, center_timestamp: float,
                             window_seconds: float = 20.0) -> list[dict]:
        models = self.models
        start = max(0.0, float(center_timestamp) - max(0.0, window_seconds))
        end = float(center_timestamp) + max(0.0, window_seconds)
        entries: list[dict] = []
        specs = [
            ("video_frames", "timestamp", "visual"),
            ("video_transcripts", "start_time", "transcript"),
            ("video_ocr", "timestamp", "ocr"),
            ("video_scene_semantics", "start_time", "action"),
        ]
        for collection, time_field, kind in specs:
            try:
                points, _ = self.client.scroll(
                    collection_name=collection,
                    scroll_filter=models.Filter(must=[
                        models.FieldCondition(key="video_id", match=models.MatchValue(value=video_id)),
                        models.FieldCondition(key=time_field, range=models.Range(gte=start, lte=end)),
                    ]),
                    limit=500, with_payload=True, with_vectors=False,
                )
                for point in points:
                    payload = point.payload or {}
                    timestamp = float(payload.get(time_field, 0.0))
                    if kind == "visual":
                        text = ", ".join(filter(None, [
                            str(payload.get("detected_objects") or ""),
                            str(payload.get("ocr_text") or ""),
                        ]))
                    else:
                        text = str(payload.get("text") or payload.get("semantic_text") or "")
                    if text.strip():
                        entries.append({"timestamp": timestamp, "type": kind, "text": text.strip()[:2000]})
            except Exception as exc:
                log.warning("Temporal context read failed (%s): %s", collection, exc)
        unique = {}
        for entry in entries:
            key = (round(entry["timestamp"], 1), entry["type"], entry["text"][:100])
            unique[key] = entry
        return sorted(unique.values(), key=lambda item: item["timestamp"])

    def close(self):
        self.client.close()
