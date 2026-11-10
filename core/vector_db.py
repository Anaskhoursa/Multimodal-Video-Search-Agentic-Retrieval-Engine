"""Qdrant collection lifecycle and write helpers."""
import logging
import time
from typing import Iterable
from qdrant_client import QdrantClient, models
from config.config import get_settings

log = logging.getLogger(__name__)

COLLECTIONS = {
    "video_frames": ("clip", 512),
    "video_transcripts": ("dense", 384),
    "video_ocr": ("dense", 384),
    "video_scene_semantics": ("dense", 384),
}


class VectorDBManager:
    def __init__(self, url: str | None = None, host: str | None = None, port: int = 6333):
        settings = get_settings()
        self.client = QdrantClient(url=url or settings.qdrant_url) if (url or not host) else QdrantClient(host=host, port=port)

    def create_collections_if_not_exists(self) -> None:
        # Compose starts dependencies before the API, but Qdrant may need a few
        # seconds to finish booting. Retry readiness instead of failing startup.
        last_error = None
        for attempt in range(30):
            try:
                existing = {item.name for item in self.client.get_collections().collections}
                break
            except Exception as exc:
                last_error = exc
                if attempt == 29:
                    raise RuntimeError("Could not connect to Qdrant after 60 seconds") from last_error
                time.sleep(2)

        for name, (dense_name, dimension) in COLLECTIONS.items():
            if name not in existing:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config={dense_name: models.VectorParams(
                        size=dimension, distance=models.Distance.COSINE
                    )},
                    sparse_vectors_config={"bm25": models.SparseVectorParams()},
                )
            # Filter fields used in every retrieval/temporal query.
            for field, schema in (
                ("video_id", models.PayloadSchemaType.KEYWORD),
                ("timestamp", models.PayloadSchemaType.FLOAT),
                ("start_time", models.PayloadSchemaType.FLOAT),
            ):
                try:
                    self.client.create_payload_index(
                        collection_name=name, field_name=field, field_schema=schema,
                    )
                except Exception as exc:
                    # Existing indexes are expected on subsequent startups.
                    log.debug("Payload index %s.%s already exists or is unsupported: %s", name, field, exc)

    def upsert(self, collection: str, points: Iterable[models.PointStruct]) -> None:
        batch = list(points)
        if batch:
            self.client.upsert(collection_name=collection, points=batch, wait=True)

    def upsert_frame_embeddings(self, points):
        self.upsert("video_frames", points)

    def upsert_transcript_embeddings(self, points):
        self.upsert("video_transcripts", points)

    def upsert_ocr_embeddings(self, points):
        self.upsert("video_ocr", points)

    def upsert_scene_semantics_embeddings(self, points):
        self.upsert("video_scene_semantics", points)

    def close(self) -> None:
        self.client.close()
