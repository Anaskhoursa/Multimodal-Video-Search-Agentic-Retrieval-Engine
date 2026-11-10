from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SparseVectorParams
from typing import List

class VectorDBManager:
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)

    def create_collections_if_not_exists(self):
        # 1. Visual Frames Collection (CLIP uses 512 dimensions)
        try:
            self.client.get_collection(collection_name="video_frames")
        except Exception:
            print("Creating Qdrant collection: video_frames")
            self.client.create_collection(
                collection_name="video_frames",
                vectors_config={"clip":VectorParams(size=512, distance=Distance.COSINE)}, 
                sparse_vectors_config={
                    "bm25": SparseVectorParams()
                }
            )

        # 2. Audio Transcripts Collection (all-MiniLM-L6-v2 uses 384 dimensions)
        try:
            self.client.get_collection(collection_name="video_transcripts")
        except Exception:
            print("Creating Qdrant collection: video_transcripts")
            self.client.create_collection(
                collection_name="video_transcripts",
                vectors_config={
                    "dense": VectorParams(
                        size=384,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "bm25": SparseVectorParams()
                }
            )

        # 3. ADDED: On-Screen OCR Text Collection (Text Embedder - 384 dims)
        try:
            self.client.get_collection(collection_name="video_ocr")
        except Exception:
            print("Creating Qdrant collection: video_ocr")
            self.client.create_collection(
                collection_name="video_ocr",
                vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
                sparse_vectors_config={
                    "bm25": SparseVectorParams()
                }
            )

        # 4. Scene Semantics Collection
        # (Objects + Actions + Future Semantic Metadata)

        try:
            self.client.get_collection(collection_name="video_scene_semantics")
        except Exception:
            print("Creating Qdrant collection: video_scene_semantics")

            self.client.create_collection(
                collection_name="video_scene_semantics",
                vectors_config={
                    "dense": VectorParams(
                        size=384,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "bm25": SparseVectorParams()
                }
            )

    def upsert_frame_embeddings(self, points: List[PointStruct]):
        self.client.upsert(collection_name="video_frames", points=points, wait=True)

    def upsert_transcript_embeddings(self, points: List[PointStruct]):
        self.client.upsert(collection_name="video_transcripts", points=points, wait=True)
        
    def upsert_ocr_embeddings(self, points: List[PointStruct]):
        self.client.upsert(collection_name="video_ocr", points=points)
        
    def upsert_scene_semantics_embeddings(self, points: List[PointStruct]):
        self.client.upsert(
            collection_name="video_scene_semantics",
            points=points,
            wait=True
        )