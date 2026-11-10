import tempfile
import unittest
from pathlib import Path
from core.ingestion import VideoIngestionPipeline


class IngestionWorkspaceTests(unittest.TestCase):
    def test_workspace_creates_expected_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = VideoIngestionPipeline(tmp, max_chunk_seconds=20)
            paths = pipeline.create_video_workspace("sample-video")
            for key in ("workspace", "chunks", "frames", "audio"):
                self.assertTrue(paths[key].is_dir(), key)
            self.assertEqual(pipeline.max_chunk_seconds, 20.0)


if __name__ == "__main__":
    unittest.main()
