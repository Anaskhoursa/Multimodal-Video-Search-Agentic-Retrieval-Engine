import unittest
from types import SimpleNamespace
from core.retriever import MultimodalRetriever


class RetrieverFusionTests(unittest.TestCase):
    def test_fuses_raw_points_at_same_video_timestamp(self):
        first = SimpleNamespace(
            payload={"video_id": "video-a", "timestamp": 12.04, "detected_objects": "person"},
            score=0.9,
        )
        second = SimpleNamespace(
            payload={"video_id": "video-a", "start_time": 12.02, "text": "hello"},
            score=0.8,
        )
        fused = MultimodalRetriever.rrf_fusion({"visual": [first], "speech": [second]})
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["timestamp"], 12.0)
        self.assertEqual(fused[0]["video_id"], "video-a")
        self.assertEqual(len(fused[0]["evidence"]), 2)

    def test_fuses_channel_results_without_losing_evidence(self):
        channel = [{
            "video_id": "video-a", "timestamp": 5.0, "rrf_score": 0.02,
            "matches": [{"stream": "dense"}],
            "evidence": [{"stream": "transcript", "text": "a person speaks"}],
        }]
        fused = MultimodalRetriever.rrf_fusion({"transcripts": channel})
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["evidence"][0]["text"], "a person speaks")

    def test_ignores_points_without_video_id(self):
        point = SimpleNamespace(payload={"timestamp": 1.0}, score=1.0)
        self.assertEqual(MultimodalRetriever.rrf_fusion({"x": [point]}), [])


if __name__ == "__main__":
    unittest.main()
