import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from core.chat_engine import AgenticVideoChatEngine


class FakeRetriever:
    def __init__(self, hits=None, entries=None):
        self.hits = hits or []
        self.entries = entries or []
        self.calls = []

    def search(self, video_id, query, limit):
        self.calls.append((video_id, query, limit))
        return self.hits

    def get_temporal_context(self, video_id, center_timestamp, window_seconds):
        return self.entries


class FakePlanner:
    async def plan_search(self, message):
        return SimpleNamespace(
            optimized_query="optimized query",
            model_dump=lambda: {"optimized_query": "optimized query"},
        )


class ChatEngineTests(unittest.TestCase):
    def test_passes_video_id_and_uses_hit_evidence(self):
        retriever = FakeRetriever(hits=[{
            "video_id": "video-1", "timestamp": 8.0,
            "evidence": [{"stream": "transcript", "text": "a car turns left"}],
        }])
        engine = AgenticVideoChatEngine(retriever, agent_engine=FakePlanner())
        engine._ask_ollama = AsyncMock(return_value="At [0m08.0s], a car turns left.")
        result = asyncio.run(engine.execute_chat("video-1", "what happens?", [], {}))
        self.assertEqual(retriever.calls[0][0], "video-1")
        self.assertEqual(retriever.calls[0][1], "optimized query")
        self.assertIn("car turns left", result["response"])
        engine._ask_ollama.assert_awaited_once()

    def test_does_not_call_llm_without_evidence(self):
        retriever = FakeRetriever()
        engine = AgenticVideoChatEngine(retriever)
        engine._ask_ollama = AsyncMock(return_value="hallucinated")
        result = asyncio.run(engine.execute_chat("video-2", "what is there?", [], {}))
        self.assertIn("couldn't find indexed evidence", result["response"])
        engine._ask_ollama.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
