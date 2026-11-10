"""Optional Ollama query planner; retrieval remains functional without Ollama."""
import json
import logging
from typing import Optional
import httpx
from pydantic import BaseModel, Field
from config.config import get_settings

log = logging.getLogger(__name__)


class QueryPlan(BaseModel):
    optimized_query: str = Field(min_length=1)
    target_channels: list[str] = Field(default_factory=lambda: ["frames", "transcripts", "ocr", "actions"])
    temporal_filter: Optional[dict[str, float]] = None
    reasoning: str = "Semantic multimodal retrieval"


class AgenticQueryEngine:
    def __init__(self, ollama_url: str | None = None, ollama_model: str | None = None,
                 timeout: float = 12.0):
        settings = get_settings()
        base = (ollama_url or settings.ollama_url).rstrip("/")
        self.ollama_url = base if base.endswith("/api/chat") else base + "/api/chat"
        self.model = ollama_model or settings.ollama_model
        self.timeout = timeout

    async def plan_search(self, raw_query: str) -> QueryPlan:
        if not raw_query.strip():
            return QueryPlan(optimized_query="video content")
        prompt = (
            "Return only JSON with keys optimized_query, target_channels, temporal_filter, reasoning. "
            "optimized_query must be a concise plain-English description of what to find. "
            "target_channels may contain frames, transcripts, ocr, actions. "
            "temporal_filter must be null unless explicit time bounds are present. "
            "Do not invent time bounds. User query: " + json.dumps(raw_query)
        )
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.ollama_url, json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "format": "json", "stream": False,
                    "options": {"temperature": 0},
                })
                response.raise_for_status()
                content = response.json().get("message", {}).get("content", "{}")
                plan = QueryPlan.model_validate_json(content)
                if not plan.optimized_query.strip():
                    raise ValueError("Planner returned an empty optimized query")
                return plan
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            log.warning("Query planning unavailable; using original query: %s", exc)
            return QueryPlan(optimized_query=raw_query, reasoning="Fallback: original user query")
