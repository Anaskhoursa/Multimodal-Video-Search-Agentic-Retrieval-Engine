# core/agent.py
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import httpx
import json

log = logging.getLogger(__name__)

class QueryPlan(BaseModel):
    optimized_query: str = Field(..., description="The unified semantic search string extracted for the embedding models")
    target_channels: List[str] = Field(default=["frames", "transcripts", "ocr"], description="Channels to query based on intent")
    temporal_filter: Optional[Dict[str, float]] = Field(default=None, description="Optional start_time / end_time boundaries if inferred")
    reasoning: str = Field(..., description="Brief explanation of search execution strategy")

class AgenticQueryEngine:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = f"{ollama_url}/api/chat"
        
    async def plan_search(self, raw_query: str) -> QueryPlan:
        """Uses a local LLM to structure unstructured user request into search configurations"""
        system_prompt = (
            "You are an expert video retrieval routing agent. Analyze the user's question and "
            "determine how to query our multimodal video database vector layers.\n"
            "CRITICAL: The 'optimized_query' must be a plain English description of the visual "
            "or textual items to find (e.g., 'a security camera mounted on a wall'). Do NOT use boolean "
            "operators, code syntax, or channel prefixes like & or ~ inside the optimized_query.\n\n"
            "Output must strictly be valid JSON matching this schema:\n"
            "{'optimized_query': string, 'target_channels': list of strings from ['frames', 'transcripts', 'ocr'], "
            "'temporal_filter': null or {'start': float, 'end': float}, 'reasoning': string}"
        )
        
        user_prompt = f"Deconstruct this video search query: '{raw_query}'"
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.ollama_url,
                    json={
                        "model": "llama3", # Or your preferred local model
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "format": "json",
                        "stream": False
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    content = data.get("message", {}).get("content", "{}")
                    parsed = json.loads(content)
                    return QueryPlan(**parsed)
        except Exception as e:
            log.warning(f"Agent planning failed, falling back to default route. Error: {e}")
            
        # Fallback implementation if LLM is offline
        return QueryPlan(
            optimized_query=raw_query,
            target_channels=["frames", "transcripts", "ocr"],
            reasoning="Fallback strategy executed due to connection timeout."
        )