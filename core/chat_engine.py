# core/chat_engine.py
import logging
import json
from typing import List, Dict, Any, Optional
import httpx

log = logging.getLogger(__name__)

class AgenticVideoChatEngine:
    def __init__(
        self,
        retriever,
        agent_engine,
        ollama_url: str = "http://localhost:11434/api/chat",
        ollama_model: str = "llama3",
        context_window_seconds: float = 20.0,
        follow_up_window_seconds: float = 45.0,
        confidence_gap_ratio: float = 0.4,
        top_k: int = 5
    ):
        self.retriever = retriever
        self.agent_engine = agent_engine
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.context_window_seconds = context_window_seconds
        self.follow_up_window_seconds = follow_up_window_seconds
        self.confidence_gap_ratio = confidence_gap_ratio
        self.top_k = top_k

    async def _fetch_video_context(self, video_id: str, query: str, active_chunk: Optional[Dict] = None) -> str:
        """
        Retrieves relevant multimodal records via vector search or extracts localized temporal context windows.
        """
        context_lines = []
        
        try:
            # If an active chunk timestamp exists in session state, prioritize fetching temporal surroundings
            if active_chunk and "timestamp" in active_chunk:
                target_ts = active_chunk["timestamp"]
                start_win = max(0.0, target_ts - self.context_window_seconds)
                end_win = target_ts + self.context_window_seconds
                
                context_lines.append(f"### LOCAL TIMELINE WINDOW NEAR {target_ts}s (Range: {start_win}s - {end_win}s)")
                # Fetch points from retriever matching temporal bounds
                raw_results = self.retriever.search(query=query, limit=self.top_k)
                for match in raw_results:
                    if match.get("video_id") == video_id:
                        ts = match.get("timestamp", 0.0)
                        if start_win <= ts <= end_win:
                            for m in match.get("matches", []):
                                context_lines.append(f"[{ts}s] [{m['stream'].upper()}]: {m.get('matched_speech') or m.get('matched_text') or m.get('detected_objects')}")
            else:
                # Standard semantic multi-channel lookup
                raw_results = self.retriever.search(query=query, limit=self.top_k)
                for match in raw_results:
                    if match.get("video_id") == video_id:
                        ts = match.get("timestamp", 0.0)
                        for m in match.get("matches", []):
                            content = m.get('matched_speech') or m.get('matched_text') or m.get('detected_objects')
                            context_lines.append(f"[{ts}s] [{m['stream'].upper()}]: {content}")
        except Exception as e:
            log.warning(f"Failed to query retriever context: {e}")
            
        if not context_lines:
            return "No specific metadata matches found for this query inside the video."
            
        return "\n".join(context_lines)

    async def execute_chat(
        self,
        video_id: str,
        message: str,
        history: List[Dict[str, str]],
        session: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes a conversational chat step, manages session context, and communicates with Llama 3.
        """
        # 1. Retrieve session context variables
        active_chunk = session.get("active_chunk")
        conversation_mode = session.get("conversation_mode", "general")

        # 2. Gather multimodal timeline track context
        timeline_context = await self._fetch_video_context(video_id, message, active_chunk)

        # 3. Construct System Prompt Instructions
        system_prompt = (
            "You are an expert AI Video Intelligence Assistant engaged in a conversational dialogue with a user. "
            "Your job is to answer questions about the video using the provided multimodal track metadata.\n\n"
            "RULES:\n"
            "1. Ground your answers strictly in the provided evidence. Cite timestamp markers (e.g., [12.5s]).\n"
            "2. If the user's question references a follow-up action or specific moment, anchor to that context.\n"
            "3. If information is missing, clearly state what is missing.\n\n"
            f"Video Context Data:\n{timeline_context}"
        )

        # 4. Assemble full message payload for Ollama
        messages = [{"role": "system", "content": system_prompt}]
        for turn in history:
            messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})
        messages.append({"role": "user", "content": message})

        # 5. Call Local Llama 3 Model via HTTPX
        response_text = "I'm sorry, I encountered an error communicating with the local LLM reasoning model."
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.ollama_url,
                    json={
                        "model": self.ollama_model,
                        "messages": messages,
                        "stream": False
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    response_text = data.get("message", {}).get("content", "")
        except Exception as e:
            log.error(f"Ollama connection error during chat execution: {e}")

        # 6. Parse timestamps out of response to update session state if available
        import re
        timestamps_found = re.findall(r'\[([\d\.]+)s\]', response_text)
        new_active_chunk = active_chunk
        if timestamps_found:
            latest_ts = float(timestamps_found[-1])
            new_active_chunk = {"timestamp": latest_ts}

        updated_session = {
            "conversation_mode": conversation_mode,
            "active_chunk": new_active_chunk
        }

        return {
            "response": response_text,
            "needs_user_input": False,
            "options": [],
            "query_plan": {"target_channels": ["frames", "transcripts", "ocr"]},
            "session": updated_session
        }