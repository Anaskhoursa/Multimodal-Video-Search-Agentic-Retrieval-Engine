# services/chat_engine.py
import json
import httpx
import logging
from typing import TypedDict, List, Dict, Optional, Literal
from langgraph.graph import StateGraph, START, END
from core.retriever import MultimodalRetriever
log = logging.getLogger(__name__)

# ── State Schema ─────────────────────────────────────────────────────────────

class VideoChatState(TypedDict):
    video_id: str
    user_message: str
    chat_history: List[Dict[str, str]]

    # Session memory (persists across turns via session dict)
    active_chunk: Optional[Dict]
    conversation_mode: Literal["idle", "discussing_chunk", "awaiting_pick"]
    options_for_user: List[Dict]

    # Per-turn
    query_intent: Literal["new_search", "follow_up", "temporal_follow_up", "clarification", "summary", "out_of_scope"]
    retrieved_hits: List[Dict]
    confidence_mode: Literal["direct", "ask_user"]
    timeline_context: str
    response_to_user: str
    needs_user_input: bool


# ── Engine ────────────────────────────────────────────────────────────────────

class AgenticVideoChatEngine:
    def __init__(
        self,
        retriever : MultimodalRetriever,
        ollama_url: str = "http://localhost:11434/api/chat",
        ollama_model: str = "llama3",
        context_window_seconds: float = 20.0,
        follow_up_window_seconds: float = 45.0,
        confidence_gap_ratio: float = 0.4,
        top_k: int = 5,
    ):
        self.retriever = retriever
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.context_window_seconds = context_window_seconds
        self.follow_up_window_seconds = follow_up_window_seconds
        self.confidence_gap_ratio = confidence_gap_ratio
        self.top_k = top_k
        self.graph = self._build_graph()

    # ─────────────────────────────────────────────────────────────────────────
    # SHARED LLM CALL
    # ─────────────────────────────────────────────────────────────────────────

    async def _call_llm(self, messages: List[Dict], temperature: float = 0.0) -> str:
        """Single point of contact for all Ollama calls."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature}
                }
            )
            if r.status_code == 200:
                return r.json().get("message", {}).get("content", "").strip()
            log.error(f"Ollama returned {r.status_code}: {r.text}")
            return ""

    # ─────────────────────────────────────────────────────────────────────────
    # NODE 1 — LLM Intent Classifier
    # ─────────────────────────────────────────────────────────────────────────

    async def analyze_intent_node(self, state: VideoChatState) -> Dict:
        """
        Uses the LLM to classify the user's intent.
        Temperature=0 for deterministic classification.
        Returns one of: new_search | follow_up | temporal_follow_up | 
                        clarification | summary | out_of_scope
        """
        conversation_mode = state.get("conversation_mode", "idle")
        active_chunk = state.get("active_chunk")
        chat_history = state.get("chat_history", [])

        # Build a compact history summary for the classifier
        # Only last 4 turns to keep the classifier prompt short
        recent_history = chat_history[-4:] if len(chat_history) > 4 else chat_history
        history_text = "\n".join(
            f"{t['role'].upper()}: {t['content'][:120]}"
            for t in recent_history
        ) or "No prior conversation."

        active_chunk_text = (
            f"Timestamp {active_chunk['timestamp']:.1f}s"
            if active_chunk else "None"
        )

        classifier_prompt = f"""You are a routing classifier for a video Q&A assistant.

CURRENT SESSION STATE:
- Active video chunk being discussed: {active_chunk_text}
- Conversation mode: {conversation_mode}

RECENT CONVERSATION:
{history_text}

NEW USER MESSAGE:
"{state['user_message']}"

Classify the user's intent into EXACTLY ONE of these categories:

- new_search: User wants to find a new moment, topic, or event in the video. No active chunk context needed.
- follow_up: User is asking more about the currently active chunk (explaining, elaborating, deeper questions about same moment).
- temporal_follow_up: User wants to know what happened before or after the current moment in the video timeline.
- clarification: User is selecting one of the numbered options the assistant just presented.
- summary: User wants a high-level overview or summary of the whole video.
- out_of_scope: User is asking something unrelated to the video content.

Rules:
- If there is no active chunk, follow_up and temporal_follow_up are impossible — classify as new_search instead.
- If conversation_mode is "awaiting_pick", strongly consider clarification.
- Output ONLY the category name. No explanation. No punctuation.

Category:"""

        intent = await self._call_llm(
            [{"role": "user", "content": classifier_prompt}],
            temperature=0.0
        )

        # Sanitize — LLM sometimes adds punctuation or extra words
        valid_intents = {"new_search", "follow_up", "temporal_follow_up", "clarification", "summary", "out_of_scope"}
        intent = intent.lower().strip().rstrip(".")
        if intent not in valid_intents:
            log.warning(f"LLM returned unexpected intent '{intent}', defaulting to new_search")
            intent = "new_search"

        log.info(f"Intent classified as: {intent}")
        return {"query_intent": intent}

    # ─────────────────────────────────────────────────────────────────────────
    # NODE 2 — Retrieval
    # ─────────────────────────────────────────────────────────────────────────

    async def retrieve_node(self, state: VideoChatState) -> Dict:
        hits = self.retriever.search(
            query=state["user_message"],
            video_id=state["video_id"],
            limit=self.top_k
        )

        if not hits:
            return {
                "retrieved_hits": [],
                "confidence_mode": "direct",
                "response_to_user": "I couldn't find anything relevant to that in this video.",
                "needs_user_input": False
            }

        # Confidence: is the top result clearly dominant?
        top_score = hits[0]["rrf_score"]
        second_score = hits[1]["rrf_score"] if len(hits) > 1 else 0.0
        gap = top_score - second_score
        confidence_mode = "direct" if gap > (top_score * self.confidence_gap_ratio) else "ask_user"

        log.info(f"Retrieved {len(hits)} hits. Top score: {top_score:.3f}, gap: {gap:.3f}, mode: {confidence_mode}")
        return {"retrieved_hits": hits, "confidence_mode": confidence_mode}

    # ─────────────────────────────────────────────────────────────────────────
    # NODE 3a — Auto-select top hit (high confidence)
    # ─────────────────────────────────────────────────────────────────────────

    async def auto_select_node(self, state: VideoChatState) -> Dict:
        top_hit = state["retrieved_hits"][0]
        context_entries = self.retriever.get_temporal_context(
            video_id=state["video_id"],
            center_timestamp=top_hit["timestamp"],
            window_seconds=self.context_window_seconds
        )
        return {
            "active_chunk": top_hit,
            "conversation_mode": "discussing_chunk",
            "timeline_context": self._format_context(context_entries),
            "needs_user_input": False
        }

    # ─────────────────────────────────────────────────────────────────────────
    # NODE 3b — Present options (low confidence)
    # ─────────────────────────────────────────────────────────────────────────

    async def present_options_node(self, state: VideoChatState) -> Dict:
        hits = state["retrieved_hits"][:3]
        options = []

        for i, hit in enumerate(hits):
            ts = hit["timestamp"]
            preview_entries = self.retriever.get_temporal_context(
                video_id=state["video_id"],
                center_timestamp=ts,
                window_seconds=5.0
            )
            # Best preview: transcript > OCR > visual label
            preview = next(
                (e["text"] for e in preview_entries if e.get("type") == "transcript" and e.get("text")),
                next(
                    (e["text"] for e in preview_entries if e.get("type") == "ocr" and e.get("text")),
                    f"Visual scene at {ts:.1f}s"
                )
            )
            options.append({
                "index": i + 1,
                "timestamp": ts,
                "preview": preview[:120],
                "rrf_score": hit["rrf_score"]
            })

        lines = ["I found a few relevant moments — which one did you mean?\n"]
        for opt in options:
            mins = int(opt["timestamp"] // 60)
            secs = opt["timestamp"] % 60
            lines.append(f"[{opt['index']}] At {mins}m{secs:.0f}s — \"{opt['preview']}\"")

        return {
            "options_for_user": options,
            "conversation_mode": "awaiting_pick",
            "response_to_user": "\n".join(lines),
            "needs_user_input": True,
            "timeline_context": ""
        }

    # ─────────────────────────────────────────────────────────────────────────
    # NODE 3c — Handle clarification pick
    # ─────────────────────────────────────────────────────────────────────────

    async def handle_clarification_node(self, state: VideoChatState) -> Dict:
        """LLM extracts which option the user picked from natural language."""
        options = state.get("options_for_user", [])
        if not options:
            return {"query_intent": "new_search"}  # guard: no options to pick from

        options_text = "\n".join(
            f"{o['index']}. At {o['timestamp']:.1f}s: \"{o['preview']}\""
            for o in options
        )

        pick_prompt = f"""The assistant presented these video moments to the user:
{options_text}

The user replied: "{state['user_message']}"

Which option number did the user select? 
Reply with ONLY a single digit (1, 2, or 3).
If the user didn't select any option and asked something else entirely, reply with 0."""

        pick = await self._call_llm(
            [{"role": "user", "content": pick_prompt}],
            temperature=0.0
        )

        try:
            pick_index = int(pick.strip()[0])
        except (ValueError, IndexError):
            pick_index = 0

        if pick_index == 0 or pick_index > len(options):
            # User didn't pick — treat as new search
            return {
                "query_intent": "new_search",
                "conversation_mode": "idle",
                "options_for_user": []
            }

        chosen_option = options[pick_index - 1]
        chosen_hit = state["retrieved_hits"][pick_index - 1]

        context_entries = self.retriever.get_temporal_context(
            video_id=state["video_id"],
            center_timestamp=chosen_option["timestamp"],
            window_seconds=self.context_window_seconds
        )

        return {
            "active_chunk": chosen_hit,
            "conversation_mode": "discussing_chunk",
            "timeline_context": self._format_context(context_entries),
            "options_for_user": [],
            "needs_user_input": False
        }

    # ─────────────────────────────────────────────────────────────────────────
    # NODE 4a — Follow-up (same chunk, wider window)
    # ─────────────────────────────────────────────────────────────────────────

    async def follow_up_node(self, state: VideoChatState) -> Dict:
        active_chunk = state["active_chunk"]
        context_entries = self.retriever.get_temporal_context(
            video_id=state["video_id"],
            center_timestamp=active_chunk["timestamp"],
            window_seconds=self.follow_up_window_seconds
        )
        return {
            "timeline_context": self._format_context(context_entries),
            "needs_user_input": False
        }

    # ─────────────────────────────────────────────────────────────────────────
    # NODE 4b — Temporal follow-up (shift window forward/backward)
    # ─────────────────────────────────────────────────────────────────────────

    async def temporal_follow_up_node(self, state: VideoChatState) -> Dict:
        """LLM extracts direction and time offset from natural language."""
        active_chunk = state["active_chunk"]
        center_ts = active_chunk["timestamp"]

        shift_prompt = f"""The user is currently at timestamp {center_ts:.1f}s in a video.
They said: "{state['user_message']}"

Extract:
1. direction: "forward" or "backward" in the video timeline
2. offset_seconds: how many seconds to shift (integer, default 60 if unspecified)

Reply in this exact JSON format only:
{{"direction": "forward", "offset_seconds": 60}}"""

        raw = await self._call_llm(
            [{"role": "user", "content": shift_prompt}],
            temperature=0.0
        )

        try:
            parsed = json.loads(raw.strip())
            direction = parsed.get("direction", "forward")
            offset = float(parsed.get("offset_seconds", 60))
        except (json.JSONDecodeError, KeyError):
            log.warning(f"Could not parse temporal shift from: {raw}")
            direction, offset = "forward", 60.0

        new_center = center_ts + offset if direction == "forward" else max(0.0, center_ts - offset)
        log.info(f"Temporal shift: {direction} {offset}s → new center {new_center:.1f}s")

        context_entries = self.retriever.get_temporal_context(
            video_id=state["video_id"],
            center_timestamp=new_center,
            window_seconds=self.context_window_seconds
        )

        new_active_chunk = {**active_chunk, "timestamp": new_center}
        return {
            "active_chunk": new_active_chunk,
            "conversation_mode": "discussing_chunk",
            "timeline_context": self._format_context(context_entries),
            "needs_user_input": False
        }

    # ─────────────────────────────────────────────────────────────────────────
    # NODE 5 — Summary context (whole-video sparse timeline)
    # ─────────────────────────────────────────────────────────────────────────

    async def build_summary_context_node(self, state: VideoChatState) -> Dict:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        points, _ = self.retriever.client.scroll(
            collection_name="video_transcripts",
            scroll_filter=Filter(must=[
                FieldCondition(key="video_id", match=MatchValue(value=state["video_id"]))
            ]),
            limit=1000,
            with_payload=True
        )

        entries = sorted(
            [{"ts": p.payload.get("start_time", 0), "text": p.payload.get("text", "")} for p in points],
            key=lambda x: x["ts"]
        )

        # One entry per 60 seconds
        sampled, last_ts = [], -60
        for e in entries:
            if e["ts"] - last_ts >= 60 and e["text"]:
                mins = int(e["ts"] // 60)
                secs = e["ts"] % 60
                sampled.append(f"[{mins}m{secs:.0f}s] {e['text'][:150]}")
                last_ts = e["ts"]

        timeline_str = "### FULL VIDEO TIMELINE (1 entry/minute)\n" + "\n".join(sampled)
        return {"timeline_context": timeline_str, "needs_user_input": False}

    # ─────────────────────────────────────────────────────────────────────────
    # NODE 6 — Out of scope
    # ─────────────────────────────────────────────────────────────────────────

    async def out_of_scope_node(self, state: VideoChatState) -> Dict:
        return {
            "response_to_user": "I can only answer questions about the video content. Could you rephrase your question about something in the video?",
            "needs_user_input": False,
            "timeline_context": ""
        }

    # ─────────────────────────────────────────────────────────────────────────
    # NODE 7 — LLM Answer
    # ─────────────────────────────────────────────────────────────────────────

    async def llm_answer_node(self, state: VideoChatState) -> Dict:
        system_prompt = (
            "You are a Video Q&A Assistant. Answer the user's question using ONLY "
            "the provided timeline context from the video.\n\n"
            "Rules:\n"
            "1. Always cite specific timestamps (e.g. 'At 4m32s...').\n"
            "2. If the answer isn't in the context, say so clearly.\n"
            "3. Be concise and direct.\n\n"
            f"{state['timeline_context']}"
        )

        messages = [{"role": "system", "content": system_prompt}]
        for turn in state.get("chat_history", []):
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": state["user_message"]})

        response = await self._call_llm(messages, temperature=0.3)
        return {
            "response_to_user": response or "I couldn't generate a response. Please try again.",
            "needs_user_input": False
        }

    # ─────────────────────────────────────────────────────────────────────────
    # ROUTING
    # ─────────────────────────────────────────────────────────────────────────

    def _route_after_intent(self, state: VideoChatState) -> str:
        intent = state["query_intent"]
        routes = {
            "new_search":          "retrieve",
            "follow_up":           "follow_up",
            "temporal_follow_up":  "temporal_follow_up",
            "clarification":       "handle_clarification",
            "summary":             "build_summary_context",
            "out_of_scope":        "out_of_scope",
        }
        return routes.get(intent, "retrieve")

    def _route_after_retrieval(self, state: VideoChatState) -> str:
        if not state["retrieved_hits"]:
            return END
        return "auto_select" if state["confidence_mode"] == "direct" else "present_options"

    def _route_to_answer_or_end(self, state: VideoChatState) -> str:
        if state.get("needs_user_input"):
            return END
        # If clarification re-routed to new_search, go back to retrieve
        if state.get("query_intent") == "new_search":
            return "retrieve"
        return "llm_answer"

    # ─────────────────────────────────────────────────────────────────────────
    # GRAPH ASSEMBLY
    # ─────────────────────────────────────────────────────────────────────────

    def _build_graph(self):
        g = StateGraph(VideoChatState)

        g.add_node("analyze_intent",         self.analyze_intent_node)
        g.add_node("retrieve",               self.retrieve_node)
        g.add_node("auto_select",            self.auto_select_node)
        g.add_node("present_options",        self.present_options_node)
        g.add_node("handle_clarification",   self.handle_clarification_node)
        g.add_node("follow_up",              self.follow_up_node)
        g.add_node("temporal_follow_up",     self.temporal_follow_up_node)
        g.add_node("build_summary_context",  self.build_summary_context_node)
        g.add_node("out_of_scope",           self.out_of_scope_node)
        g.add_node("llm_answer",             self.llm_answer_node)

        g.add_edge(START, "analyze_intent")
        g.add_conditional_edges("analyze_intent",       self._route_after_intent)
        g.add_conditional_edges("retrieve",             self._route_after_retrieval)
        g.add_conditional_edges("handle_clarification", self._route_to_answer_or_end)
        g.add_conditional_edges("present_options",      self._route_to_answer_or_end)

        for node in ["auto_select", "follow_up", "temporal_follow_up", "build_summary_context"]:
            g.add_edge(node, "llm_answer")

        g.add_edge("out_of_scope", END)
        g.add_edge("llm_answer",   END)

        return g.compile()

    # ─────────────────────────────────────────────────────────────────────────
    # HELPER
    # ─────────────────────────────────────────────────────────────────────────

    def _format_context(self, entries: List[Dict]) -> str:
        lines = []
        for e in sorted(entries, key=lambda x: x["timestamp"]):
            ts = e["timestamp"]
            mins, secs = int(ts // 60), ts % 60
            label = f"[{mins}m{secs:.0f}s]"
            if e["type"] == "transcript":
                lines.append(f"{label} [SPOKEN]: \"{e['text']}\"")
            elif e["type"] == "ocr":
                lines.append(f"{label} [ON-SCREEN]: \"{e['text']}\"")
            elif e["type"] == "visual":
                parts = [f"{label} [VISUAL]"]
                if e.get("objects"):
                    parts.append(f"Objects: {', '.join(e['objects'])}")
                if e.get("ocr_text"):
                    parts.append(f"Text: \"{e['ocr_text']}\"")
                lines.append(" | ".join(parts))
        return "### TIMELINE CONTEXT\n" + "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # ─────────────────────────────────────────────────────────────────────────

    async def execute_chat(
        self,
        video_id: str,
        message: str,
        history: List[Dict],
        session: Dict = None
    ) -> Dict:
        session = session or {}

        initial_state: VideoChatState = {
            "video_id":           video_id,
            "user_message":       message,
            "chat_history":       history,
            # Restored from session
            "active_chunk":       session.get("active_chunk"),
            "conversation_mode":  session.get("conversation_mode", "idle"),
            "options_for_user":   session.get("options_for_user", []),
            # Fresh per turn
            "query_intent":       "new_search",
            "retrieved_hits":     [],
            "confidence_mode":    "direct",
            "timeline_context":   "",
            "response_to_user":   "",
            "needs_user_input":   False,
        }

        output = await self.graph.ainvoke(initial_state)

        return {
            "response":         output["response_to_user"],
            "needs_user_input": output["needs_user_input"],
            "options":          output.get("options_for_user", []),
            # Caller stores this and passes it back next turn
            "session": {
                "active_chunk":      output.get("active_chunk"),
                "conversation_mode": output.get("conversation_mode", "idle"),
                "options_for_user":  output.get("options_for_user", []),
            }
        }