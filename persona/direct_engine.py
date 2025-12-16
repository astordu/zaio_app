
from typing import Dict, Any
from llm.client import LLMClient


class DirectEngine:
    """L-Engine 直答人格：用户问明确问题时，尽量直接、高效地回答。"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def answer(self, text: str, user_state: Dict[str, Any]) -> str:
        payload = {
            "question": text,
            "user_state": user_state,
        }
        reply = self.llm.call_llm("persona_direct", payload, temperature=0.3)
        if not reply:
            reply = "这个问题我先给你一个大致方向，如果你愿意，我们可以再慢慢细化。"
        return reply
