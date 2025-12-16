# persona/sum_engine.py

import json
from typing import Dict, Any, List
from llm.client import LLMClient


class SumEngine:
    """
    在哦 · 总结人格（Sum-Engine）
    用于用户明确要求“帮我总结一下”“你整理一下重点”等场景。

    输入：
    - user_text: 用户最新一句话
    - user_state: 快照
    - talk_history: 最近对话
    - current_tree: 当前观点树（JSON）

    输出：
    - 一段自然中文，总结 T 引擎刚刚展开的观点结构。
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.ROLE = "persona_sum"

    def respond(
        self,
        user_text: str,
        user_state: Dict[str, Any],
        talk_history: List[Dict[str, Any]],
        current_tree: Dict[str, Any]
    ) -> str:
        """
        生成总结文字。
        """
        payload = {
            "user_text": user_text,
            "snapshot": user_state,
            "talk_history": talk_history,
            "current_tree": current_tree,
        }

        reply = self.llm.call_llm(
            role=self.ROLE,
            payload=payload,
            temperature=0.3,
        )

        return reply or ""

