from typing import Dict, Any, List
from llm.client import LLMClient


class FastEngine:
    """Q-Engine 快人格：短句、轻反馈、接住情绪，同时帮忙补全 user_state。"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def build_opening_question(self, user_profile: Dict[str, Any], snapshot: Dict[str, Any]) -> (str, Dict[str, Any]):
        """
        用快引擎（persona_fast）来生成开场白。
        把 snapshot 当作 user_state 传进去，talk_his 为空列表。
        """
        payload = {
            "mode": "opening",           # 提示一下这是“开场模式”
            "user_state": snapshot or {},# 和你系统 prompt 里写的一致
            "talk_his": [],              # 第一轮对话，还没有历史
            "user_profile": user_profile or {},  # 你以后可以在 prompt 里用
        }

        text = self.llm.call_llm("persona_fast", payload, temperature=0.6)

        if not text or not text.strip():
            # 兜底：LLM 出错或返回空的时候
            text = "我来啦～我们要不要先从你今天最有感觉的一件小事聊起？"

        meta = {"opening_type": "Q", "mode": "opening"}
        return text, meta

    def respond(self, text: str, user_state: Dict[str, Any], talk_his: List[Dict[str, Any]]) -> str:
        """
        text: 用户本轮输入
        user_state: 当前快照
        talk_his: 最近 10 句对话，包含 time / who / text
        """
        payload = {
            "user_text": text,
            "user_state": user_state,
            "talk_his": talk_his,
        }
        reply = self.llm.call_llm("persona_fast", payload, temperature=0.5)
        if not reply:
            reply = "我在听，你可以多跟我说说。"
        return reply
