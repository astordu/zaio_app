# persona/slow_engine.py

from typing import Dict, Any, List, Tuple
from llm.client import LLMClient


class SlowEngine:
    """
    在哦 · T-Engine 慢人格：
    - 用于结构化讨论、观点输出、深层需求挖掘。
    - 会结合 snapshot（用户状态）、talk_history（最近对话）、
      current_tree（当前观点树）来组织一轮“承上启下”的回复。
    """

    ROLE = "persona_slow"

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def build_opening_viewpoint(
        self,
        user_profile: Dict[str, Any],
        snapshot: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        当一开始就想用 T 引擎开场时，可以用这个函数生成第一句。
        现在先做一个相对简单的版本：基于 snapshot 给出一个“帮你理一理”的入口。
        """
        emotion = snapshot.get("emotion", {}).get("value", "中性") if snapshot else "中性"
        need_list = snapshot.get("need") or []

        if need_list:
            need_str = "、".join(need_list)
            text = f"我感觉你现在整体状态有点偏「{emotion}」，心里好像装着「{need_str}」这些事。不如我们先挑一件，你觉得最卡的那个，一起理一理？"
        else:
            text = f"从现在这些信息看，你大概是在「{emotion}」的状态里绕来绕去。我可以帮你把最近的事情捋一捋，你可以先随便说一个最近让你在意的点。"

        meta = {"opening_type": "T"}
        return text, meta

    def respond(
        self,
        text: str,
        snapshot: Dict[str, Any],
        talk_history: List[Dict[str, Any]],
        current_tree: Dict[str, Any],
    ) -> str:
        """
        T 引擎主力回复：
        - text: 本轮用户说的话
        - snapshot: 当前用户状态快照
        - talk_history: 最近若干句对话（含 role / time / text）
        - current_tree: 当前观点树结构（含当前节点、上一节点、children 等）

        会把这四个字段打包成 JSON，发给 persona_slow 对应的 system prompt。
        """
        payload = {
            "user_text": text,
            "snapshot": snapshot,
            "talk_history": talk_history,
            "current_tree": current_tree,
        }

        reply = self.llm.call_llm(
            role=self.ROLE,
            payload=payload,
            temperature=0.6,
        )

        if not reply:
            # 兜底：LLM 没回时给一句安全的提示
            reply = "我可以帮你把这件事拆成两三块来看看，你可以先告诉我，哪一块对你来说压力最大？"

        return reply
