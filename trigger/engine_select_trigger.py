from typing import Dict, Any, List
from llm.client import LLMClient
import json


class EngineSelectTrigger:
    """Q/T/L/SUM 引擎选择触发器：由 LLM 决定本轮采用哪一种人格。

    对应 llm.client 里的 role = trigger_select_engine。
    只接受一个简单的上下文：
    - user_text: 本轮用户输入（tick 场景为空字符串）
    - snapshot: 当前用户状态快照
    - history: 最近若干轮对话（list[dict]）
    - user_triggered: 是否为用户触发（True=用户说话，False=系统 tick）
    """

    ROLE = "trigger_select_engine"

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def select(
        self,
        user_text: str,
        snapshot: Dict[str, Any],
        history: List[Dict[str, Any]],
        user_triggered: bool,
    ) -> str:
        """返回当前应该采用的引擎模式：Q / T / L / SUM / D。"""
        payload: Dict[str, Any] = {
            "user_text": user_text,
            "user_triggered": user_triggered,
            "snapshot": snapshot,
            # prompt 中使用的是 talk_history 这个字段名
            "talk_history": history,
        }

        raw = self.llm.call_llm(self.ROLE, payload, temperature=0.0)
        if not raw:
            # 兜底：无响应时统一退回 Q
            return "Q"

        try:
            data = json.loads(raw)
            mode = str(data.get("mode") or "Q").upper().strip()
            if mode in ("Q", "T", "L", "SUM", "D"):
                return mode
        except Exception:
            pass

        return "Q"
