# persona/deep_engine.py
from typing import Dict, Any, List
from llm.client import LLMClient


class DeepEngine:
    """
    D-Engine（Deep）· 深度加强人格

    适用场景：
    - T 引擎已经对一个观点做过一轮理性分析；
    - 系统希望在不换话题的前提下，再“加深一层”：
      · 用故事、比喻、具象画面来延展刚才的观点；
      · 或者从另一个角度补充，让同一主题更立体。

    不负责：
    - 换话题；
    - 做完全新的拆解（那是 T 的工作）；
    - 给实用操作步骤（那是 L 的工作）；
    - 总结收尾（那是 SUM 的工作）。
    """

    ROLE = "persona_deep_engine"

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def deepen(
        self,
        user_text: str,
        user_state: Dict[str, Any],
        talk_history: List[Dict[str, Any]],
    ) -> str:
        """
        user_text:
            - 如果是用户刚刚又说了一句（user_triggered=True），这里会有内容；
            - 如果是 9 秒 tick，这里通常是 ""，需要主要依赖 talk_history。
        user_state:
            - current_state_snapshot.json 的内容；
        talk_history:
            - 最近若干轮对话，格式为：
              [{"time": "...", "who": "user/assistant", "text": "..."}, ...]
        """
        payload: Dict[str, Any] = {
            "user_text": user_text,
            "user_state": user_state,
            "talk_history": talk_history,
        }
        raw = self.llm.call_llm(self.ROLE, payload, temperature=0.7)
        if not raw:
            return ""
        # persona_deep_engine 的 prompt 约定：直接返回一段自然语言文本
        return raw.strip()
