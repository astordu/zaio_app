# trigger/talk_trigger.py

import json
from typing import List, Dict, Any
from llm.client import LLMClient


class TalkTrigger:
    """
    在哦 · 小触发器（TalkTrigger）

    - 只负责根据 recent_lines 判断【此刻要不要继续说下一句话】。
    - 使用 LLM role: trigger_should_speak
    - 具体规则写在 llm.client.py 中该 role 对应的 system prompt 里。
    """

    ROLE = "trigger_should_speak"

    def __init__(self, llm_client: LLMClient):
        # 这里一定要接收 llm_client，才能在 orchestrator 里注入进来
        self.llm = llm_client

    def should_reply(
        self,
        recent_lines: List[Dict[str, Any]],
        user_triggered: bool
    ) -> bool:
        """
        参数：
            recent_lines: 最近最多 5 条对话，每条是：
                {
                    "time": "2025-11-23 03:01:19",
                    "role": "user" / "assistant",
                    "text": "……"
                }
            user_triggered: 本轮是否是用户先开口（用于兜底策略）

        返回：
            True  → 此刻应该继续说话
            False → 此刻不该说话
        """

        payload = {
            "recent_lines": recent_lines
        }

        raw = self.llm.call_llm(self.ROLE, payload, temperature=0.1)

        # 如果 LLM 没正常返回，就用一个安全兜底策略：
        # - 用户刚说话 → 默认回复
        # - 纯 Tick → 默认沉默
        if not raw:
            return bool(user_triggered)

        try:
            data = json.loads(raw)
            val = data.get("should_reply")

            # 严格按你的提示词，只看 should_reply
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.strip().lower() == "true"

        except Exception:
            # LLM 返回格式不对，也走兜底
            return bool(user_triggered)

        # 万一没有 should_reply 字段，也兜底
        return bool(user_triggered)
