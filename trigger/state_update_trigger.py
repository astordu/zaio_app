# trigger/state_update_trigger.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Any, List, Optional
import json
import re
import time

from llm.client import LLMClient


class StateUpdateTrigger:
    """
    用户 state_snapshot 属性更新触发器（稳定版）。

    目标：
    - LLM 输出不严格时也能解析出 dict（避免永远 {}）
    - 最少返回一个心跳字段，保证 snapshot 能落盘
    - 字段短、低噪声，便于后续 prompt 使用

    role = trigger_state_update
    """

    ROLE = "trigger_state_update"

    # 你可以在这里收敛“允许写入”的字段，避免 LLM 胡乱造 key
    # 如果你想完全开放，就把 ALLOWED_KEYS 设为 None
    ALLOWED_KEYS = {
        "emotion",
        "need",
        "energy",
        "activity",
        "concern",
        "body_state",
        "social_state",
        "topic",
        "mood",
        "risk",
        "sleep",
        "focus",
        # 内部字段
        "_last_update_ts",
        "_last_user_text",
    }

    SYSTEM_PROMPT = r"""
你是「在哦 · state_snapshot 更新触发器」。

任务：根据 user_text + history + snapshot，输出“需要更新的最小字段集合”。

重要规则：
1) 你只能输出【严格 JSON 对象】（必须以 { 开头，以 } 结尾），不要输出任何解释、不要 markdown、不要 ```json。
2) 字段值要短：每个字符串字段尽量 2~8 个字。
3) 只更新“本轮有把握的字段”；不确定就不要编造。
4) 你必须至少输出一个字段："_last_update_ts"（unix 秒）以及 "_last_user_text"（原样 user_text）。

允许字段（尽量从中选）：
emotion, need, energy, activity, concern, body_state, social_state, topic, mood, risk, sleep, focus
以及内部字段：_last_update_ts, _last_user_text

输出示例（仅示例，不要带注释）：
{"emotion":"烦","need":"想被理解","energy":"低","_last_update_ts":1730000000,"_last_user_text":"..."}
""".strip()

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    # ---------- public ----------

    def infer_updates(
        self,
        user_text: str,
        history: List[Dict[str, Any]],
        snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = {
            "system": self.SYSTEM_PROMPT,
            "user_text": user_text,
            "history": history,
            "snapshot": snapshot,
        }

        raw = self.llm.call_llm(self.ROLE, payload, temperature=0.2)
        # 兜底：至少写心跳，保证 snapshot 能落盘
        fallback = {
            "_last_update_ts": int(time.time()),
            "_last_user_text": user_text,
        }

        if not raw:
            return fallback

        data = self._parse_json_obj(raw)
        if not isinstance(data, dict):
            return fallback

        # 合并心跳字段（无论 LLM 是否给）
        if "_last_update_ts" not in data:
            data["_last_update_ts"] = fallback["_last_update_ts"]
        if "_last_user_text" not in data:
            data["_last_user_text"] = fallback["_last_user_text"]

        # 可选：过滤 key，避免 LLM 乱写
        if self.ALLOWED_KEYS is not None:
            data = {k: v for k, v in data.items() if k in self.ALLOWED_KEYS}

        # 清理无意义值
        data = self._clean_values(data)

        # 如果清理后只剩心跳，也没关系——至少不会是 {}
        return data or fallback

    # ---------- helpers ----------

    def _parse_json_obj(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None

        s = text.strip()

        # 去掉 ```json ... ``` 包裹
        s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.I)
        s = re.sub(r"\s*```\s*$", "", s)

        # 先尝试整体解析
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        # 再尝试提取第一个 {...}（允许 LLM 在前后夹杂文字）
        m = re.search(r"\{.*\}", s, flags=re.S)
        if not m:
            return None

        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None

        return None

    def _clean_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        清理一些容易污染 snapshot 的值：
        - None / "" / "null" / "None" 等 → 丢弃（除非是内部字段）
        - 把非基础类型转成字符串（尽量不崩）
        """
        cleaned: Dict[str, Any] = {}
        for k, v in data.items():
            if k in {"_last_update_ts", "_last_user_text"}:
                cleaned[k] = v
                continue

            if v is None:
                continue

            if isinstance(v, str):
                vv = v.strip()
                if not vv:
                    continue
                if vv.lower() in {"null", "none"}:
                    continue
                # 可选：再截断一下长度，防止 LLM 写长句
                if len(vv) > 20:
                    vv = vv[:20]
                cleaned[k] = vv
                continue

            # 允许基本类型
            if isinstance(v, (int, float, bool)):
                cleaned[k] = v
                continue

            # 其他类型兜底转字符串
            cleaned[k] = str(v)

        return cleaned
