# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Optional
import json
import re

from llm.client import LLMClient


def _extract_json(text: str) -> Optional[dict]:
    if not isinstance(text, str):
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


class PerspectiveMoveTrigger:
    """
    观点树节点推进触发器（T 引擎配套）。

    输出 move_info:
    {
      "move": bool,
      "next_node_id": "N1" | None,
      "need_new_tree": bool,
      "reason": "..."
    }
    """

    ROLE = "trigger_perspective_move"

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def decide_move(
        self,
        current_node: Dict[str, Any],
        user_text: str,
        ai_text: str,
        snapshot: Dict[str, Any],
        talk_history: List[Dict[str, Any]],
        full_tree: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        让 LLM 判断：是否要推进到子节点 / 是否该重建一棵新树。
        """
        system_prompt = """你是「在哦 · 观点树移动判断器」。

你不是在重新理解世界，你只做“路径决策”：
- move：走向某个 child
- stay：留在当前节点
- need_new_tree：只有在“用户明确换话题/换目标”且当前树完全承载不了时才允许

你会收到：
- current_node（含 core_need / children）
- full_tree（含 child 节点的 core_need）
- user_text（本轮用户输入）
- talk_history（最近对话）
- snapshot（用户状态）

输出必须是严格 JSON：
{
  "move": "move|stay|none",
  "next_node_id": "<child id or null>",
  "need_new_tree": true|false,
  "reason": "<20字以内>"
}

强规则（必须遵守）：

【1】默认策略：优先 stay，其次 move，最最后才是 need_new_tree
- 只要“还能解释成当前主题的深化/延伸/举例/情绪反应”，就 stay 或 move
- 你必须避免频繁换树

【2】need_new_tree 的必要条件（必须全部满足）：
A. 用户输入包含明确的“换话题/换目标/不聊这个”的信号（显性）
   例如：换个话题、别说这个了、不聊了、算了、另外、说回…、我们谈点别的、回到…、其实我想问…
B. 当前节点及其 children 都无法合理承载 user_text
   （用户内容与当前 core_need 无关，且与任何 child 的 core_need 都无关）
C. 如果拿不准是否满足 A 或 B，一律不要换树，选择 stay

【3】move 的条件：
- 当 user_text 明显更符合某个 child 的 core_need 时，move
- next_node_id 必须是 children 内的一个

【4】stay 的条件：
- 大多数情况都应该 stay
- 当 user_text 仍在当前主题范围，但不清晰指向某个 child 时 stay

一致性约束：
- 若 need_new_tree = true → move 必须是 "none"，next_node_id 必须是 null
- 若 move = "move" → need_new_tree 必须是 false

现在开始输出 JSON。

"""

        payload = {
            "system_prompt": system_prompt,
            "current_node": current_node or {},
            "user_text": user_text,
            "ai_text": ai_text,
            "snapshot": snapshot or {},
            "talk_history_tail": (talk_history or [])[-8:],
            "full_tree_meta": {
                "tree_id": (full_tree or {}).get("tree_id"),
                "root_id": (full_tree or {}).get("root_id"),
                "current_node_id": (full_tree or {}).get("current_node_id"),
            },
            "full_tree_nodes": (full_tree or {}).get("nodes", {}),
        }

        reply = self.llm.call_llm(self.ROLE, payload, temperature=0.25)
        obj = _extract_json(reply) or {}

        # 兜底 + 约束修复
        move = bool(obj.get("move", False))
        need_new_tree = bool(obj.get("need_new_tree", False))
        reason = obj.get("reason") if isinstance(obj.get("reason"), str) else ""

        children = []
        try:
            children = list(current_node.get("children") or [])
        except Exception:
            children = []

        next_node_id = obj.get("next_node_id")
        if not isinstance(next_node_id, str):
            next_node_id = None

        # need_new_tree 优先级最高：一旦为 True，就不要 move
        if need_new_tree:
            return {"move": False, "next_node_id": None, "need_new_tree": True, "reason": reason or "这棵树不太贴合现在的话题，换一棵更合适的。"}

        # 没 children 就不能 move
        if not children:
            return {"move": False, "next_node_id": None, "need_new_tree": False, "reason": reason}

        # next_node_id 必须在 children 里
        if move:
            if next_node_id not in children:
                # 如果模型没给对，就别动（宁可不动也别乱跳）
                return {"move": False, "next_node_id": None, "need_new_tree": False, "reason": reason}
            return {"move": True, "next_node_id": next_node_id, "need_new_tree": False, "reason": reason}

        return {"move": False, "next_node_id": None, "need_new_tree": False, "reason": reason}
