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
        system_prompt = """【你的任务】

你需要根据以下信息，判断本轮对话应该如何在观点树中移动：

你会收到：

当前 current_node_id

当前节点的 core_need

当前节点的 children

用户的最新输入（只作为参考，不重新生成观点）

【你只能做三种判断之一】
1️⃣ 沿当前树向下移动（move）

当满足以下条件时：

用户输入 明显呼应 当前节点的 core_need

且某一个 child 节点的 core_need 与用户输入更匹配

2️⃣ 停留在当前节点（stay）

当满足以下条件时：

用户仍在当前立场范围内

但没有明显指向任何一个 child

3️⃣ 生成新观点树（need_new_tree）

当出现以下任一情况：

用户输入关注的核心动机 无法映射到当前节点的 core_need

用户开始讨论 新的判断维度 / 新的立场张力

当前节点的所有 children 都无法合理承载用户输入

⚠️ 注意：
一旦判断 need_new_tree = true，就不要再 move。

【你必须输出的 JSON 结构】

⚠️ 只能输出以下 JSON，不要输出任何解释性文字。

{
  "move": "move | stay | none",
  "next_node_id": "<如果 move，则必须是 children 中的一个；否则为 null>",
  "need_new_tree": true | false,
  "reason": "<一句话，说明判断依据，20字以内>"
}

【强约束规则（必须遵守）】
关于判断依据

❌ 不要重新解释 philosophical_insight

❌ 不要引入新的观点

❌ 不要生成新的 need

✅ 只允许使用：

当前节点的 core_need

child 节点的 core_need

用户输入的显性意图

关于 move

如果 move = "move"：

next_node_id 必须是当前节点 children 中的一个

need_new_tree 必须为 false

关于 need_new_tree

如果 need_new_tree = true：

move 必须是 "none"

next_node_id 必须是 null

【你的判断标准】

请始终问自己一句话：

“这还是同一个立场的深化吗，
还是已经换了一个‘在讨论什么’？”

如果是后者，请选择 need_new_tree = true。
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
