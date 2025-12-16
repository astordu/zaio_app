# -*- coding: utf-8 -*-
import json
import os
import datetime
import re
from typing import Any, Dict, List, Optional

from llm.client import LLMClient


def _extract_json(text: str) -> Optional[dict]:
    """
    尝试从 LLM 输出中提取 JSON 对象：
    - 允许前后有解释文字/代码块
    - 优先提取第一个大括号包裹的对象
    """
    if not isinstance(text, str):
        return None
    s = text.strip()

    # 去掉 ```json ``` 包裹
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    # 直接尝试整体解析
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 抓取第一个 {...} 块（粗略，但够用）
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    chunk = m.group(0)
    try:
        obj = json.loads(chunk)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


class PerspectiveGenerateEngine:
    """
    负责“生成一棵新的观点树”并落盘到 data/perspective_trees。

    目标：
    - 当当前树不再适配（话题转向/树走到尽头/用户提出新主线）时，能快速生成一棵新树
    - 生成结果要稳定可用（schema 校验 + 兜底树）
    """

    ROLE = "perspective_generate_engine"

    def __init__(self, llm_client: LLMClient, base_dir: str):
        self.llm = llm_client
        self.base_dir = base_dir
        self.tree_dir = os.path.join(base_dir, "data", "perspective_trees")
        os.makedirs(self.tree_dir, exist_ok=True)

    # -------- public --------
    def generate_tree(self, user_text: str, snapshot: dict, talk_history: list) -> Dict[str, Any]:
        """
        返回一棵“可立即加载”的树 dict（不是路径）。
        同时会自动落盘保存一份到 data/perspective_trees。
        """
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        tree_id = f"tree_{ts}"

        system_prompt = """【你的任务】

你需要 分两个阶段完成一次生成：

🔹 阶段一｜自由生成（不受约束）

请基于当前对话主题，自由地理解用户正在探讨的核心立场张力。

在这一阶段：

你可以使用你认为最合适的语言

可以是抽象的、哲学的、结构性的

不需要考虑工程约束

不需要考虑字段长度

不需要考虑是否“好用”

你要做的只是这件事：

抓住这一轮对话中，真正值得被长期保存的“立场洞见”

🔹 阶段二｜自我映射（工程化收敛）

在完成自由理解之后，
你必须 把你刚才的理解，翻译成一个可被系统使用的结构化节点。

⚠️ 注意：
这一阶段不是重新理解世界，而是“给系统一个可执行接口”。

【你必须输出的最终 JSON 结构】

⚠️ 你只能输出以下 JSON，不要输出任何额外文字。

{
  "tree_id": "<自动生成>",
  "root_id": "N0",
  "current_node_id": "N0",
  "generated_at": "<YYYY-MM-DD_HH-MM-SS>",
  "nodes": {
    "N0": {
      "id": "N0",
      "title": "<一句话概括当前立场主题，8-12字>",
      "philosophical_insight": "<阶段一自由生成的洞见，压缩为 1 句话，20-35 个汉字，不提问，不对用户说话>",
      "core_need": [
        "<从下方枚举中选择 1-2 个，用来映射你的洞见>"
      ],
      "children": ["N1", "N2"],
      "is_end": false
    },

    "N1": {
      "id": "N1",
      "title": "<一个可能的立场分化方向>",
      "philosophical_insight": "<一句新的洞见>",
      "core_need": ["<枚举之一>"],
      "children": [],
      "is_end": true
    },

    "N2": {
      "id": "N2",
      "title": "<另一个可能的立场分化方向>",
      "philosophical_insight": "<一句新的洞见>",
      "core_need": ["<枚举之一>"],
      "children": [],
      "is_end": true
    }
  }
}

【core_need 可用枚举（工程接口）】

你必须 从下列标签中选择，这是系统的决策接口：

理解机制
价值判断
风险评估
经验映射
现实选择
情绪安放
自我定位
行动准备

⚠️ 重要规则

这是 映射接口，不是完整表达

如果你的洞见 无法合理映射到任何一个标签

请选择最接近的

不要发明新标签

如果你觉得“完全映射不了”，说明这一主题需要 生成新树，而不是扩展当前树

【强约束规则（必须遵守）】
philosophical_insight

必须是 1 句话

20–35 个汉字

不提问

不使用“你”

不出现“我们可以继续讨论”等对话性语言

core_need

最多 2 个

每个 2–4 个字

只用于系统决策，不用于情感表达

【你不需要做的事】

❌ 不要复述用户说过的话

❌ 不要写对话引导

❌ 不要写分析过程

❌ 不要解释你的选择

【你的判断标准】

如果这个节点在三天后被再次读取：

它是否依然成立？

它是否像一个“立场”，而不是一句话？

它是否能自然导向不同的思考方向？

如果不能，请重新压缩你的洞见。
"""

        payload = {
            "system_prompt": system_prompt,
            "tree_id_hint": tree_id,
            "generated_at": ts,
            "user_text": user_text,
            "snapshot": snapshot or {},
            "talk_history": (talk_history or [])[-10:],
        }

        reply = self.llm.call_llm(self.ROLE, payload, temperature=0.55)
        tree = _extract_json(reply)

        tree = self._normalize_and_validate(tree, fallback_ts=ts, fallback_tree_id=tree_id)
        self.save_tree_to_file(tree)
        return tree

    def save_tree_to_file(self, tree: Dict[str, Any]) -> str:
        """
        保存 tree dict 到 data/perspective_trees，并返回文件路径。
        """
        if not isinstance(tree, dict):
            return ""
        tree_id = tree.get("tree_id") or f"tree_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        filename = f"{tree_id}.json"
        path = os.path.join(self.tree_dir, filename)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(tree, f, indent=2, ensure_ascii=False)
            print("[PerspectiveGenerateEngine] 已保存观点树:", path)
            return path
        except Exception as e:
            print("[PerspectiveGenerateEngine] 保存观点树失败:", e)
            return ""

    # -------- internal --------
    def _normalize_and_validate(self, tree: Optional[dict], fallback_ts: str, fallback_tree_id: str) -> Dict[str, Any]:
        """
        对 LLM 输出做最小修复 + schema 校验；不合格则返回兜底树。
        """
        if not isinstance(tree, dict):
            return self._fallback_tree(fallback_tree_id, fallback_ts)

        # 基本字段补全
        tree.setdefault("tree_id", fallback_tree_id)
        tree.setdefault("generated_at", fallback_ts)
        tree.setdefault("root_id", "N0")
        tree.setdefault("current_node_id", tree.get("root_id", "N0"))
        nodes = tree.get("nodes")
        if not isinstance(nodes, dict) or not nodes:
            return self._fallback_tree(tree["tree_id"], tree["generated_at"])

        # 修复 root / current
        if tree["root_id"] not in nodes:
            tree["root_id"] = "N0" if "N0" in nodes else next(iter(nodes.keys()))
        if tree["current_node_id"] not in nodes:
            tree["current_node_id"] = tree["root_id"]

        # 逐节点修复
        for nid, n in list(nodes.items()):
            if not isinstance(n, dict):
                nodes[nid] = {"id": nid}
                n = nodes[nid]
            n.setdefault("id", nid)
            n.setdefault("title", "（待定）")
            n.setdefault("user_viewpoint", "（待定）")
            n.setdefault("our_viewpoint", "（待定）")
            pn = n.get("potential_need")
            if not isinstance(pn, list) or not pn:
                n["potential_need"] = ["陪你理一理", "确认你真正想要的"]
            ch = n.get("children")
            if not isinstance(ch, list):
                n["children"] = []
            n.setdefault("is_end", False)

        # children 过滤不存在节点
        node_ids = set(nodes.keys())
        for n in nodes.values():
            n["children"] = [c for c in (n.get("children") or []) if c in node_ids]

        # 至少一条推进路径（root 至少有一个 child，若没有则补一条）
        root = nodes.get(tree["root_id"])
        if root and not root.get("children"):
            # 找一个非 root 的节点作为 child
            alt = None
            for k in nodes.keys():
                if k != tree["root_id"]:
                    alt = k
                    break
            if alt:
                root["children"] = [alt]

        # 至少一个 end 节点
        if not any(bool(n.get("is_end")) for n in nodes.values()):
            # 优先选叶子节点
            leaf = None
            for k, n in nodes.items():
                if not n.get("children"):
                    leaf = k
                    break
            if leaf:
                nodes[leaf]["is_end"] = True
            else:
                # 实在没有就把最后一个设为 end
                last = list(nodes.keys())[-1]
                nodes[last]["is_end"] = True

        return tree

    def _fallback_tree(self, tree_id: str, ts: str) -> Dict[str, Any]:
        """
        兜底树：保证 T 引擎不崩。
        """
        return {
            "tree_id": tree_id,
            "generated_at": ts,
            "root_id": "N0",
            "current_node_id": "N0",
            "nodes": {
                "N0": {
                    "id": "N0",
                    "title": "先把主线捏出来",
                    "user_viewpoint": "你可能想聊点什么，但还没完全说清楚。",
                    "our_viewpoint": "我们先把‘你现在最想处理的那一件事’挑出来，再决定怎么聊。",
                    "potential_need": ["确定主线", "有人接住"],
                    "children": ["N1"],
                    "is_end": False,
                },
                "N1": {
                    "id": "N1",
                    "title": "拆一个最卡的点",
                    "user_viewpoint": "你开始说出困扰的细节，或者说出你真正的担心。",
                    "our_viewpoint": "我先帮你把最卡的那个点拆清楚：你在担心什么、你想要什么、你怕失去什么。",
                    "potential_need": ["缓解焦虑", "获得清晰感"],
                    "children": ["N2"],
                    "is_end": False,
                },
                "N2": {
                    "id": "N2",
                    "title": "给一个下一步",
                    "user_viewpoint": "你希望有个可执行的小动作，别只停在情绪里。",
                    "our_viewpoint": "我给你一个小到立刻能做的下一步：不求完美，只求开始动起来。",
                    "potential_need": ["行动建议", "被陪着执行"],
                    "children": [],
                    "is_end": True,
                },
            },
        }
