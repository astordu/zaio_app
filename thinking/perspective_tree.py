from typing import Dict, Any


class PerspectiveTree:
    """
    观点树管理器

    - 负责保存当前使用的观点树结构（由 LLM 生成或默认占位）
    - 负责记录当前所在节点、上一节点
    - 对外提供：
        - get_current_node()
        - move_to(node_id)
        - load_tree(tree_dict)
        - get_raw_tree()
    """

    def __init__(self, tree: Dict[str, Any] | None = None):
        self.tree: Dict[str, Any] = {}
        self.current_node_id: str = "ROOT"
        self.previous_node_id: str | None = None

        if tree:
            self.load_tree(tree)
        else:
            self._init_default_tree()

    # === 内部：初始化一棵默认的“占位树” ===
    def _init_default_tree(self):
        """
        当还没有真正的观点树时，提供一个安全的默认树。
        """
        self.tree = {
            "tree_id": "runtime_default",
            "root_id": "ROOT",
            "nodes": {
                "ROOT": {
                    "id": "ROOT",
                    "user_viewpoint": "现在还在随意聊聊，没有形成特别明确的主题。",
                    "our_viewpoint": "先陪你把最近的状态说清楚，再慢慢一起找关键点。",
                    "potential_need": "陪伴 / 倾诉 / 有人听你说",
                    "children": []
                }
            }
        }
        self.current_node_id = "ROOT"
        self.previous_node_id = None

    # === 对外：加载一棵新的观点树（来自 LLM 生成引擎） ===
    def load_tree(self, tree: Dict[str, Any]):
        """
        接收 perspective_generate_engine 生成的树结构，并接管它。

        兼容几种可能的结构：
        - 带 nodes + root_id
        - 带 nodes + current_node_id
        - 只有 root 节点
        """
        self.tree = tree or {}

        # 尝试确定 root / current 节点
        root_id = self._get_root_id_from_tree(self.tree)

        # 如果树里指定了 current_node_id，就优先用它
        current_id = self.tree.get("current_node_id") or root_id or "ROOT"

        self.current_node_id = current_id
        self.previous_node_id = None

        # 如果树结构太简陋，至少保证有一个节点可用
        nodes = self._get_nodes_dict()
        if self.current_node_id not in nodes:
            nodes[self.current_node_id] = {"id": self.current_node_id, "children": []}
            self.tree["nodes"] = nodes

    # === 获取当前节点（并附带上一节点 id、children 等信息） ===
    def get_current_node(self) -> Dict[str, Any]:
        """
        返回当前节点的信息，并做一些补充：

        - 一定包含 id
        - 一定包含 children（没有则给空列表）
        - 如果有上一节点，则在 _previous_node_id 里附加
        """
        nodes = self._get_nodes_dict()
        raw_node = nodes.get(self.current_node_id, {}) or {}

        node: Dict[str, Any] = dict(raw_node)  # 拷贝一份，避免改到原数据

        node.setdefault("id", self.current_node_id)
        node.setdefault("children", [])

        if self.previous_node_id:
            node["_previous_node_id"] = self.previous_node_id

        return node

    # === 对外：移动到某个节点 ===
    def move_to(self, node_id: str):
        """
        将当前节点移动到 node_id。

        - 如果 node_id 在 nodes 里，就正常前进
        - 如果不在，仍然允许移动，但视为“树外节点”，以免逻辑卡死
        """
        if not node_id:
            return

        nodes = self._get_nodes_dict()

        # 记录上一节点
        self.previous_node_id = self.current_node_id
        self.current_node_id = node_id

        # 如果该节点不存在，补一个最小占位节点
        if node_id not in nodes:
            nodes[node_id] = {"id": node_id, "children": []}
            self.tree["nodes"] = nodes

    # === 对外：导出整棵树（给 T 引擎 / SUM 引擎 / 触发器用） ===
    def get_raw_tree(self) -> Dict[str, Any]:
        """
        返回整棵观点树的原始结构，并补充 current_node_id / root_id。
        """
        tree = self.tree or {}
        if "current_node_id" not in tree:
            tree["current_node_id"] = self.current_node_id
        if "root_id" not in tree:
            tree["root_id"] = self._get_root_id_from_tree(tree)
        return tree

    # === 可选：重置到根节点 ===
    def reset_to_root(self):
        self.previous_node_id = None
        self.current_node_id = self._get_root_id_from_tree(self.tree)

    # === 内部工具：获取 nodes dict ===
    def _get_nodes_dict(self) -> Dict[str, Any]:
        """
        尝试从 tree 中解析出 nodes 字典。
        兼容多种可能格式：
        - 明确的 tree["nodes"] 为 dict
        - 只有一个 root 节点（tree["root"]）
        """
        tree = self.tree or {}

        if isinstance(tree.get("nodes"), dict):
            return tree["nodes"]

        # 兼容只有 root 的情况
        root = tree.get("root")
        if isinstance(root, dict):
            node_id = root.get("id", "ROOT")
            return {node_id: root}

        # 都没有时，给一个最小占位
        return {
            self.current_node_id: {
                "id": self.current_node_id,
                "children": []
            }
        }

    # === 内部工具：从 tree 里推 root_id ===
    def _get_root_id_from_tree(self, tree: Dict[str, Any]) -> str:
        if not tree:
            return "ROOT"

        if "root_id" in tree and isinstance(tree["root_id"], str):
            return tree["root_id"]

        if "root" in tree and isinstance(tree["root"], dict):
            return tree["root"].get("id", "ROOT")

        nodes = tree.get("nodes")
        if isinstance(nodes, dict) and nodes:
            # 任意取一个节点作为 root（兜底）
            return next(iter(nodes.keys()))

        return "ROOT"
        
    def apply_move(self, move_info: Dict[str, Any]) -> None:
        """
        根据观点树触发器给出的 move_info 推进当前节点。

        move_info 的典型结构（由 PerspectiveMoveTrigger.decide_move 返回）：
        {
            "move": bool,
            "next_node_id": str | None,
            "need_new_tree": bool,
            "reason": str,
        }

        当前简化实现：
        - 如果 move=False：不做任何事；
        - 如果 move=True 且 next_node_id 在当前树中，调用 self.move_to(next_node_id)；
        - need_new_tree 暂时不处理（后面可以在这里接上“生成新树”的逻辑）。
        """
        if not isinstance(move_info, dict):
            return

        # 不需要移动就直接退出
        if not move_info.get("move"):
            return

        next_id = move_info.get("next_node_id")
        if not isinstance(next_id, str) or not next_id:
            return

        try:
            # 假设你之前已经实现了 move_to(node_id)
            self.move_to(next_id)
        except Exception:
            # 如果节点不存在或其它错误，先静默忽略，避免中断对话
            return
