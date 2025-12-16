
from typing import Dict, Any, List
from .perspective_tree import PerspectiveTree


class GuessEngine:
    """综合观点树 + 调理树，对本轮做一个“猜”（简单占位版）。"""

    def __init__(self, perspective_tree: PerspectiveTree):
        self.perspective_tree = perspective_tree

    def guess(self, user_text: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        node = self.perspective_tree.get_current_node()
        return {
            "node": node,
            "raw_text": user_text,
        }
