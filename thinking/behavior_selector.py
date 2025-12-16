import json
from typing import Dict, Any, List
from llm.client import LLMClient
from trigger.engine_select_trigger import EngineSelectTrigger
from state.snapshot_manager import StateSnapshotManager
from state.history_manager import HistoryManager


class BehaviorSelector:
    """
    LLM 决策模块：选择 Q / T / L
    """

    def __init__(self, llm, snapshot_manager, history_manager):
        self.llm = llm
        self.snapshot_manager = snapshot_manager
        self.history_manager = history_manager
        self.ROLE = "trigger_select_engine"

    def select(self, user_text: str, user_triggered: bool) -> str:
        """
        输出： "Q" / "T" / "L" / "SUM"
        根据：
        - 用户最新一句话
        - 是否主动触发
        - snapshot（用户状态）
        - talk_history（上下文）
        - current_node / previous_node / next_nodes（观点树上下文）
        - full_tree（完整观点树）
        - has_sum_invite（当前节点是否具备总结邀请）
    
        用 LLM 决策。
        """
    
        # === 基础上下文 ===
        snapshot = self.snapshot_manager.get()
        talk_his = self.history_manager.get_talk_his(limit=10)
    
        # === 获取观点树结构 ===
        try:
            full_tree = self.snapshot_manager.tree_manager.get_raw_tree()
        except Exception:
            full_tree = {}
    
        # 当前节点
        try:
            current_node = self.snapshot_manager.tree_manager.get_current_node()
        except Exception:
            current_node = {"id": "ROOT", "children": []}
    
        # 上一节点 ID
        previous_node_id = current_node.get("_previous_node_id")
        previous_node = None
    
        nodes = full_tree.get("nodes") or {}
        if previous_node_id and previous_node_id in nodes:
            previous_node = nodes[previous_node_id]
        elif previous_node_id:
            previous_node = {"id": previous_node_id}
    
        # 下一节点
        next_ids = current_node.get("children", [])
        next_nodes = []
        if isinstance(next_ids, list):
            for nid in next_ids:
                next_nodes.append(nodes.get(nid, {"id": nid}))
    
        # === 是否具有总结邀请 ===
        # 若当前节点含 sum_invite，则 SUM 可触发
        has_sum_invite = bool(current_node.get("sum_invite"))
    
        # === 构造 payload ===
        payload = {
            "user_text": user_text,
            "user_triggered": user_triggered,
            "snapshot": snapshot,
            "talk_history": talk_his,
            "current_node": current_node,
            "previous_node": previous_node,
            "next_nodes": next_nodes,
            "full_tree": full_tree,
            "has_sum_invite": has_sum_invite,
        }
    
        # === 交给 LLM ===
        raw = self.llm.call_llm(self.ROLE, payload, temperature=0.0)
        if not raw:
            return "Q"
    
        try:
            data = json.loads(raw)
            mode = data.get("mode")
            if mode in ("Q", "T", "L", "SUM"):
                return mode
        except:
            return "Q"
    
        return "Q"
