from typing import Dict, Any
from persona.fast_engine import FastEngine
from persona.slow_engine import SlowEngine
from state.user_profile import UserProfileManager
from state.snapshot_manager import StateSnapshotManager


class FirstTurnEngine:
    """负责“第一句话”的逻辑：Q 开场 or T 开场。"""

    def __init__(
        self,
        fast_engine: FastEngine,
        slow_engine: SlowEngine,
        user_profile_manager: UserProfileManager,
        snapshot_manager: StateSnapshotManager,
    ):
        self.fast = fast_engine
        self.slow = slow_engine
        self.user_profile_manager = user_profile_manager
        self.snapshot_manager = snapshot_manager

    def decide_mode(self) -> str:
        profile = self.user_profile_manager.get()
        flags = profile.get("flags", {})
        if flags.get("long_time_no_see") or flags.get("high_stress"):
            return "T"
        if flags.get("prefer_direct_question"):
            return "Q"
        return "Q"

    def build_opening(self) -> Dict[str, Any]:
        profile = self.user_profile_manager.get()
        snapshot = self.snapshot_manager.get()
        mode = self.decide_mode()
        if mode == "Q":
            text, meta = self.fast.build_opening_question(profile, snapshot)
        else:
            text, meta = self.slow.build_opening_viewpoint(profile, snapshot)
        return {
            "mode": mode,
            "text": text,
            "meta": meta or {},
        }
