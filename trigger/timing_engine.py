
from typing import Dict, Any, List
from state.snapshot_manager import StateSnapshotManager
from llm.client import LLMClient


class TimingEngine:
    """
    新 TimingEngine（V2）：
    - 不再判断是否需要说话
    - 不调用任何 LLM
    - 只负责节奏保护：限制 AI 连续说话的次数
    """

    def __init__(self, snapshot_manager, max_consecutive_ai=2):
        self.snapshot_manager = snapshot_manager
        self.consecutive_ai = 0
        self.max_consecutive_ai = max_consecutive_ai

    def on_ai_spoken(self):
        """AI 说话后调用，用于累加连续计数"""
        self.consecutive_ai += 1

    def on_user_spoken(self):
        """用户说话后调用，重置连续计数"""
        self.consecutive_ai = 0

    def allow_ai_speak(self) -> bool:
        """节奏保护：限制 AI 狂刷输出"""
        return self.consecutive_ai < self.max_consecutive_ai

