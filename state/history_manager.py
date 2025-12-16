from typing import List, Dict, Any
import datetime


class HistoryManager:
    """在内存里维护最近对话。

    说明：
    - 不再负责写 TXT 文件，TXT 日志统一由 main.py 的 _append_log 完成；
    - 这里只维护一个结构化的 history 列表，供触发器 / 人格引擎使用。
    """

    def __init__(self, log_dir: str):
        # log_dir 现在只是占位，为兼容旧接口，仍然保留参数，但不再使用
        self.log_dir = log_dir
        self.history: List[Dict[str, Any]] = []

    def get_recent_lines(self, limit: int = 5):
        """
        返回最近 limit 条对话，格式为：
        [
            {"time": "...", "role": "...", "text": "..."},
            ...
        ]
        """
        items = self.history[-limit:]
        result = []
        for it in items:
            result.append({
                "time": it.get("time", ""),
                "role": it.get("role", ""),  # user / assistant
                "text": it.get("text", "")
            })
        return result

    def get_talk_his(self, limit: int = 10):
        """
        返回最近 limit 句对话，格式为：
        [
          {"time": "...", "who": "user"/"assistant", "text": "..."},
          ...
        ]
        """
        items = self.history[-limit:]
        result = []
        for it in items:
            role = it.get("role")
            who = "user" if role == "user" else "assistant"
            result.append({
                "time": it.get("time", ""),
                "who": who,
                "text": it.get("text", ""),
            })
        return result

    def append_user(self, text: str):
        """记录一条用户发言（只进内存，不写文件）。"""
        now = datetime.datetime.now()
        ts = now.strftime("%Y-%m-%d %H:%M:%S")
        self.history.append({
            "role": "user",
            "text": text,
            "time": ts,
        })
        # 不再写 TXT，避免和 main._append_log 的格式冲突

    def append_ai(self, text: str):
        """记录一条 AI 发言（只进内存，不写文件）。"""
        now = datetime.datetime.now()
        ts = now.strftime("%Y-%m-%d %H:%M:%S")
        self.history.append({
            "role": "assistant",
            "text": text,
            "time": ts,
        })
        # 不再写 TXT，避免和 main._append_log 的格式冲突

    def get_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        """返回最近 limit 条原始记录。"""
        return self.history[-limit:]
