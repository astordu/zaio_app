
import json
import os
from typing import Dict, Any


class StateSnapshotManager:
    """负责 current_state_snapshot.json 的简单读写。"""

    def __init__(self, path: str):
        self.path = path
        self._data: Dict[str, Any] = {}
        self.load()

    def load(self) -> Dict[str, Any]:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}
        else:
            self._data = {}
        return self._data

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def get(self) -> Dict[str, Any]:
        return dict(self._data)

    def update_multi(self, updates: Dict[str, Any]):
        self._data.update(updates)
        self.save()
