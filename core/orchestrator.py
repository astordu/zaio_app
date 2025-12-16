import os
import threading
import time
import json
import random

from typing import Callable, Dict, Any, List

from llm.client import LLMClient
from state.snapshot_manager import StateSnapshotManager
from state.user_profile import UserProfileManager
from state.history_manager import HistoryManager

from trigger.timing_engine import TimingEngine          # 只做节奏保护
from trigger.talk_trigger import TalkTrigger            # 新·小触发器：是否说话
from trigger.state_update_trigger import StateUpdateTrigger
from trigger.perspective_move_trigger import PerspectiveMoveTrigger

from thinking.perspective_tree import PerspectiveTree
from thinking.guess_engine import GuessEngine
from trigger.engine_select_trigger import EngineSelectTrigger

from persona.fast_engine import FastEngine
from persona.slow_engine import SlowEngine
from persona.direct_engine import DirectEngine
from persona.deep_engine import DeepEngine   # ← 新增

from .first_turn import FirstTurnEngine
from thinking.perspective_generate_engine import PerspectiveGenerateEngine

from persona.sum_engine import SumEngine


class ConversationOrchestrator:
    """在哦 · V2 业务骨架：TalkTrigger + Q/T/L 三人格 + 观点树。"""

    def __init__(
        self,
        ui_callback: Callable[[str], None],
        log_dir: str = None,
        trigger_interval: int = 9,
        snapshot_path: str = None,
    ):
        # ====== 主动说话次数上限相关 ======
        self.auto_reply_max = 3          # 上限 3 句
        self.auto_reply_used = 0         # 已经说了几句（当前这轮）
        
        self.ui_callback = ui_callback
        self.ui_on_thinking_start = None
        self.ui_on_thinking_end = None
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = os.path.dirname(base_dir)

        self.log_dir = log_dir or os.path.join(self.base_dir, "data", "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.trigger_interval = trigger_interval

        # --- 基础组件 ---
        self.llm_client = LLMClient()
        self.snapshot_manager = StateSnapshotManager(
            snapshot_path or os.path.join(self.base_dir, "data", "current_state_snapshot.json")
        )
        self.user_profile_manager = UserProfileManager(
            os.path.join(self.base_dir, "data", "user_profile.json")
        )
        self.history_manager = HistoryManager(self.log_dir)

        # --- 触发器 ---
        # 小触发器：是否要继续说话（LLM）
        self.talk_trigger = TalkTrigger(self.llm_client)
        # 用户状态更新触发器（LLM）
        self.state_update_trigger = StateUpdateTrigger(self.llm_client)
        # T 引擎观点树推进触发器（LLM）
        self.perspective_move_trigger = PerspectiveMoveTrigger(self.llm_client)

        # --- 节奏保护引擎（不走 LLM，只限制连续发言次数） ---
        self.timing_engine = TimingEngine(self.snapshot_manager, max_consecutive_ai=2)

        # --- 思考层 ---
        self.perspective_tree = PerspectiveTree()
        self.guess_engine = GuessEngine(self.perspective_tree)
        # Q/T/L/SUM 引擎选择器：直接走 EngineSelectTrigger
        self.engine_select_trigger = EngineSelectTrigger(self.llm_client)
        # --- 观点树生成引擎 ---
        self.perspective_generate_engine = PerspectiveGenerateEngine(
            self.llm_client,
            self.base_dir
        )
        # --- 人格引擎 ---
        self.fast_engine = FastEngine(self.llm_client)    # Q-Engine
        self.slow_engine = SlowEngine(self.llm_client)    # T-Engine
        self.direct_engine = DirectEngine(self.llm_client)  # L-Engine
        self.sum_engine = SumEngine(self.llm_client)        # SUM-Engine 总结人格（新增）
        self.deep_engine = DeepEngine(self.llm_client)      # D-Engine 深度加强人格
        
        #self._last_mode = None  # 记录上一轮采用的模式：Q/T/L/SUM/D,未来可以有一个节奏表，或者用离散引擎输出节奏，用来指导引擎选择器。也就是收集大量的人机交互引擎交换节奏，然后模拟这个节奏。反向工程。
        
        # 第一拍开场引擎
        self.first_turn_engine = FirstTurnEngine(
            self.llm_client,
            self.user_profile_manager,
            self.history_manager,
            self.snapshot_manager,
        )

        # 触发循环线程
        self._trigger_thread = None
        self._trigger_running = False
        # 初次启动时，先加载一棵“初见默认树”作为占位
        self.reset_perspective_tree_to_default()
        
    def _build_default_first_meet_tree(self) -> dict:
        """
        默认“初见”观点树：
        - 用来在重置/回到初见后占位，保证观点树是全新的。
        - 结构尽量简单，3~4 个节点即可。
        """
        return {
            "tree_id": "default_first_meet_v1",
            "root_id": "N0",
            "nodes": {
                "N0": {
                    "id": "N0",
                    "title": "刚见面，先摸摸状态",
                    "user_viewpoint": "用户刚打开在哦，可能只是随便打个招呼，状态还不明确。",
                    "our_viewpoint": "先大致感受一下你现在的状态，再决定要陪你聊轻松的，还是帮你处理一点正事。",
                    "potential_need": ["有人听一听", "先暖个场再深入"],
                    "children": ["N1"],
                    "is_end": False
                },
                "N1": {
                    "id": "N1",
                    "title": "一起定一个这次的主线",
                    "user_viewpoint": "用户已经表示出一个更明确的方向（比如倾诉/讨论某个观点/解决某个问题）。",
                    "our_viewpoint": "把这次对话的主线先捏一下：是陪你发发牢骚，还是一起拆一个问题，还是陪你脑暴点新东西。",
                    "potential_need": ["确定主线", "进入下一个主题树"],
                    "children": [],
                    "is_end": True
                }
            }
        }
    def reset_perspective_tree_to_default(self):
        """
        把当前观点树重置为“初见默认树”。
        用于：启动时 / 点击『回到初见』时。
        """
        try:
            default_tree = self._build_default_first_meet_tree()
            self.perspective_tree.load_tree(default_tree)
            self.perspective_generate_engine.save_tree_to_file(default_tree)
            print("[Orchestrator] 观点树已重置为默认初见树")
        except Exception as e:
            print("[Orchestrator] 重置默认观点树失败:", e)
            
    def _load_random_history_tree(self) -> bool:
        """
        从 data/perspective_trees 目录里随机挑一棵【非默认】观点树，
        并加载到当前的 PerspectiveTree 中。

        返回：
        - True  : 成功加载了一棵历史树
        - False : 没有可用的历史树（或者出错）
        """
        trees_dir = os.path.join(self.base_dir, "data", "perspective_trees")
        if not os.path.isdir(trees_dir):
            print("[Orchestrator] 没找到观点树目录:", trees_dir)
            return False

        files = [f for f in os.listdir(trees_dir) if f.endswith(".json")]
        if not files:
            print("[Orchestrator] 观点树目录为空")
            return False

        candidates = []
        for fname in files:
            lower = fname.lower()
            # 排除默认树（包含 default / 默认 的文件名）
            if "default" in lower or "默认" in fname:
                continue
            candidates.append(os.path.join(trees_dir, fname))

        if not candidates:
            print("[Orchestrator] 没有找到非默认观点树 json，全部是默认")
            return False

        path = random.choice(candidates)
        try:
            with open(path, "r", encoding="utf-8") as f:
                tree = json.load(f)

            # 没有 tree_id 的话，用文件名兜底
            tree.setdefault("tree_id", os.path.splitext(os.path.basename(path))[0])

            # 直接加载到当前观点树
            self.perspective_tree.load_tree(tree)
            print(f"[Orchestrator] 已加载历史观点树: {os.path.basename(path)}")
            return True
        except Exception as e:
            print("[Orchestrator] 加载历史观点树失败:", path, e)
            return False
            
    def handle_time_jump(self):
        """
        【时光飞逝一下】按钮逻辑：

        1. 重置“在哦连续说话”的计数 → 相当于又有 3 句预算。
        2. 立刻让在哦说一句：
           - 有 50% 概率：强制走 T 引擎 + 随机一棵历史非默认观点树；
           - 另外 50%：按原来的 EngineSelectTrigger 正常选人格。
        3. 这一次也会被 TimingEngine 计入连续说话次数（= 1）。
        """
        print("[Orchestrator] ▶ 时光飞逝一下")

        # 1）重置连续 AI 计数，相当于“又可以重新说 3 句”
        #    （复用 on_user_spoken 的重置逻辑）
        self.timing_engine.on_user_spoken()

        # 2）准备上下文（这里没有新的用户输入，所以 user_text 为空字符串）
        user_text = ""
        snapshot = self.snapshot_manager.get()
        history = self.history_manager.get_recent(limit=10)

        # 3）决定本轮用什么人格
        use_T = (random.random() < 0.5)
        if use_T and self._load_random_history_tree():
            # 成功加载了历史树 → 强制走 T 模式
            mode = "T"
            print("[Orchestrator] 本次时光飞逝：采用 T 引擎 + 历史观点树")
        else:
            # 否则走原本的自动选择逻辑
            mode = self.engine_select_trigger.select(
                user_text=user_text,
                snapshot=snapshot,
                history=history,
                user_triggered=False,
            )
            print(f"[Orchestrator] 本次时光飞逝：采用默认引擎选择模式 → {mode}")

        # 4）执行人格行为并发送到 UI
        reply = self._run_behavior(mode, user_text)
        self._send_ai_message(reply)

    # === 启动 ===
    def start(self):
        """启动 orchestrator：目前主要是启动 tick 触发循环。"""
        if not self._trigger_thread:
            self._trigger_running = True
            self._trigger_thread = threading.Thread(target=self._trigger_loop, daemon=True)
            self._trigger_thread.start()

        # 打第一拍
        self._send_first_message()

    # === 停止 ===
    def stop(self):
        self._trigger_running = False
        
    def register_thinking_start(self, fn):
        self.ui_on_thinking_start = fn

    def register_thinking_end(self, fn):
        self.ui_on_thinking_end = fn
    # === UI 回调包装 ===
    def _send_ai_message(self, text: str):
        if not text:
            return
        # 更新历史
        self.history_manager.append_ai(text)
        # UI
        try:
            self.ui_callback(text)
        except Exception:
            pass
        # 告诉节奏引擎：AI 说过话了
        self.timing_engine.on_ai_spoken()

    # === 第一拍 ===
    def _send_first_message(self):
        opening = self.first_turn_engine.build_opening()
        text = opening.get("text") or ""
        self._send_ai_message(text)

    # === 对外接口：用户输入 ===
    def handle_user_message(self, text: str):
        user_text = text.strip()
        if not user_text:
            return

        # 写入历史
        self.history_manager.append_user(user_text)
        # 用户开口 → 重置连续 AI 计数
        self.timing_engine.on_user_spoken()

        # 用户输入 → 先更新用户状态快照
        self._update_state_snapshot(user_text)

        # 再走一轮完整决策
        self._handle_event_user_message(user_text)

    # === 触发循环控制 ===
    def start_trigger_loop(self):
        if self._trigger_thread is not None:
            return

        self._trigger_running = True
        self._trigger_thread = threading.Thread(target=self._trigger_loop, daemon=True)
        self._trigger_thread.start()

    def stop_trigger_loop(self):
        self._trigger_running = False
        if self._trigger_thread is not None:
            self._trigger_thread.join(timeout=1.0)

    def _trigger_loop(self):
        while self._trigger_running:
            time.sleep(self.trigger_interval)
            try:
                self._handle_event_tick()
            except Exception as e:
                print("[Orchestrator] trigger_loop error:", e)

    # === 内部辅助 ===
    def _build_recent_lines_for_trigger(self) -> List[Dict[str, str]]:
        """
        给 TalkTrigger 用的 recent_lines：
        [
          {"time": "...", "role": "user"/"assistant", "text": "..."},
          ...
        ]
        """
        lines: List[Dict[str, str]] = []
        for item in self.history_manager.get_recent(5):
            lines.append(
                {
                    "time": item.get("time", ""),
                    "role": item.get("role", ""),
                    "text": item.get("text", ""),
                }
            )
        return lines

    def _update_state_snapshot(self, user_text: str):
        """调用 StateUpdateTrigger 由 LLM 推断本轮需要更新的 snapshot 字段。"""
        # 1）先拿到当前 snapshot（全字段）
        snapshot = self.snapshot_manager.get()
        # 2）拿最近若干轮对话历史，给 LLM 做判断依据
        history = self.history_manager.get_recent(30)
        # 3）让 StateUpdateTrigger 根据 user_text + history + snapshot 推断更新项
        updates = self.state_update_trigger.infer_updates(user_text, history, snapshot)
        print("[Snapshot] updates =", updates)
        # 4）有结果就批量写入 snapshot；没结果也写一个心跳，保证文件存在并可追踪
        if not updates:
            updates = {"_last_update_ts": time.time(), "_last_user_text": user_text}
        self.snapshot_manager.update_multi(updates)


    # === Tick 事件（系统主动说话） ===
    def _handle_event_tick(self):
        recent_lines = self._build_recent_lines_for_trigger()

        # 1）小触发器：是否应该继续说话？（LLM）
        if not self.talk_trigger.should_reply(recent_lines, user_triggered=False):
            return

        # 2）节奏保护：避免 AI 连续说太多
        if not self.timing_engine.allow_ai_speak():
            return

        # 3）选择人格（Q/T/L/SUM）——通过 EngineSelectTrigger
        user_text = ""  # tick 场景下没有新的用户输入
        snapshot = self.snapshot_manager.get()
        history = self.history_manager.get_recent(limit=10)
        mode = self.engine_select_trigger.select(
            user_text=user_text,
            snapshot=snapshot,
            history=history,
            user_triggered=False,
        )
        
        
        # 4）执行人格行为
        reply = self._run_behavior(mode, user_text)
        self._send_ai_message(reply)

    # === 用户消息事件 ===
    def _handle_event_user_message(self, user_text: str):
        recent_lines = self._build_recent_lines_for_trigger()

        # 1）小触发器：是否应该继续说话？（LLM）
        if not self.talk_trigger.should_reply(recent_lines, user_triggered=True):
            return

        # 2）节奏保护：避免 AI 一直说个没完
        if not self.timing_engine.allow_ai_speak():
            return

        # 3）选择人格（Q/T/L/SUM）——通过 EngineSelectTrigger
        snapshot = self.snapshot_manager.get()
        history = self.history_manager.get_recent(limit=10)
        mode = self.engine_select_trigger.select(
            user_text=user_text,
            snapshot=snapshot,
            history=history,
            user_triggered=True,
        )
        
        # 4）执行人格行为
        reply = self._run_behavior(mode, user_text)
        self._send_ai_message(reply)

    # === 行为执行：Q / T / L 等===
    def _run_behavior(self, mode: str, user_text: str) -> str:
        """
        根据引擎选择器给出的 mode，调用对应人格引擎：
        - Q  : 快人格（收集信息）
        - T  : 慢人格（观点树 + 深度讨论）
        - SUM: 总结人格（复盘整理）
        - L  : 直答人格（给结论 / 给方案）
        """
        mode = (mode or "Q").upper()
        user_state = self.snapshot_manager.get()

        # ===== Q 引擎 =====
        if mode == "Q":
            talk_his = self.history_manager.get_talk_his(limit=10)
            return self.fast_engine.respond(user_text, user_state, talk_his)

        # ===== T 引擎 =====
        if mode == "T":
            # 1）准备 T 引擎和观点树触发器需要的上下文
            snapshot = user_state
            talk_his = self.history_manager.get_talk_his(limit=10)

            # 尝试获取整棵观点树
            try:
                full_tree = self.perspective_tree.get_raw_tree()
            except AttributeError:
                # 若还没实现导出整棵树，就至少传一个“只含当前节点”的简版
                current_node_tmp = self.perspective_tree.get_current_node()
                cid = current_node_tmp.get("id", "ROOT")
                full_tree = {
                    "tree_id": "runtime_current",
                    "current_node_id": cid,
                    "nodes": {cid: current_node_tmp},
                }

            # 2）让 T 引擎根据观点树 + 历史 + snapshot 生成本轮回复
            ai_text = self.slow_engine.respond(
                user_text,
                snapshot,
                talk_his,
                full_tree,
            )

            # 3）当前节点（注意：respond 期间树理论上可能会变，这里重新拿一次）
            current_node = self.perspective_tree.get_current_node()

            # 4）让观点树触发器基于当前节点 & 对话，判断下一步动作
            move_info = self.perspective_move_trigger.decide_move(
                current_node=current_node,
                user_text=user_text,
                ai_text=ai_text,
                snapshot=snapshot,
                talk_history=talk_his,
                full_tree=full_tree,
            )

            # 5）根据 move_info 推进观点树 / 或者生成新树
            if move_info:
                # 如果 LLM 判断“这棵树方向不对，需要重建”
                if move_info.get("need_new_tree"):
                    try:
                        # 用本轮的 user_text + 当前 snapshot + 最近对话，生成一棵新树
                        new_tree = self.perspective_generate_engine.generate_tree(
                            user_text=user_text,
                            snapshot=snapshot,
                            talk_history=talk_his,
                        )
                        if new_tree:
                            self.perspective_tree.load_tree(new_tree)
                            print("[Orchestrator] 观点树已根据 need_new_tree 生成并加载新树")
                    except Exception as e:
                        print("[Orchestrator] generate_tree error:", e)
                else:
                    # 否则按原逻辑，只做节点移动
                    try:
                        self.perspective_tree.apply_move(move_info)
                    except Exception as e:
                        print("[Orchestrator] apply_move error:", e)
        
            return ai_text


        # ===== SUM 引擎 =====
        if mode == "SUM":
            talk_his = self.history_manager.get_talk_his(limit=20)
            # 尝试获取当前观点树（可以为空）
            try:
                current_tree = self.perspective_tree.get_raw_tree()
            except AttributeError:
                current_tree = None

            return self.sum_engine.respond(
                user_text=user_text,
                user_state=user_state,
                talk_history=talk_his,
                current_tree=current_tree,
            )

        # ===== L 引擎 =====
        if mode == "L":
            return self.direct_engine.answer(user_text, user_state)
            
        # ===== D 引擎 =====
        if mode == "D":
            talk_his = self.history_manager.get_talk_his(limit=12)
            return self.deep_engine.deepen(
                user_text=user_text,
                user_state=user_state,
                talk_history=talk_his,
            )
            
        # ===== 兜底：模式异常时退回 Q =====
        talk_his = self.history_manager.get_talk_his(limit=10)
        return self.fast_engine.respond(user_text, user_state, talk_his)
