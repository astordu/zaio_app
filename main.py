import re
import os
import math
import random
import threading
import datetime
import json
from log_view_controller import load_and_format_llm_log
import dearpygui.dearpygui as dpg
from core.orchestrator import ConversationOrchestrator
from ui.chat_view import ChatView

# ================================
# 基本参数（固定窗口）
# ================================
WINDOW_W = 1400
WINDOW_H = 768
TITLEBAR_EXTRA = 90

TOP_GRADIENT_H = 60
BOTTOM_GRADIENT_H = 120

DOLL_WIDTH = 360           # 娃娃展示宽度（可调）
DOLL_LEFT_X = 40           # 娃娃左边距
DOLL_TOP_OFFSET = 100      # 娃娃顶端距离顶部渐变的间距

FONT_SIZE = 33             # 聊天 & 按钮基础字号

LEFT_AREA_W_RATIO = 0.3   # 左侧娃娃占宽比例
CHAT_AREA_W_RATIO = 0.5   # 中间聊天区比例
RIGHT_AREA_W_RATIO = 0.20  # 右侧按钮比例

DOLL_TEX_SIZE = (1, 1)

# 聊天区宽度（在 build_ui 里赋值，给气泡用）
CHAT_INNER_WIDTH = 420
CHAT_VIEW_H = 0  # 聊天窗口可视高度

BOTTOM_GRADIENT_OFFSET = 40  # 正数 = 往下压一点

# 气泡宽度控制
BUBBLE_MAX_WIDTH = 750          # 气泡最大宽度
BUBBLE_MIN_WIDTH = 20          # 气泡最小宽度
BUBBLE_EXTRA_W = 70            # 每个气泡统一额外加宽
MAX_CHARS_PER_LINE = 11      # 每行最多字符数（控制自动换行）
RIGHT_BUBBLE_MARGIN = 100       # 右侧气泡整体左移，预留滚动条空间

# 主题用的颜色（基础糖果系）
BEIGE = (245, 232, 220, 255)
PINK = (255, 174, 200, 255)
CYAN = (132, 224, 233, 255)
WHITE = (255, 255, 255, 255)

# 额外一个紫色按钮颜色
BUTTON_PURPLE = (190, 160, 255, 255)

# 设计图按钮颜色池（正黄、正蓝、正红、粉色、卡其）
BUTTON_COLORS = [
    (255, 211, 84, 255),   # 黄
    (104, 216, 226, 255),  # 蓝
    (255, 111, 111, 255),  # 红
    (255, 159, 194, 255),  # 粉
    (240, 216, 192, 255),  # 卡其
]

# 气泡颜色（填充 + 文本描边）
BUBBLE_PINK_FILL = (255, 170, 200, 255)
BUBBLE_PINK_OUTLINE = (230, 110, 160, 255)

BUBBLE_CYAN_FILL = (132, 224, 233, 255)
BUBBLE_CYAN_OUTLINE = (70, 180, 200, 255)

# DEMO 标题颜色：白字 + 暖色描边
TITLE_MAIN_COLOR = (255, 255, 255, 255)
TITLE_OUTLINE_COLOR = (255, 210, 190, 255)

# 标题固定偏移（先居中再整体挪一点）
TITLE_X_OFFSET = -130   # 负数 = 向左一点
TITLE_Y_OFFSET = -30    # 负数 = 向上一点

# 聊天区当前高度偏移
chat_y_offset = 0

orchestrator = None

# 字体 tag
TITLE_FONT_TAG = "title_font"
MAIN_FONT_TAG = "cn_font"

# ==== 获取 main.py 所在目录（解决工作目录不一致问题）====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PERSPECTIVE_DIR = os.path.join(BASE_DIR, "data", "perspective_trees")
DEFAULT_TREE_PATH = os.path.join(BASE_DIR, "data", "tree_default.json")

def asset_path(relative_path):
    """始终以 main.py 的位置为基准查找资源路径"""
    return os.path.join(BASE_DIR, relative_path)

# ==== 对话日志设置（自动写入 TXT）====
LOG_DIR = os.path.join(BASE_DIR, "data", "logs")

def _append_log(side: str, text: str):
    """将每一条气泡写入当天的对话日志 TXT。"""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        now = datetime.datetime.now()
        filename = f"对话_{now.strftime('%Y%m%d')}.txt"
        log_path = os.path.join(LOG_DIR, filename)
        role = "USER" if side == "right" else "AI"
        # 将多行内容压成一行，避免 TXT 里换行太乱
        safe_text = text.replace("\r", " ").replace("\n", " / ")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] [{role}] {safe_text}\n")
    except Exception:
        # 日志失败不影响前台对话
        pass

# ==== LLM Prompt 日志查看路径（与 llm_client 保持一致）====
PROMPT_LOG_PATH = os.path.join(BASE_DIR, "data", "prompt_logs", "llm_prompt_log.txt")
PROMPT_LOG_WINDOW_TAG = "prompt_log_window"
PROMPT_LOG_TEXT_TAG = "prompt_log_text"

# ==== 人格引擎查看窗口 ====
PERSONA_WINDOW_TAG = "persona_engine_window"
PERSONA_TEXT_TAG = "persona_engine_text"

def doll_set_off():
    """让娃娃显示 OFF 态图片"""
    try:
        dpg.configure_item("tex_doll_on_image", texture_tag="tex_doll_off")
        print("[UI] 娃娃切换为 OFF 图（忙碌中）")
    except Exception as e:
        print("[UI] doll_set_off 出错：", e)

def doll_set_on():
    """让娃娃显示 ON 态图片"""
    try:
        dpg.configure_item("tex_doll_on_image", texture_tag="tex_doll_on")
        print("[UI] 娃娃切换为 ON 图（空闲）")
    except Exception as e:
        print("[UI] doll_set_on 出错：", e)

def load_texture(path, tag):
    abs_path = os.path.abspath(path)
    print(f"🔍 尝试加载纹理: {abs_path}")

    if not os.path.exists(abs_path):
        print(f"❌ 文件不存在: {abs_path}")
        return None

    result = dpg.load_image(abs_path)
    if result is None:
        print(f"❌ dpg.load_image 返回 None: {abs_path}")
        return None

    w, h, c, data = result

    with dpg.texture_registry(show=False):
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
        dpg.add_static_texture(w, h, data, tag=tag)

    print(f"✔ 成功加载纹理 {tag}: {w} x {h}")
    return w, h

def sanitize_text(text: str) -> str:
    """把所有容易导致 UI 宽度计算出错的字符统一替换为空格。"""

    # 1. 删除 markdown 或特殊格式符号（任何非中英文 / 数字 / 常用标点）
    #   中日韩：\u4e00-\u9fa5
    #   英文数字：A-Za-z0-9
    #   常用中文标点：。，！？、；：（）《》“”‘’
    #   常用英文标点：.,!?;:()'"  
    safe_pattern = r"[^A-Za-z0-9\u4e00-\u9fa5。，！？、；：（）《》“”‘’.,!?;:()'\"\s]"

    text = re.sub(safe_pattern, " ", text)

    # 2. 多个空格压缩为一个
    text = re.sub(r"\s{2,}", " ", text)

    # 3. 去掉首尾空格
    return text.strip()
    
def _get_latest_perspective_tree_path() -> str:
    """
    返回最新的观点树 JSON 文件路径：
    1）优先 data/perspective_trees 目录中，按修改时间取最新的 .json；
    2）如果目录不存在或为空，退回 data/tree_default.json；
    3）如果都没有，返回空字符串。
    """
    # 先看目录
    if os.path.isdir(PERSPECTIVE_DIR):
        candidates = [
            os.path.join(PERSPECTIVE_DIR, f)
            for f in os.listdir(PERSPECTIVE_DIR)
            if f.lower().endswith(".json")
        ]
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return candidates[0]

    # 退回默认
    if os.path.isfile(DEFAULT_TREE_PATH):
        return DEFAULT_TREE_PATH

    return ""
    
# ------------------------------------
# 全局主题：背景米色 + 圆角 + 滚动条颜色
# ------------------------------------
def apply_global_theme():
    with dpg.theme() as theme:
        # 背景颜色 & 滚动条
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, BEIGE, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, BEIGE, category=dpg.mvThemeCat_Core)

            # 滚动条槽 + 滑块颜色，接近蜡笔
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, (252, 227, 230, 255), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, (246, 160, 190, 255), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, (246, 160, 190, 255), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, (246, 160, 190, 255), category=dpg.mvThemeCat_Core)

        # 所有按钮的通用样式（圆角 + 文本居中）
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 22, 22, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ButtonTextAlign, 0.5, 0.5, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Text, WHITE, category=dpg.mvThemeCat_Core)

        # 输入框
        with dpg.theme_component(dpg.mvInputText):
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (250, 255, 255, 255), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Text, (80, 80, 80, 255), category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 22, 22, category=dpg.mvThemeCat_Core)

    dpg.bind_theme(theme)


# ------------------------------------
# 文本手动换行：按字符数插入 \\n
# ------------------------------------
def _wrap_text_by_chars(text: str, chars_per_line: int) -> str:
    if chars_per_line <= 0:
        return text
    lines = []
    cur = 0
    n = len(text)
    while cur < n:
        lines.append(text[cur:cur + chars_per_line])
        cur += chars_per_line
    return "\n".join(lines)

def show_perspective_tree_window():
    """
    点击“观点树”按钮时调用：
    - 找到最新的观点树 JSON 文件
    - 读内容
    - 弹出一个窗口，用只读多行文本展示
    """
    path = _get_latest_perspective_tree_path()
    if not path:
        dpg.add_window(
            label="观点树查看",
            width=600,
            height=400,
            modal=True,
            no_resize=False,
            no_collapse=True,
        )
        with dpg.window(label="观点树查看", width=600, height=200, modal=True, no_resize=False) as win:
            dpg.add_text("当前没有找到任何观点树 JSON 文件。")
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pretty = json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as e:
        pretty = f"加载观点树失败：{e!r}\n文件路径：{path}"

    # 弹出窗口
    with dpg.window(
        label=f"观点树查看 - {os.path.basename(path)}",
        width=700,
        height=500,
        no_resize=False,
        no_collapse=False,
        no_close=False,
        modal=False,
    ) as win_id:
        dpg.add_text(f"文件路径：{path}")
        dpg.add_separator()
        dpg.add_input_text(
            multiline=True,
            readonly=True,
            default_value=pretty,
            width=-1,     # 占满窗口
            height=-1,    # 占满窗口
        )
        
# ------------------------------------
# 在 drawlist 里绘制“描边文字”
# ------------------------------------
def draw_outlined_text(drawlist_tag, x, y, text, size, main_color, outline_color):
    # 4 个方向的描边
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dx, dy in offsets:
        dpg.draw_text(
            pos=(x + dx, y + dy),
            text=text,
            size=size,
            color=outline_color,
            parent=drawlist_tag
        )
    # 中心主文字
    dpg.draw_text(
        pos=(x, y),
        text=text,
        size=size,
        color=main_color,
        parent=drawlist_tag
    )


# ------------------------------------
# 聊天气泡：在 drawlist 中绘制
#   左边：粉色（娃娃）
#   右边：青色（用户）
# ------------------------------------
def draw_one_bubble(text, side="left", y=0):
    text = sanitize_text(text)

    drawlist_tag = "chat_drawlist"

    padding_x = 30
    padding_y = 18
    line_height = int(FONT_SIZE * 1.1)

    wrapped_text = _wrap_text_by_chars(text, MAX_CHARS_PER_LINE)
    lines = wrapped_text.split("\n") if wrapped_text else [""]

    avg_char_w = FONT_SIZE * 0.76 or 10
    max_line_chars = max(len(line) for line in lines) if lines else 1
    estimated_max_line_width = max_line_chars * avg_char_w

    bubble_w = estimated_max_line_width + 2 * padding_x
    bubble_w = min(BUBBLE_MAX_WIDTH, bubble_w)
    bubble_w += BUBBLE_EXTRA_W
    bubble_w = max(BUBBLE_MIN_WIDTH, bubble_w)

    bubble_h = int(padding_y * 2 + len(lines) * line_height)

    if side == "left":
        x = 0
        fill_color = BUBBLE_PINK_FILL
        outline_color = BUBBLE_PINK_OUTLINE
    else:
        x = max(0, CHAT_INNER_WIDTH - RIGHT_BUBBLE_MARGIN - bubble_w)
        fill_color = BUBBLE_CYAN_FILL
        outline_color = BUBBLE_CYAN_OUTLINE

    dpg.draw_rectangle(
        pmin=(x, y),
        pmax=(x + bubble_w, y + bubble_h),
        color=fill_color,
        fill=fill_color,
        rounding=18,
        thickness=0,
        parent=drawlist_tag
    )

    for i, line in enumerate(lines):
        draw_outlined_text(
            drawlist_tag,
            x + padding_x,
            y + padding_y + i * line_height,
            line,
            FONT_SIZE,
            WHITE,
            outline_color
        )

    return bubble_h




def on_ai_message(text):
    if text:
        chat_view.add_message(text, "left")


CONFIG_API_PATH = os.path.join("config", "api_key.txt")
API_KEY_WINDOW_TAG = "api_key_window"
API_KEY_INPUT_TAG = "api_key_input"


def _ensure_config_dir():
    cfg_dir = os.path.dirname(CONFIG_API_PATH)
    if cfg_dir and not os.path.exists(cfg_dir):
        os.makedirs(cfg_dir, exist_ok=True)


def _load_api_key_for_dialog():
    """从本地 TXT 读取已保存的 KEY，没有就返回空字符串。"""
    try:
        with open(CONFIG_API_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def _close_api_key_dialog():
    if dpg.does_item_exist(API_KEY_WINDOW_TAG):
        dpg.configure_item(API_KEY_WINDOW_TAG, show=False)


def _save_api_key_from_dialog():
    global orchestrator
    if not dpg.does_item_exist(API_KEY_INPUT_TAG):
        _close_api_key_dialog()
        return

    key = dpg.get_value(API_KEY_INPUT_TAG).strip()
    _ensure_config_dir()
    try:
        with open(CONFIG_API_PATH, "w", encoding="utf-8") as f:
            f.write(key)
    except Exception as e:
        chat_view.add_message(f"保存 API KEY 出错: {e}", "left")
        _close_api_key_dialog()
        return

    try:
        if orchestrator is not None and hasattr(orchestrator, "llm_client"):
            orchestrator.llm_client.reload_api_key()
    except Exception:
        pass

    chat_view.add_message("API KEY 已保存。", "left")
    _close_api_key_dialog()


def open_api_key_dialog(sender=None, app_data=None, user_data=None):
    chat_view.add_message("（正在打开 API KEY 设置窗口）", "left")
    """右侧按钮触发的弹窗：展示 & 编辑 API KEY。"""
    existing_key = _load_api_key_for_dialog()

    if not dpg.does_item_exist(API_KEY_WINDOW_TAG):
        win_w = 420
        win_h = 200
        pos_x = int((WINDOW_W - win_w) / 2)
        pos_y = int((WINDOW_H - win_h) / 2)

        with dpg.window(
            tag=API_KEY_WINDOW_TAG,
            label="输入 API KEY",
            modal=True,
            no_title_bar=True,
            no_resize=True,
            width=win_w,
            height=win_h,
            pos=(pos_x, pos_y),
        ):
            dpg.add_text("请输入你的 DeepSeek API KEY：")
            dpg.add_spacer(height=10)
            dpg.add_input_text(
                tag=API_KEY_INPUT_TAG,
                width=win_w - 40,
                default_value=existing_key,
                hint="例如：sk-******** 开头的一串字符"
            )
            dpg.add_spacer(height=10)
            with dpg.group(horizontal=True):
                dpg.add_button(label="确认", width=120, callback=_save_api_key_from_dialog)
                dpg.add_spacer(width=20)
                dpg.add_button(label="取消", width=120, callback=_close_api_key_dialog)
    else:
        dpg.configure_item(API_KEY_WINDOW_TAG, show=True)
        if dpg.does_item_exist(API_KEY_INPUT_TAG):
            dpg.set_value(API_KEY_INPUT_TAG, existing_key)

def open_state_snapshot_dialog(sender=None, app_data=None, user_data=None):
    """
    打开一个大窗口，展示 current_state_snapshot.json 的内容。
    多次点击不会重复创建窗口，而是复用同一个 window。
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    snapshot_path = os.path.join(base_dir, "data", "current_state_snapshot.json")

    # 读取 snapshot 内容
    if os.path.exists(snapshot_path):
        try:
            with open(snapshot_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                content = "（state_snapshot 为空）"
        except Exception as e:
            content = f"无法读取 snapshot：{e}"
    else:
        content = "（state_snapshot 文件不存在）"

    # 如果窗口已经存在：只更新内容 + 显示窗口，直接 return
    if dpg.does_item_exist("state_snapshot_window"):
        # 更新文本内容（注意我们给输入框起了 tag：state_snapshot_text）
        if dpg.does_item_exist("state_snapshot_text"):
            dpg.set_value("state_snapshot_text", content)
        dpg.configure_item("state_snapshot_window", show=True)
        return

    # 第一次点击：创建窗口
    with dpg.window(
        label="在哦对你现状的理解",
        tag="state_snapshot_window",
        width=800,
        height=600,
        pos=[200, 100],
        modal=True,
        no_resize=False,
    ):
        dpg.add_text("下面是当前状态快照：")
        dpg.add_input_text(
            tag="state_snapshot_text",       # ← 方便后面更新内容
            default_value=content,
            multiline=True,
            readonly=True,
            width=760,
            height=520
        )
        dpg.add_button(
            label="关闭",
            callback=lambda: dpg.configure_item("state_snapshot_window", show=False)
        )


def open_persona_engine_dialog(sender=None, app_data=None, user_data=None):
    """查看人格引擎：使用大窗口显示格式化后的 llm_prompt_log 内容。"""

    # 使用外部模块读取日志并按字符宽度排版
    content = load_and_format_llm_log(PROMPT_LOG_PATH, max_chars=90)

    # ---- 新版窗口尺寸：与软件窗口同宽 ----
    win_w = WINDOW_W
    win_h = int(WINDOW_H * 0.9)       # 高度 90%（你可自行调成 WINDOW_H）
    
    # ---- 新版窗口位置：左上角对齐软件窗口 ----
    pos_x = 0
    pos_y = int(TOP_GRADIENT_H * 1.2)  # 让它避开顶部渐变（你可以调）

    if not dpg.does_item_exist(PERSONA_WINDOW_TAG):

        with dpg.window(
            tag=PERSONA_WINDOW_TAG,
            label="LLM Prompt 调用日志（大窗口）",
            modal=True,
            no_collapse=True,
            no_resize=False,
            width=win_w,
            height=win_h,
            pos=(pos_x, pos_y),
        ):
            dpg.add_text("以下为解析并重排后的 LLM 调用日志：")
            dpg.add_spacer(height=6)

            dpg.add_input_text(
                tag=PERSONA_TEXT_TAG,
                default_value=content,
                multiline=True,
                readonly=True,
                width=-1,
                height=win_h - 90
            )

            dpg.add_spacer(height=6)
            dpg.add_button(
                label="关闭",
                width=80,
                callback=lambda: dpg.configure_item(PERSONA_WINDOW_TAG, show=False)
            )

    else:
        if dpg.does_item_exist(PERSONA_TEXT_TAG):
            dpg.set_value(PERSONA_TEXT_TAG, content)
        dpg.configure_item(PERSONA_WINDOW_TAG, show=True)
        
def reset_to_first_meet(sender=None, app_data=None, user_data=None):
    """
    “回到初见”按钮回调：
    - 删除 data/logs/ 下所有对话日志
    - 删除 data/prompt_logs/ 下所有 LLM 日志
    - 删除 data/current_state_snapshot.json
    （不动 user_profile.json，让长期画像保留）
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    logs_dir = os.path.join(data_dir, "logs")
    prompt_logs_dir = os.path.join(data_dir, "prompt_logs")
    snapshot_path = os.path.join(data_dir, "current_state_snapshot.json")

    # 删除对话日志
    if os.path.exists(logs_dir):
        for name in os.listdir(logs_dir):
            path = os.path.join(logs_dir, name)
            if os.path.isfile(path) and name.endswith(".txt"):
                try:
                    os.remove(path)
                except Exception as e:
                    print("删除对话日志失败:", path, e)

    # 删除 LLM 日志（人格 & 触发器）
    if os.path.exists(prompt_logs_dir):
        for name in os.listdir(prompt_logs_dir):
            path = os.path.join(prompt_logs_dir, name)
            if os.path.isfile(path) and name.endswith(".txt"):
                try:
                    os.remove(path)
                except Exception as e:
                    print("删除 LLM 日志失败:", path, e)

    # 删除状态快照
    if os.path.exists(snapshot_path):
        try:
            os.remove(snapshot_path)
        except Exception as e:
            print("删除 snapshot 失败:", snapshot_path, e)
            
    orchestrator.reset_perspective_tree_to_default()
    
    # 用在哦自己说一句话当作提示（左侧粉色气泡）
    try:
        chat_view.add_message("我刚刚把我们的对话记录和状态快照都清空啦，现在就像第一次见面一样。", "left")
    except Exception as e:
        print("提示气泡添加失败:", e)

def send_message():
    global orchestrator
    txt = dpg.get_value("input_field").strip()
    if not txt:
        return
    # 用户说话：右边青色气泡
    chat_view.add_message(txt, "right")
    dpg.set_value("input_field", "")

    if orchestrator is None:
        chat_view.add_message("（系统还在初始化，请稍后再试）", "left")
        return

    # 丢给 Orchestrator 的任务队列，后台线程会统一处理并通过 ui_callback 画出 AI 气泡
    orchestrator.handle_user_message(txt)
    
def handle_time_jump_button():
    global orchestrator
    if not orchestrator:
        print("⚠ orchestrator 未初始化")
        return

    try:
        orchestrator.handle_time_jump()
    except Exception as e:
        print("时光飞逝执行出错:", e)


def time_jump():
    """底部【时光飞逝一下】按钮的回调。"""
    global orchestrator
    if orchestrator is None:
        chat_view.add_message("（在哦还在启动中，等它准备好再试一试～）", "left")
        return

    try:
        orchestrator.handle_time_jump()
    except Exception as e:
        print("⚠ 时光飞逝一下 调用失败：", e)
        chat_view.add_message("（我刚刚有点卡壳，再点一次试试？）", "left")

# 绘制 DEMO 标题（白字 + 描边，靠上靠左一点）
# ------------------------------------
def draw_title_with_outline():
    # 先用一个隐藏文本测宽高
    temp_id = dpg.add_text("在哦 DEMO版", show=False)
    if dpg.does_alias_exist(TITLE_FONT_TAG):
        dpg.bind_item_font(temp_id, TITLE_FONT_TAG)
    tw, th = dpg.get_item_rect_size(temp_id)
    dpg.delete_item(temp_id)

    # 先居中，再用偏移量挪一挪
    base_x = WINDOW_W // 2 - tw // 2 + TITLE_X_OFFSET
    base_y = TOP_GRADIENT_H // 2 - th // 2 + TITLE_Y_OFFSET

    # 使用一个 drawlist 来画描边文字
    with dpg.drawlist(width=WINDOW_W, height=TOP_GRADIENT_H, pos=(0, 0), parent="root", tag="title_drawlist"):
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in offsets:
            tid = dpg.draw_text(
                pos=(base_x + dx, base_y + dy),
                text="在哦 DEMO版",
                size=FONT_SIZE + 10,
                color=TITLE_OUTLINE_COLOR
            )
            if dpg.does_alias_exist(TITLE_FONT_TAG):
                dpg.bind_item_font(tid, TITLE_FONT_TAG)

        main_id = dpg.draw_text(
            pos=(base_x, base_y),
            text="在哦 DEMO版",
            size=FONT_SIZE + 10,
            color=TITLE_MAIN_COLOR
        )
        if dpg.does_alias_exist(TITLE_FONT_TAG):
            dpg.bind_item_font(main_id, TITLE_FONT_TAG)
            
def simulate_busy():
    cfg = dpg.get_item_configuration("tex_doll_on_image")
    cur = cfg.get("texture_tag")
    if cur == "tex_doll_on":
        doll_set_off()
    else:
        doll_set_on()


# ------------------------------------
# 构建 UI（布局：左娃娃 / 中聊天 / 右栏）
# ------------------------------------
def build_ui():
    global CHAT_INNER_WIDTH, chat_y_offset, CHAT_VIEW_H
    chat_y_offset = 0  # 每次重建 UI 重置

    left_area_w = int(WINDOW_W * LEFT_AREA_W_RATIO)
    chat_area_w = int(WINDOW_W * CHAT_AREA_W_RATIO)
    right_area_w = WINDOW_W - left_area_w - chat_area_w

    chat_x = left_area_w + 20
    chat_y = TOP_GRADIENT_H + 30
    chat_h = WINDOW_H - TOP_GRADIENT_H - BOTTOM_GRADIENT_H - 60

    right_x = left_area_w + chat_area_w + 20
    right_y = TOP_GRADIENT_H + 40
    right_w = right_area_w - 40

    bottom_y = WINDOW_H - BOTTOM_GRADIENT_H + BOTTOM_GRADIENT_OFFSET

    chat_window_width = chat_area_w - 40
    CHAT_INNER_WIDTH = chat_window_width  # 聊天内容最大宽度（也是右侧对齐参考）
    CHAT_VIEW_H = chat_h

    with dpg.window(
        tag="root",
        label="ZAIO",
        width=WINDOW_W,
        height=WINDOW_H,
        no_title_bar=True,
        no_move=True,
        no_resize=True,
        no_scrollbar=True,
    ):

        # ----- 顶部渐变 -----
        dpg.add_image(
            "tex_top_gradient",
            tag="img_top_gradient",
            pos=(0, 0),
            width=WINDOW_W,
            height=TOP_GRADIENT_H
        )

        # ----- 底部渐变 -----
        dpg.add_image(
            "tex_bottom_gradient",
            tag="img_bottom_gradient",
            pos=(0, bottom_y),
            width=WINDOW_W,
            height=BOTTOM_GRADIENT_H
        )

        # ----- 右侧竖向渐变，铺满整个右栏 -----
        right_area_left = left_area_w + chat_area_w
        right_area_width = WINDOW_W - right_area_left
        dpg.add_image(
            "tex_side_gradient",
            tag="img_side_gradient",
            pos=(right_area_left, TOP_GRADIENT_H),
            width=right_area_width,
            height=WINDOW_H - TOP_GRADIENT_H
        )

        # ----- 娃娃 -----
        global DOLL_TEX_SIZE
        w0, h0 = DOLL_TEX_SIZE
        if not w0:
            w0 = 1
        scale = DOLL_WIDTH / w0
        doll_h = int(h0 * scale)

        doll_top = TOP_GRADIENT_H + DOLL_TOP_OFFSET
        dpg.add_image(
            "tex_doll_on",
            tag="tex_doll_on_image",
            pos=(DOLL_LEFT_X, doll_top),
            width=DOLL_WIDTH,
            height=doll_h
        )

        # ----- 中间聊天区域（可滚动） -----
        with dpg.child_window(
            tag="chat_scroll",
            pos=(chat_x, chat_y-10),
            width=chat_window_width,
            height=chat_h+80,
            border=False,
            no_scrollbar=False
        ):
            # 初始高度 = chat_h，之后会随 chat_y_offset 动态增高
            dpg.add_drawlist(
                tag="chat_drawlist",
                width=chat_window_width,
                height=chat_h,
                parent="chat_scroll"
            )
        global chat_view
        chat_view = ChatView(
            drawlist_tag="chat_drawlist",
            scroll_tag="chat_scroll",
            view_height=CHAT_VIEW_H,
            view_width=CHAT_INNER_WIDTH,
            draw_bubble_fn=draw_one_bubble
        )

        # ----- DEMO 标题（描边文字） -----
        draw_title_with_outline()
            
        # ----- 初始几条气泡（左粉 = 娃娃, 右青 = 用户） -----
        chat_view.add_message("在哦，你好", "left")

        # ----- 右侧按钮列（随机颜色） -----
        right_x = left_area_w + chat_area_w + 20
        right_y = TOP_GRADIENT_H + 40
        right_w = right_area_w - 40
        right_h = WINDOW_H - right_y - 40
        
        button_labels = [
            "输入 API KEY", "查看人格引擎", "回到初见", "在哦理解你", "观点树", "模拟忙碌",
            "时光飞逝下", "时光飞逝下", "时光飞逝下", "时光飞逝下"
        ]
        
        btn_h = 56
        btn_gap = 12
        btn_y = right_y
        
        for label in button_labels:
        
            callback = None
            if label == "输入 API KEY":
                callback = open_api_key_dialog
            elif label == "查看人格引擎":
                callback = open_persona_engine_dialog
            elif label == "回到初见":
                callback = reset_to_first_meet    
            elif label == "在哦理解你":
                callback = open_state_snapshot_dialog
            elif label == "观点树":
                callback = show_perspective_tree_window
            elif label == "模拟忙碌":
                callback = simulate_busy
            elif label == "时光飞逝下":
                callback = handle_time_jump_button
        
            btn = dpg.add_button(
                label=label,
                width=right_w,
                height=btn_h,
                pos=(right_x, btn_y),
                callback=callback
            )
        
            # 随机颜色主题（保持你原来设计）
            color = random.choice(BUTTON_COLORS)
            color_theme = dpg.add_theme()
            with dpg.theme_component(dpg.mvAll, parent=color_theme):
                dpg.add_theme_color(dpg.mvThemeCol_Button, color, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, color, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, color, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 22, 22, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_ButtonTextAlign, 0.5, 0.5, category=dpg.mvThemeCat_Core)
        
            dpg.bind_item_theme(btn, color_theme)
            btn_y += btn_h + btn_gap

    
        # ----- 底部输入+发送 -----
        # 左边从娃娃右侧开始
        doll_right = DOLL_LEFT_X + DOLL_WIDTH
        send_w = 80
        margin_left = 20
    
        input_left = doll_right + margin_left
    
        # 右侧对齐到“聊天区域的右边缘”，也就是右侧青色气泡的右边缘
        chat_right = chat_x + chat_window_width
        max_right = chat_right
    
        # input_width + 间隔10 + send_w = 整块宽度
        input_width = max_right - input_left - send_w - 10
        if input_width < 200:
            input_width = 200  # 防止太窄
    
        input_x = input_left
        input_y = bottom_y + 50
    
        dpg.add_input_text(
            tag="input_field",
            width=input_width,
            pos=(input_x, input_y),
            on_enter=True,
            callback=send_message
        )
    
        send_btn = dpg.add_button(
            label="发送",
            width=send_w,
            pos=(input_x + input_width + 10, input_y - 2),
            callback=send_message
        )
    
        # 发送按钮使用固定青色，呼应聊天气泡
        with dpg.theme() as send_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, BUBBLE_CYAN_FILL, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, BUBBLE_CYAN_FILL, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, BUBBLE_CYAN_FILL, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Text, WHITE, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 22, 22, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_ButtonTextAlign, 0.5, 0.5, category=dpg.mvThemeCat_Core)
    
        dpg.bind_item_theme(send_btn, send_theme)


# ------------------------------------
# 主函数
# ------------------------------------
def main():
    dpg.create_context()

    # ===== 字体加载 =====
    font_path = asset_path("assets/ZH80.ttf")

    if os.path.exists(font_path):
        # 避免重复注册字体（同一个 tag 只能 add 一次）
        if not dpg.does_alias_exist(MAIN_FONT_TAG):
            with dpg.font_registry():
                with dpg.font(font_path, FONT_SIZE, tag=MAIN_FONT_TAG) as f:
                    dpg.add_font_range_hint(dpg.mvFontRangeHint_Chinese_Full, parent=f)
                with dpg.font(font_path, FONT_SIZE + 10, tag=TITLE_FONT_TAG) as ft:
                    dpg.add_font_range_hint(dpg.mvFontRangeHint_Chinese_Full, parent=ft)

        # 无论是否新建，都绑定主字体
        dpg.bind_font(MAIN_FONT_TAG)
    else:
        print("⚠ 字体文件未找到:", font_path)

    
    # ===== 纹理加载 =====
    load_texture(asset_path("assets/top_gradient.png"), "tex_top_gradient")
    load_texture(asset_path("assets/bottom_gradient.png"), "tex_bottom_gradient")
    load_texture(asset_path("assets/side_gradient.png"), "tex_side_gradient")
    
    global DOLL_TEX_SIZE
    DOLL_TEX_SIZE = load_texture(asset_path("assets/doll_on.png"), "tex_doll_on")
    load_texture(asset_path("assets/doll_off.png"), "tex_doll_off")
    
    if DOLL_TEX_SIZE is None:
        # 兜底尺寸，防止除 0
        DOLL_TEX_SIZE = (712, 1181)

    # ===== 窗口 =====
    dpg.create_viewport(
        title="ZAIO DEMO - V12.1",
        width=WINDOW_W,
        height=WINDOW_H + TITLEBAR_EXTRA,
        resizable=False
    )

    dpg.setup_dearpygui()
    apply_global_theme()
    build_ui()

    global orchestrator
    orchestrator = ConversationOrchestrator(ui_callback=on_ai_message)
    orchestrator.start_trigger_loop()
    orchestrator.register_thinking_start(doll_set_off)
    orchestrator.register_thinking_end(doll_set_on)
    
    # 也要给 llm_client 注册
    orchestrator.llm_client.on_thinking_start = doll_set_off
    orchestrator.llm_client.on_thinking_end = doll_set_on
    # --- 每次启动时重置 snapshot：所有字段写成“等待发掘” ---
    try:
        from state.snapshot_manager import StateSnapshotManager
    
        # ✅ 用 JSON 文件，而不是 .txt
        snapshot_file = os.path.join(BASE_DIR, "data", "current_state_snapshot.json")
    
        _ssm = StateSnapshotManager(snapshot_file)
    
        _ssm.update_multi({
            "emotion": "等待发掘",
            "energy": "等待发掘",
            "activity": "等待发掘",
            "location": "等待发掘",
            "need": "等待发掘",
            "social_state": "等待发掘",
            "micro_desire": "等待发掘",
            "body_state": "等待发掘",
            "concern": "等待发掘",
        })
        # 不需要再手动 save，update_multi 里面已经保存了
        # 如果你想显式一点，也可以写：
        # _ssm.save()

    
    except Exception as e:
        print("snapshot 重置失败:", e)
        

        
    dpg.show_viewport()
    dpg.set_primary_window("root", True)

    dpg.start_dearpygui()
        
    orchestrator.stop_trigger_loop()
    dpg.destroy_context()


if __name__ == "__main__":
    main()