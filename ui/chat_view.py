import dearpygui.dearpygui as dpg

class ChatView:
    """
    聊天视图（无状态布局版）

    原则：
    - 不保存 y_offset
    - 每次 render 都从 0 开始算
    - 所有历史只存在 messages 里
    """

    def __init__(
        self,
        drawlist_tag: str,
        scroll_tag: str,
        view_height: int,
        view_width: int,
        draw_bubble_fn
    ):
        self.drawlist_tag = drawlist_tag
        self.scroll_tag = scroll_tag
        self.view_height = view_height
        self.view_width = view_width
        self.draw_bubble_fn = draw_bubble_fn

        self.messages = []   # [{"text": str, "side": "left|right"}]

    def add_message(self, text: str, side: str):
        if not text:
            return
        self.messages.append({"text": text, "side": side})
        self.render()

    def clear(self):
        self.messages.clear()
        self.render()

    def render(self):
        # 1. 清空 drawlist（只删孩子，不删节点）
        if dpg.does_item_exist(self.drawlist_tag):
            dpg.delete_item(self.drawlist_tag, children_only=True)

        # 2. 从 y = 0 重新排版（关键点）
        y = 0
        GAP = 24

        for msg in self.messages:
            bubble_h = self.draw_bubble_fn(
                text=msg["text"],
                side=msg["side"],
                y=y
            )
            y += bubble_h + GAP

        # 3. 更新高度 & 强制滚到底
        content_h = max(self.view_height, y + 20)

        try:
            dpg.configure_item(self.drawlist_tag, height=content_h)
            dpg.set_y_scroll(self.scroll_tag, content_h)
        except Exception:
            pass

