import os
import textwrap

def load_and_format_llm_log(base_dir: str = None, max_chars: int = 80) -> str:
    """
    读取 data/prompt_logs/llm_prompt_log.txt，并按 UI 传入的 max_chars 自动换行。
    """

    # 如果传入的是完整路径，就直接用；否则认为是目录
    if base_dir and base_dir.endswith(".txt"):
        log_path = base_dir
    else:
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        log_path = os.path.join(base_dir, "data", "prompt_logs", "llm_prompt_log.txt")

    if not os.path.exists(log_path):
        return "（暂无 LLM 调用日志）"

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return "（暂无 LLM 调用日志）"

        # 自动按字符宽度换行
        wrapped = textwrap.fill(content, width=max_chars)
        return wrapped

    except Exception as e:
        return f"⚠ 无法读取 LLM 日志：{e}"
