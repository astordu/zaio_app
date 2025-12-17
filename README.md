# 在哦 DEMO - 主动沟通的知识机器人
- 首先看：

        【「在哦2025」我要卖下一代AI媒介-哔哩哔哩】 https://b23.tv/PGngjtE

- 进阶看：

        【「在哦demo」GIT项目解说-哔哩哔哩】 https://b23.tv/guFBNON

- 感谢好友**舜岳同学**的宣传

        【他不是 AI，是盯着你的灵魂镜！❤️‍🔥-哔哩哔哩】 https://b23.tv/mvVOnDr

- 感谢好友**程序员少北晨**的宣传

        【当AI娃娃机器人能够拥有人类的思维方式时？！-哔哩哔哩】 https://b23.tv/OY7S6mB

**众筹地址**

https://m.tb.cn/h.7YCTqev

**一个人摸爬滚打真的很心酸，请大家支持一下孩子**
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/e0fd8a59-1a62-4a79-94e2-2dd8f4dcb386" />

## 项目概述

**在哦 DEMO** 是一个创新的**知识机器人**，致力于成为一个 **主动沟通的智能体**。它不是传统意义上的智慧体，而是一个新型的对话媒介，通过深层交流帮助用户发现新的问题视角和知识洞见。

### 核心特性

- 🎯 **主动性** - 主动思考、推进话题
- 🔍 **审时度势** - 在因缘际会时推介对应观点
- 💬 **互动反馈** - 平等对话、双向反馈

### 您可以把项目扔给GPT

如果您不想要费事费时的研究项目，可以把它扔给GPT
然后问GPT

- 这个项目的项目结构如何啊？什么文件负责什么功能呢？
- 这个项目的大模型prompt都放在哪些文件里呢？我想看看具体prompt

## 使用场景

在哦 DEMO 特别适合以下场景：

- 需要深层次对话的知识探讨
- 寻求新的问题解决视角
- 自我认识和内心对话
- 知识梳理和总结输出

## 快速开始

### 环境要求

- Python 3.10 或更高版本
- pip 或 uv 包管理器

### 安装步骤

1. **克隆或解压项目文件**
   ```bash
   # 保持以下目录结构完整
   # assets/        - 图片资源（娃娃、渐变图等）
   # config/        - 配置文件
   # data/          - 数据文件（对话日志、状态等）
   # core/          - 核心引擎模块
   ```

2. **创建虚拟环境（可跳过使用默认环境）**

   或使用 pip：
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   ```

   使用 uv：
   ```bash
   uv venv
   source .venv/bin/activate  # macOS/Linux
   # 或
   .venv\Scripts\activate     # Windows
   ```

3. **安装依赖**
   ```bash
   uv pip install -r requirements.txt
   # 或
   pip install -r requirements.txt
   ```

4. **配置 API KEY**

   程序首次运行会提示输入 API KEY，或点击界面右侧"输入 API KEY"按钮：
   - 目前支持 DeepSeek API
   - 获取方式：https://platform.deepseek.com/
   - API KEY 格式：`sk-` 开头的一串字符

5. **启动程序**
   ```bash
   python main.py
   ```

## 程序运行
<img width="1038" height="613" alt="image" src="https://github.com/user-attachments/assets/13ea27a3-3622-445e-8313-6b598cf4a3a1" />

<img width="1038" height="613" alt="image" src="https://github.com/user-attachments/assets/ba0f6d79-9c1c-4aa2-bf83-bd7b13e04e13" />

### 界面组成

```
┌─────────────────────────────────────────┐
│          📌 在哦 DEMO版                  │
├──────────────┬──────────────┬───────────┤
│              │              │           │
│   娃娃虚拟形象 │  聊天气泡区域 │ 功能按钮栏 │
│   (ON/OFF状态)│  (左粉/右青)  │           │
│              │              │           │
├──────────────┴──────────────┴───────────┤
│  输入框                        发送 按钮 │
└─────────────────────────────────────────┘
```

### 工作流程

1. **初期交互** - 在哦会问一些问题了解你的情况
2. **主题推荐** - 转入观点引擎，推荐新的思考角度
3. **知识总结** - 进入总结引擎，梳理对话内容
4. **持续探索** - 可基于总结继续深层对话

### 交互提示

- 🔴 **娃娃灭灯（OFF）** - 在哦正在思考中，请耐心等待
- 🟢 **娃娃亮灯（ON）** - 在哦已准备好，可以继续交流

### 右侧功能按钮

| 按钮 | 功能描述 |
|-----|--------|
| 输入 API KEY | 配置或更新 DeepSeek API 密钥 |
| 查看人格引擎 | 查看 LLM 调用日志和提示词 |
| 回到初见 | 清空对话记录，重新开始 |
| 在哦理解你 | 查看当前状态快照（情绪、能量、活动等） |
| 观点树 | 查看自动生成的观点树结构 |
| 模拟忙碌 | 测试娃娃状态切换 |
| 时光飞逝下 | 触发时间跳过，让在哦思念你 |

## 项目结构

```
zaio_app_v2/
├── main.py
├── log_view_controller.py
├── run.ipynb
├── 文件结构说明.txt（本文件）
│
├── assets/
│
├── config/
│   ├── api_key.txt          ← 存储你的key
│   └── settings.yaml
│
├── core/
│   ├── __init__.py
│   ├── first_turn.py        ← 第一拍开场引擎
│   └── orchestrator.py      ← 我们刚刚改的数据流总控
│
├── data/
│   ├── current_state_snapshot.json
│   ├── tree_default.json
│   ├── logs/
│   │   └── ...（每天的对话日志，对应 对话_日期.txt）
│   ├── perspective_trees/
│   │   └── ...（观点树 JSON）
│   └── prompt_logs/
│       ├── llm_prompt_log.txt
│       └── llm_trigger_log.txt
│
├── llm/
│   ├── __init__.py
│   └── client.py            ← 所有 LLM 调用入口（role → prompt_map）重要节点
│
├── persona/
│   ├── __init__.py
│   ├── fast_engine.py       ← Q-Engine（快人格）
│   ├── slow_engine.py       ← T-Engine（慢人格）
│   ├── direct_engine.py     ← L-Engine（直答人格）
│   ├── deep_engine.py        ← D-Engine（描绘人格）
│   └── sum_engine.py        ← SUM-Engine（总结人格）
│
├── state/
│   ├── __init__.py
│   ├── history_manager.py   ← 对话历史管理
│   ├── snapshot_manager.py  ← current_state_snapshot 的读写
│   └── user_profile.py      ← 用户长期画像
│
├── thinking/
│   ├── __init__.py
│   ├── behavior_selector.py           ← 现在已经不负责选引擎了，只是历史遗留
│   ├── guess_engine.py
│   ├── perspective_generate_engine.py ← 观点树生成引擎
│   └── perspective_tree.py            ← 运行时观点树结构与操作
│
├── trigger/
│   ├── __init__.py
│   ├── engine_select_trigger.py       ← ✅ 现在真正的 Q/T/L/SUM 选择器
│   ├── perspective_move_trigger.py    ← T 引擎内部，决定怎么看树 / 跳节点
│   ├── state_update_trigger.py        ← 更新 snapshot 的 LLM 触发器
│   ├── talk_trigger.py                ← “要不要说话”的触发器
│   └── timing_engine.py               ← 非 LLM 的节奏/冷却引擎
│
└── ui/
    └── __init__.py    ← DearPyGUI 的 UI 逻辑（chat 窗口、娃娃、按钮等）
```

## 数据文件说明

### 对话日志
- 位置：`data/logs/对话_YYYYMMDD.txt`
- 内容：每日所有对话记录（用户 + AI）
- 格式：`[时间] [角色] 内容`

### LLM 日志
- 位置：`data/prompt_logs/llm_prompt_log.txt`
- 内容：所有 LLM API 调用的提示词和响应
- 用途：调试和优化人格引擎

### 状态快照
- 位置：`data/current_state_snapshot.json`
- 内容：在哦对用户状态的理解
- 字段包括：情绪、能量、活动、位置、需求等

### 观点树
- 位置：`data/perspective_trees/`
- 描述：自动生成的对话观点结构
- 格式：JSON，可通过"观点树"按钮查看

## 常见问题

### Q: 程序卡顿或崩溃？
A: 这是 DEMO 版本的已知问题。建议：
- 避免快速连续点击按钮
- 避免疯狂发送大量消息
- 给在哦充足的思考时间（娃娃灭灯时）

### Q: 如何更改 API 提供商？
A: 目前 DEMO 版默认使用 DeepSeek API。修改需在 `core/` 模块中调整 LLM 客户端配置。

### Q: 如何重新开始？
A: 点击右侧"回到初见"按钮，在哦会：
- 清空所有对话日志
- 清空 LLM 调用日志
- 重置状态快照
- 保留用户长期画像数据

### Q: macOS 显示"无法验证开发者"？
A: 系统设置 → 隐私与安全性 → 允许运行此应用

## 设计特色

### 观点树生成引擎
在哦使用 LLM 控制的观点树生成引擎，能够：
- 实时分析对话主题
- 自动推导相关观点
- 根据上下文调整讨论方向

### 排队系统
为了保证 API 调用的稳定性：
- 所有 LLM 请求按顺序排队
- 每个请求之间间隔 0.2 秒
- 避免高并发导致的 API 超时

### 思考状态反馈
- 娃娃灭灯（OFF）表示正在处理
- 娃娃亮灯（ON）表示已准备好
- 视觉反馈增强用户体验

## 局限性与改进方向

### 已知局限
1. **流程优化空间** - DEMO 版流程可能需进一步调整
2. **观点树稳定性** - LLM 生成的观点有时不够稳定
3. **长对话处理** - 长时间对话可能导致性能下降

### 后续改进方向
- 更强大的观点推导引擎
- 改进对话上下文管理
- 优化用户界面和交互体验
- 支持多 LLM 提供商
- 数据标注和模型训练

## 贡献与反馈

这是一个 VIBE CODING 的独立制作项目。我们期待社区的：
- 意见反馈
- 功能建议
- Bug 报告
- 代码改进

如遇任何问题，请：
1. 检查终端输出的错误信息（红字）
2. 确保项目目录结构完整
3. 验证 API KEY 配置
4. 提供详细的问题描述和截图
## 理论

### 1.什么是对话交流，交流的模型

**交流的模型， 并不是你一句话，我一句话，这么表面。**

**我们认为，两个人沟通实际上就是互相拟合的一个过程。** 这个过程如何发生的呢？ 

1. 首先同时发生两件事，
- 第一，**你听到的每句话，自己实际上都在猜，对面是什么意思，对面为什么要说这句话，对面说这句话脑回路是什么？对面在关心什么？** 

- 第二，人都会有预判，所以在人说第一句话之前，就会预判一个对方的意思。并在这个过程中搭建一个假设的对方大脑回路，

2. 其次。我们开始语言的组织，
- **实际上人是在对着，你假设的对方的脑回路说话。而不是真的能跟对方的脑子说话。**

- 接着，对方听到了你的言语，马上回想，他是在说什么。于是马上生成一个对方的预判，并在这个过程中搭建一个假设的对方大脑回路，

- 其次，对方，又会向着他搭建的，你的脑回路说话。

3. 一场交流就这么发生了，这是对话的本质，**实际上是四个灵魂的事情，其中2个被模拟猜测出来。我们都对着自己模拟的对方说话，对方刚好也听到了，** 所以也反应。这个有点缸中大脑的意味。
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/7d617892-cf4e-4b81-8392-174bf36f8120" />

其次，拟合。

我们认为在这个过程中，以一个人的角度看，拟合成功与否，以及这个拟合是否带来好处，会发生几种事情。

- 第一象限，拟合成功，**带来好处**，就是被模拟出来的两个思维链灵魂，越来越趋近于你看到的对方，这个就是双方，相识相知的过程。是一个被理解的交流

- 第二象限，拟合成功，带来坏处，就是你发现了对方其实帮不到你，你们不在同一个世界里活着，两个人没交集。是一种孤独的发现

- 第三象限，拟合失败，带来坏处，吵架吵起来其实就是因为这个，我们认为，每个人的本质都不是坏的，他都有自己的出发点，而且灵魂都是不坏的。坏的是我们误以为对方是一个邪恶的存在，是我们拟合对方是个坏心眼，对我不好的情况。于是我们和拟合出来的人吵架。

- 第四象限，拟合失败，**带来好处**，这种情况是对方帮到里你，但是对方都不知道怎么发生的，你像柯南一样眼睛一亮，就说我们就聊到这里吧，我有想法了！对方还是懵懵的，等待毛利小五郎时刻，其实这个想法是你和你拟合的对方产生的。这是一种美丽的误会。

<img width="1024" height="1024" alt="754d41aa50a5062e135e8cb74ec7747b" src="https://github.com/user-attachments/assets/9211f5d6-18c8-42f9-9484-a5630dd5390f" />

在哦追求第一，第四象限的结果。

- 对于在哦，怎么达到第一象限呢，就是首先你要自己能帮到对方，有新观念，因为拟合最终会成功，或者是你们讨论一件事，这件事的过程，是另一件事的启发。对方拟合你的过程中，产生的结论对另一件事有效。

- 对于在哦，怎么达到第四象限呢，就是表达要多带比喻，故事，描述，等等，让对方在没完全拟合你之前，从你的新观念产生的具体描绘中，他误会到另外一个拟合的情况，而这个情况能帮到他，或者是你们讨论一件事，这件事的过程，是另一件事的启发。对方拟合你的过程中，产生的结论对另一件事有效。

最终推论：
- 我们要有一个自己的新观念，因此要有观点树，因此，有观点树引擎。
- 我们的表达可以故意增加描绘，故事，情景假设，等等增加误会又帮到人的可能。因此，有DEEP引擎。
  
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/5af52b45-63df-441f-966a-5a615eaa69b2" />
  
## 如何玩转这个项目
---

# 在哦 · 开发者参与指南

下面分几部分说明：

* 如何改 Prompt / 对话风格
* 如何改存储结构（snapshot / logs / 用户画像 等）
* 如何扩展新的引擎或模式
* 如何本地跑起来 & 做最小修改测试
* 推荐的参与方式清单（轻量 → 中度 → 深度）

---

## 一、如何改 Prompt / 对话风格

### 1. 所有 system prompt 的总入口

**核心文件：**`llm/client.py`
**关键字段：**`LLMClient.__init__()` 里的 `self.role_prompts = {...}`

这里集中定义了所有角色 / 触发器的 system prompt，包括：

* 人格引擎：

  * `"persona_fast"`：Q-Engine 快人格（轻松闲聊 + 信息收集）
  * `"persona_slow"`：T-Engine 慢人格（结构化深聊）
  * `"persona_direct"`：L-Engine 知识引擎（知识讲解）
  * `"persona_sum"`：Sum-Engine 总结人格
  * `"persona_deep"`：D-Engine 深度补充人格
* 触发器：

  * `"trigger_should_speak"`：小触发器，是否继续说话
  * `"trigger_select_engine"`：选择 Q / T / L / SUM / D 的引擎选择器
  * `"trigger_state_update"`：根据对话更新 `state_snapshot`
  * `"trigger_perspective_move"`：T 引擎观点树推进
  * `"perspective_generate_engine"`：自动生成新的观点树结构
* 观点树生成引擎（核心思想亮点）：

  * `"perspective_generate_engine"`：让在哦demo带着一个新思想，推动对话，通过trigger_state_update收集的信息，这样相当于prompt是LLM调用生成的资料，解决方案又是用这个资料生成的prompt，最后递给人格引擎生成对话，就是LLM调用的三次方，这给回答带来的随意性，而这种随意性，又是符合对话场景的。导致对话常用常新（目前这个引擎的触发敏感度有点低，需要对话多轮，大家可以调一下敏感度）

**要改风格 /立场 /语气：**
→ 直接在 `self.role_prompts` 对应键下修改那大段中文说明即可。

例如：

* 想改 Q-Engine 语气更「毒舌一点」——改 `"persona_fast"` 那一段；
* 想让 T-Engine 更「哲学向」——改 `"persona_slow"` 那一段；
* 想让 L-Engine 更像教程/老师——改 `"persona_direct"` 那一段。

> ⚠ 注意：
>
> * Prompt 中很多地方会描述输入 JSON 的结构，比如 `user_state` / `snapshot` / `talk_history` 等。
>   如果你修改了这些字段结构，记得同时更新描述，否则 LLM 会误解。
> * prompt 只是说明，不要改 `role` 的键名（比如 `persona_fast` 这几个字符串），因为代码里是硬编码用这个名字调用的。

### 2. 每个引擎是怎么调用 prompt 的

人格引擎的调用都很规整：

* 快人格：`persona/fast_engine.py`

  * `FastEngine.respond()` → `self.llm.call_llm("persona_fast", payload, temperature=0.5)`
* 慢人格：`persona/slow_engine.py`

  * `SlowEngine.deep_reply()` → `self.llm.call_llm("persona_slow", ...)`
* 知识引擎：`persona/direct_engine.py`

  * `DirectEngine.answer()` → `self.llm.call_llm("persona_direct", ...)`
* 总结人格：`persona/sum_engine.py`

  * `SumEngine.summarize()` → `self.llm.call_llm("persona_sum", ...)`
* 深度人格：`persona/deep_engine.py`

  * `DeepEngine.deepen()` → `self.llm.call_llm("persona_deep", ...)`

**如果你只是要改风格**，通常只需要改 `llm/client.py` 的 prompt，
这些 `persona/*.py` 文件一般不需要动。

### 3. 触发器 prompt 在哪里

触发器的 system prompt 一样也在 `llm/client.py` 的 `self.role_prompts` 里：

* `"trigger_should_speak"`：对应 `trigger/talk_trigger.py` → `TalkTrigger.should_reply(...)`
* `"trigger_select_engine"`：对应 `trigger/engine_select_trigger.py` → `EngineSelectTrigger.select(...)`
* `"trigger_state_update"`：对应 `trigger/state_update_trigger.py`
* `"trigger_perspective_move"`：对应 `trigger/perspective_move_trigger.py`
* `"perspective_generate_engine"`：对应 `thinking/perspective_generate_engine.py`

如果你想改「何时说话 / 何时切换 Q/T/L/SUM/D」，
就改 `"trigger_should_speak"` 和 `"trigger_select_engine"` 那两段提示词。

---

## 二、如何改存储结构（Snapshot / Logs / 用户画像）

整个存储层目前是很薄的一层，基本是「随便放 dict」，约束都在 prompt 里。

### 1. 当前状态快照：`current_state_snapshot.json`

**文件：**

* `data/current_state_snapshot.json` —— 当前快照（运行时会被更新）
* `state/snapshot_manager.py` —— 封装读写

`StateSnapshotManager` 很简单：

```python
class StateSnapshotManager:
    def __init__(self, path: str):
        self.path = path
        self._data: Dict[str, Any] = {}
        self.load()

    def get(self) -> Dict[str, Any]:
        return dict(self._data)

    def update_multi(self, updates: Dict[str, Any]):
        self._data.update(updates)
        self.save()
```

**要点：**

* **没有强 schema 限制**，你可以往 snapshot 里随便加字段；
* 真正的「约定」存在于：

  * `data/current_state_snapshot.json` 的初始结构
  * `llm/client.py` 里各个 prompt 中对 snapshot 的描述
  * `trigger/state_update_trigger.py` 使用的 `"trigger_state_update"` prompt

**如果你要增加新字段（例如：`"workload"`, `"money_anxiety"`）：**

1. 在 `data/current_state_snapshot.json` 里加初始字段；
2. 修改 `llm/client.py` 里：

   * `"persona_fast"` / `"persona_slow"` 提到 user_state 的解释；
   * `"trigger_state_update"` 里的字段说明，让 LLM 知道该怎么推断新字段；
3. 重启程序后，新字段会自动出现在快照里，`StateSnapshotManager` 不用改代码。

### 2. 长期画像：`user_profile.json`

**文件：**

* `data/user_profile.json`
* `state/user_profile.py` → `UserProfileManager`

结构完全开放，也是一个 dict。
目前主要被：

* `core/first_turn.py`（FirstTurnEngine）用来在第一次见面时定制开场；
* 将来可以在触发器 / persona 里引入。

**扩展方式：**

* 想记录「长期偏好」：比如 `{"prefer_night_chat": true, "favorite_topics": ["工作", "创作"]}`
* 直接在 `user_profile.json` 编辑即可，或者运行中通过 `UserProfileManager.update_multi` 写入。

如果你要在 prompt 里用到这些字段，记得同样要更新相关 prompt 的说明。

### 3. 对话历史 & 文本日志

**结构化对话历史：**

* 管理类：`state/history_manager.py` → `HistoryManager`
* 存储在内存 `self.history` 中，用于：

  * 触发器 (`trigger/*`)
  * 人格引擎（Q/T/L/SUM/D）

如果你想改变「历史窗口大小」「时间戳格式」之类，可以在这个文件里改。

**TXT 对话日志：**

* 统一入口：`main.py` → `_append_log(side, text)`
* 日志路径：`data/logs/对话_YYYYMMDD.txt`
* 格式：`[时间] [USER/AI] 文本...`（所有换行被压成一行）

你可以：

* 改写 `_append_log()`，比如增加 JSON log 或者更多字段；
* 或者增加新的日志文件（如 CSV、JSONL）。

**LLM prompt 调试日志：**

* 文件路径：`data/prompt_logs/llm_prompt_log.txt`
* 写入位置：`llm/client.py` → `_append_trigger_log(...)`

里面会按引擎区分每次的 system prompt / user payload / LLM 回复，方便调试。

### 4. 观点树存储

**文件结构：**

* 默认树：`data/tree_default.json`
* 生成树目录：`data/perspective_trees/*.json`

管理类：

* `thinking/perspective_tree.py` → `PerspectiveTree`
* `thinking/perspective_generate_engine.py` → 生成新树 & 落盘

你可以：

* 手动写一棵新树 JSON 放到 `data/perspective_trees`，让程序通过「时光飞逝」或触发器去用；
* 改 `PerspectiveTree` 的结构，比如支持多层 children、标签等；
* 改生成引擎的 prompt，控制生成的树风格。

---

## 三、如何扩展新的引擎或模式（加一个「特殊模式」）

这里用一个**示例流程**说明：
假设你要增加一个新的模式 `"R"`（Reflection-Engine，自我反思人格），大致步骤如下：

### 步骤 1：写一个新的 persona 类

新建文件：`persona/reflection_engine.py`：

```python
from typing import Dict, Any, List
from llm.client import LLMClient

class ReflectionEngine:
    ROLE = "persona_reflection"

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def reflect(self, user_text: str, user_state: Dict[str, Any], talk_history: List[Dict[str, Any]]) -> str:
        payload = {
            "user_text": user_text,
            "user_state": user_state,
            "talk_history": talk_history,
        }
        reply = self.llm.call_llm(self.ROLE, payload, temperature=0.5)
        return reply or ""
```

### 步骤 2：在 `llm/client.py` 里增加对应 prompt

在 `self.role_prompts` 里面加一个条目：

```python
"persona_reflection": """
你是“在哦 · 反思人格 R-Engine”。
（这里写你的角色设定 + 输入 JSON 字段含义 + 输出风格要求）
""",
```

同时可以在 `engine_name_human_readable` 映射里加一个说明（方便以后做日志或 UI 展示）：

```python
self.engine_name_human_readable = {
    ...
    "persona_reflection": "R 引擎·反思人格",
}
```

### 步骤 3：让引擎选择器认得这个模式

**文件：**`trigger/engine_select_trigger.py`

1. 修改 `LLM` prompt（`"trigger_select_engine"`）的说明文本，让它知道有一个新的模式 `"R"`，并说明什么时候选：

   * 例如：**当用户明确提到要复盘 / 反思 / 总结教训时，选择 R。**
2. 在解析结果时允许 `"R"`：

```python
data = json.loads(raw)
mode = str(data.get("mode") or "Q").upper().strip()
if mode in ("Q", "T", "L", "SUM", "D", "R"):
    return mode
```

### 步骤 4：在 Orchestrator 里接上新引擎

**文件：**`core/orchestrator.py`

1. 引入新类并实例化：

```python
from persona.reflection_engine import ReflectionEngine

class ConversationOrchestrator:
    def __init__(...):
        ...
        self.reflection_engine = ReflectionEngine(self.llm_client)
```

2. 在 `_run_behavior(self, mode: str, user_text: str) -> str` 里增加分支：

```python
if mode == "R":
    talk_his = self.history_manager.get_talk_his(limit=10)
    return self.reflection_engine.reflect(
        user_text=user_text,
        user_state=user_state,
        talk_history=talk_his,
    )
```

这样当 `EngineSelectTrigger` 返回 `"R"` 时，系统就会调用你的新人格。

> 提示：
> 你也可以不放进自动选择，而是临时在 `_run_behavior` 里写个硬编码，比如在用户输入中检测某个关键字，然后强行进入 R 引擎。这适合「试验性引擎」。

---

## 四、如何本地跑起来 & 做最小修改测试

### 1. 基本运行步骤

这部分和现有的 `使用说明_简洁版.txt` 一致，简要列一下（给开发者看的版本）：

1. 安装 Python 3.10+
2. `pip install -r requirements.txt`
3. 在 `config/api_key.txt` 里填入可用的 DeepSeek API Key
4. 命令行进入项目根目录（包含 `main.py` 的那个文件夹）：

   ```bash
   python main.py
   ```
5. 会弹出一个固定大小的 UI 窗口：

   * 左侧：在哦娃娃 + 背景
   * 中间：聊天区（粉色 = 在哦，青色 = 用户）
   * 右侧：分析区 + 日志查看按钮等

### 2. 做「最小修改」的推荐流程

**（1）只改 prompt 看效果**

* 改动文件：`llm/client.py` 中某个 `role_prompts[...]`
* 保存文件，重新运行 `python main.py`
* 在 UI 里直接聊几句，看：

  * Q 模式：随便说「在吗」「睡了没」这一类，观察开场语气；
  * T 模式：说一些「我最近有点烦」+「帮我理理」看看结构；
  * L 模式：问「帮我科普一下 Markdown 换行」。

**（2）修改 snapshot 结构的最小闭环测试**

* 在 `data/current_state_snapshot.json` 里加一个新字段，比如 `"stress_level": "等待发掘"`；
* 在 `llm/client.py` 的 `"trigger_state_update"` prompt 中，

  * 增加对 `stress_level` 的解释；
* 重启后随便聊几句，

  * 看控制台是否有报错；
  * 打开 `data/current_state_snapshot.json`，确认新字段被写入。

**（3）加一个新模式的快速冒烟测试**

* 按上面「扩展新引擎」的步骤新增一个 `R` 模式；
* 在 `trigger/engine_select_trigger.py` 里，先简陋一点：

  * 如果 `user_text` 里出现「反思」两个字，就强行返回 `"R"`；
* 重启后输入「我想反思一下今天的事」，

  * 看终端 log / `data/prompt_logs/llm_prompt_log.txt`，
  * 确认引擎选择为 `"R"`，且返回内容来自新 prompt。

---

## 五、推荐的『参与方式清单』

最后，可以直接给其他开发者一个「参与菜单」：

### 1. 轻量玩法：**只改 Prompt / 对话风格**

适合：

* 想玩「人格调教」「话术风格」的同学；
* 不想碰太多 Python 逻辑。

推荐入口：

* `llm/client.py` → `self.role_prompts`

  * 修改各人格的说话风格、禁忌、输出长度；
  * 修改各触发器的逻辑偏好（比如更积极、更佛系）。
* 查看效果：

  * 运行 `main.py`，观察不同场景下的回答变化；
  * 若需要精细调试，看 `data/prompt_logs/llm_prompt_log.txt`。

### 2. 中度玩法：**改数据结构 / 存储逻辑**

适合：

* 对「用户画像」「状态快照」「日志」有自己一套想法；
* 想接入自己的数据分析、可视化工具。

推荐入口：

* `state/snapshot_manager.py` + `data/current_state_snapshot.json`

  * 加字段、改状态结构，让在哦能识别更多维度。
* `state/user_profile.py` + `data/user_profile.json`

  * 设计一个更丰富的长期画像结构。
* `state/history_manager.py`

  * 改历史长度、调整时间戳格式、扩展字段。
* `main.py` → `_append_log`

  * 改 TXT 日志格式，或者加一个 JSON 日志方案。
* 同时配合改：

  * `llm/client.py` 中与这些字段相关的 prompt；
  * `trigger/state_update_trigger.py` 的字段更新逻辑说明。

### 3. 深度玩法：**加新引擎 / 改触发器 / 改 UI**

适合：

* 想把这个框架当成「自家对话 OS」来玩；
* 愿意深入 orchestrator / trigger / UI 结构。

推荐方向：

1. **新引擎 / 新模式**

   * 新建 `persona/xxx_engine.py`
   * 在 `llm/client.py` 注册新 `role_prompt`
   * 在 `trigger/engine_select_trigger.py` 里加入新模式的选择策略
   * 在 `core/orchestrator.py` 的 `_run_behavior()` 中接入。

2. **改触发器逻辑**

   * `trigger/talk_trigger.py`：什么时候继续说话 / 闭嘴；
   * `trigger/engine_select_trigger.py`：不同场景下选择 Q/T/L/SUM/D；
   * `trigger/perspective_move_trigger.py`：T 模式下的节点推进策略；
   * `thinking/perspective_generate_engine.py`：自动生成观点树的规则。

3. **改 UI 行为**

   * `main.py`：窗口布局 / 聊天气泡 / 滚动条样式等；
   * `log_view_controller.py`：在右侧增加更多开发者调试工具（比如在线查看 snapshot / user_profile）。

---


## 愿景

> "既然三生了万物，那这个万物里面可能包含有 AGI，这是我们的奢望。"

在哦的目标是作为一个沟通媒介，在生活中存在 5-10 年，通过与用户的交互积累数据，最终探索能否达成真正的 AGI。

在哦DEMO是希望把我们的思维框架开源，与社区讨论共同进步

这不是一个竞速项目，而是一个对话的见证。

## 杂谈-另类的图灵测试

对于图灵测试，我们有一个新的看法，

那就是人其实是对人和在身边旧的事物产生留恋的，

那么回答，机器能思考么？机器是人么？是硅基生命么？，

可以依据这一点，从是非问题转化成一个度量问题， 

* 当我们在一个人面前消灭另一个人的时候，你肯定是不想让这种事发生，并且原因是，他是人。 

* 当我们在宠物狗的主人面前要消灭有的时候，主人的反应是不想要杀掉狗，而且好多主任的理由是，狗是我的儿子，女儿（它是人） 

* 那当我们在一个人面前杀掉一个ai的时候，有没有可能它的主人不想让我们杀死他，并且理由是，它是人 

如果有这种可能，主人和ai，经历的最短磨合时间是多少 

这个时间值是我们要努力的目标，让它越来越短，直到变为零。 

比如10年？1年？3个月？ 

这样就把图灵测试的是非答案，换成了一种量化的答案 

这个思考和数值，是我们把机器能力的测验，转化成人机关系的一种思考

## 许可证

本项目为教育和研究用途。详见项目许可证文件。

---

**感谢使用在哦 DEMO！** 🎉

希望它能成为你的思维伙伴，帮助你在纷乱的世界中找到清晰的问题视角。

*"充分的沟通，解决未见之问题。"*

<img width="300" height="300" alt="微信图片_20240427151732" src="https://github.com/user-attachments/assets/34455983-c567-401c-ac99-e1c69a2c1f7c" />
<img width="300" height="300" alt="9cc53f4294103210ea4117a037fb62fb" src="https://github.com/user-attachments/assets/0190ad25-e4e9-4f0f-87ce-fbac295b2d29" />
<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/fe9d4ede-c37d-4806-ba50-46c3196c88d1" />

# 🇺🇸 **For English Developers: How to Understand and Work With This Project Using AI**

This project is primarily documented in Chinese.
If you are an English-speaking developer, **don’t worry**—you can use ChatGPT (or any modern AI assistant) to fully understand every part of this repository without needing to know Chinese.

Below is a simple guide to help you do that effectively.

---

## ✅ **1. Load the entire project into ChatGPT**

When you want AI to help you understand or translate the repo:

1. Compress the entire project into a ZIP file
2. Upload it into ChatGPT
3. Use this prompt:

```
I’m an English-speaking developer. Please analyze this entire project.
Explain all code, architecture, folder structure, and design concepts in English.
Translate (but do not rewrite) all Chinese comments or documentation.
Act as my technical guide for this repo.
```

This lets the AI build a mental map of the repo and help you navigate it.

---

## ✅ **2. Ask ChatGPT to “convert the entire project into English”**

If you want your own English version of the repo:

```
Convert this entire project to English.
– Translate all comments to English
– Translate all documentation to English
– Keep variable names and code logic unchanged
– Preserve the folder structure
– Output a downloadable ZIP file with the translated content
```

ChatGPT will give you a full English clone of the project.
This is extremely useful for teams with bilingual developers.

---

## ✅ **3. Ask ChatGPT to explain any Chinese file or function**

Example:

```
Explain this file in English.
Tell me:
1) What it does
2) How it fits into the overall system
3) What each function is responsible for
4) What I can modify safely
```

Or for a function:

```
Explain this function in English.  
What does it do? Why does it exist? How does it interact with other modules?
```

---

## ✅ **4. Ask for an English mental model of the architecture**

```
Give me a high-level English explanation of this project:
– How does information flow?
– What is the role of Orchestrator?
– What do the triggers do?
– What are the personas?
– How does the state snapshot work?
– How does the perspective tree influence conversation?
```

You will receive a complete architectural description in English.

---

## ✅ **5. Ask ChatGPT to generate English onboarding materials**

For example:

```
Create an English onboarding guide for a new developer joining this project.
Explain:
– Key modules
– What to read first
– How to run the project
– How to modify behavior
– Where the extension points (hooks) are
```

Or:

```
Create an English summary of how to contribute to this project.
```

This is perfect for open-source contributors.

---

## ✅ **6. Ask ChatGPT to be your "English project mentor"**

```
From now on, act as my project mentor.
Whenever I paste a file, explain:
– What it does
– How it connects to the system
– Where it is called
– Whether I can modify it
– How to extend it
```

This is extremely valuable for new developers joining the team.

---

## ✅ **7. Ask ChatGPT to generate English diagrams**

```
Generate an English architecture diagram (Mermaid) for this project.
```

Or:

```
Summarize the system in a simple English sequence diagram.
```

Or:

```
Translate this Chinese Mermaid diagram into English.
```

---

# ⭐ FINAL NOTE

**You can use AI as a universal translator + technical mentor + architecture guide.**
Even if the codebase is in Chinese, English-speaking developers can fully navigate, modify, and extend the system using the prompts above.



