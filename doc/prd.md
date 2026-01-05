# DeepTrans-RLM: Recursive Multi-Agent Translation System

## 1. 项目概述 (Project Overview)

构建一个基于 **LangGraph** 的长文本翻译 Agent 系统，旨在实现“大师级”的小说翻译，能够稳定处理百万字级别的超长文本。

### 核心技术支柱
*   **RLM (Recursive Language Models)**: 借鉴 MIT 最新论文，将长文本处理视为“编程环境中的变量操作”。通过逻辑代码精确控制 Token 窗口、切分文本和管理全局状态，有效解决 LLM 隐性上下文记忆的局限。
*   **Multi-Agent Collaboration (TransAgents)**: 模拟人类编辑部的协作流程（初翻 -> 反思 -> 校对 -> 润色），通过多角色对抗与协作提升翻译质量。

### 终极目标
产出高一致性、保留文化隐喻、流畅且准确的译文。解决长文本翻译中常见的“术语不一”、“前文遗忘”以及“Token 溢出导致的逻辑降智”等痛点。

---

## 2. 系统架构设计 (System Architecture)

系统采用 **控制层 (Control Layer)** 与 **执行层 (Execution Layer)** 分层设计。

### 2.1 全局状态定义 (State Schema)

在 LangGraph 中定义全局 `TranslationState`，负责整个翻译生命周期的信息流转。

```python
from typing import TypedDict, List, Dict, Optional

class TranslationState(TypedDict):
    # --- RLM 变量 (全局记忆) ---
    raw_text: str                # 完整原文内容或路径
    glossary: Dict[str, str]     # 统一术语表 {原文: 译文}
    style_guide: str             # 风格指南 (如：古风、硬核科幻、海明威风格)
    plot_summary: List[str]      # 剧情摘要链 (用于保持上下文连贯)
    character_profiles: Dict     # 角色状态 (如：{'主角': '目前受伤状态'})
    
    # --- 运行时变量 (循环控制) ---
    chapter_map: List[Dict]      # 章节映射表 [{'title': '第一章', 'start': 0, 'end': 5000}]
    total_chunks: int            # 总任务块数
    current_chunk_index: int     # 当前处理进度索引
    current_source_text: str     # 当前待处理文本块
    
    # --- Agent 协作区 (临时变量) ---
    draft_translation: str       # 初翻草稿
    critique_comments: str       # 评审意见
    final_chunk_translation: str # 当前块定稿
    
    # --- 输出结果 ---
    completed_translations: List[str] # 已完成的译文列表
```

### 2.2 工作流设计 (Graph Topology)

系统由 **主图 (Main Graph)** 驱动 RLM 逻辑，嵌套 **翻译子图 (Translation Subgraph)** 实现 Agent 协作。

#### 主图 (Main Graph) - RLM 控制逻辑
1.  **Analyzer Node (Gemini 3 Pro)**: 利用 2M+ 超长上下文，一次性扫描全书提取人名、地名及核心术语，生成全局 `glossary`。
2.  **Structural Chunking Node (Python Logic)**: 基于章节及 Token 阈值进行动态切分，预生成 `chapter_map`。
3.  **Translation Subgraph**: 循环处理单个文本块。
4.  **Aggregation Node**: 接收子图输出，更新 `plot_summary` 并追加译文。

#### 翻译子图 (Translation Subgraph) - Agent 协作
*   **Drafter (Gemini 3 Flash)**: 高速初翻员，基于术语表和前情提要产出初稿。
*   **Critic (Gemini 3 Pro)**: 资深主编，检查术语一致性、漏译、翻译腔及文化内涵。
*   **Editor (Gemini 3 Flash)**: 润色专家，结合评审建议产出出版级译文，确保指代关系准确。

---

## 3. 详细开发需求 (Detailed Requirements)

### 3.1 预处理能力 (Preprocessing)
*   **模型选择**: 推荐使用 Gemini 3 Pro和Flash模型，利用其长窗口优势减少术语提取的碎片化。
*   **Prompt 策略**: "Scan the entire text. Identify all Proper Nouns and Invented Terms. Output strictly in JSON format."

### 3.2 动态切分策略 (Structural Chunking)
*   **原则**: 章节完整性优先。
*   **阈值设置**:
    *   **Max Output Limit**: ~21000 Tokens (Gemini 安全上限)。
    *   **Source Chunk Threshold**: 以章节为单位切分，不超过200k Tokens (为翻译膨胀预留空间)。

#### RLM 切分实现示例：
```python
import re

def plan_chunks(full_text: str, max_source_tokens: int = 4000) -> List[str]:
    """
    两级切分策略：
    Level 1: 识别章节头 (Regex: 第[一二三...]章 | Chapter \d+)
    Level 2: 若单章超过 max_source_tokens，则按段落 (\n\n) 进行二次切分
    """
    chapter_pattern = r"(^\s*第[一二三四五六七八九十百千万\d]+章.*?$|^\s*Chapter\s+\d+.*?$)"
    # 实现分块逻辑...
    # (此处省略具体实现细节)
    return chunks
```

### 3.3 Prompt 核心指令
| 角色 | 模型 | 核心职责 |
| :--- | :--- | :--- |
| **Drafter** | Flash | 准确还原语义，严格遵守术语表，参考 `{plot_summary}`。 |
| **Critic** | Pro | 苛刻审查，指出术语背离、文化丢失及语气不连贯处。 |
| **Editor** | Flash | 结合多方意见润色，执行 `{style_guide}`。 |

---

## 4. 技术栈规格 (Tech Stack)

*   **编排框架**: LangGraph (必选), LangChain Core.
*   **LLM 接口**: `langchain_google_genai`.
*   **模型配置**:
    *   **主力逻辑/高质量**: `gemini-3-flash` (Temp 1.0).
    *   **校对及反思**: `gemini-3-pro` (Temp 1.0).
    *   **高效初翻**: `gemini-3-flash` (Temp 1.0).
*   **安全配置**: 必须设置 `Safety Settings` 为 `BLOCK_NONE`，以防文学内容触发拦截。

---

## 5. 开发步骤 (Implementation RoadMap)

1.  **Phase 1**: 定义 `TranslationState` 与基础 Client 初始化。
2.  **Phase 2**: 实现基于正则与 Token 计数的 `plan_chunks` 逻辑。
3.  **Phase 3**: 构建多 Agent 协作的 `Translation Subgraph`。
4.  **Phase 4**: 实现全局 `Analysis Node` 进行术语提取。
5.  **Phase 5**: 优化长章节润色截断问题，完善 `plot_summary` 迭代逻辑。

---

## 6. 特别指令 (Special Instructions)

*   **Consistency**: `glossary` 变量在所有节点间强制传递。
*   **Context Optimization**: 充分利用 Gemini 的窗口，在 Prompt 中增加前 5-10 章的详细摘要。
*   **Error Handling**: 处理 API 限制，针对截断风险动态调整切分阈值。