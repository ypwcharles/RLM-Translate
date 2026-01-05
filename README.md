# DeepTrans-RLM: Recursive Multi-Agent Translation System

基于 **RLM (Recursive Language Models)** 和 **TransAgents** 的长文本翻译系统，旨在实现"大师级"的小说翻译。

## 核心特性

- **RLM 范式**: 将长文本作为环境变量处理，按需访问而非直接注入上下文
- **Multi-Agent 协作**: 模拟人类编辑部流程（初翻 → 评审 → 润色）
- **Addition-by-Subtraction**: 迭代协作策略，确保翻译质量收敛
- **长期记忆管理**: 术语表、剧情摘要、角色档案全局传递
- **断点续传**: 支持从检查点恢复长文本翻译

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 设置 API Key

```bash
export GOOGLE_API_KEY="your-api-key"
```

### 基本用法

```bash
# 翻译文本文件
python scripts/run_translation.py input.txt

# 指定输出目录
python scripts/run_translation.py input.txt -o output/

# 启用调试模式
python scripts/run_translation.py input.txt --debug --verbose

# 从检查点恢复
python scripts/run_translation.py input.txt --resume-from checkpoint_name
```

## 项目结构

```
RLM-Translate/
├── src/
│   ├── core/           # 核心模块
│   │   ├── state.py    # TranslationState 定义
│   │   ├── rlm_context.py  # RLM 上下文管理器
│   │   ├── chunker.py  # 文本切分
│   │   └── client.py   # LLM 客户端
│   ├── agents/         # Agent 模块
│   │   ├── drafter.py  # 初翻 Agent (Addition)
│   │   ├── critic.py   # 评审 Agent (Subtraction)
│   │   ├── editor.py   # 润色 Agent
│   │   └── collaboration.py  # 协作策略
│   ├── memory/         # 记忆管理
│   │   ├── short_term.py  # 短期记忆
│   │   └── long_term.py   # 长期记忆
│   ├── graphs/         # LangGraph 工作流
│   │   ├── main_graph.py  # 主图 (RLM 控制层)
│   │   └── translation_subgraph.py  # 翻译子图
│   └── utils/          # 工具函数
├── prompts/            # Prompt 模板
├── config/             # 配置文件
├── scripts/            # 入口脚本
└── doc/                # 文档
```

## 核心概念

### RLM (Recursive Language Models)

将长 Prompt 作为环境变量，通过代码驱动的方式按需访问：

- **Prompt-as-Variable**: 完整文本作为变量存储
- **Selective Viewing**: 按需 peek 文本片段
- **Recursive Sub-calling**: 递归分解复杂任务

### TransAgents 协作

基于 Addition-by-Subtraction 策略：

1. **Addition Agent (Drafter)**: 产出详尽初稿
2. **Subtraction Agent (Critic)**: 审查并精简
3. **Early Exit**: 若无改动建议则提前终止
4. **Editor**: 结合反馈产出定稿

## 使用模型

- **Analyzer**: `gemini-2.0-flash` - 术语提取
- **Drafter**: `gemini-2.0-flash` - 高速初翻
- **Critic**: `gemini-2.0-flash` - 深度审查
- **Editor**: `gemini-2.0-flash` - 润色定稿

## 参考论文

- [TransAgents (2405.11804)](doc/2405.11804v2.pdf) - 多 Agent 文学翻译协作框架
- [RLM (2512.24601)](doc/2512.24601v1.pdf) - 递归语言模型

## License

MIT License
