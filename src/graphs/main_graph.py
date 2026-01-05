"""
主图 (Main Graph) - RLM 控制层

实现 RLM 风格的翻译控制流程。
Analyzer → Chunker → Translation Loop → Aggregation
"""

from typing import Dict, Any, Literal, Optional
import logging
import json
import re

from langgraph.graph import StateGraph, END

from ..core.state import TranslationState, update_state, create_initial_state, ChunkInfo
from ..core.rlm_context import RLMContext
from ..core.rlm_context import RLMContext
from ..core.chunker import TextChunker, ChunkerConfig, ChunkInfo
from ..core.structure_scanner import StructureScanner
from ..core.client import LLMClientManager
from ..agents.drafter import DrafterAgent
from ..agents.critic import CriticAgent
from ..agents.editor import EditorAgent
from ..agents.editor import EditorAgent
from .translation_subgraph import TranslationSubgraph
from ..utils.debugger import TranslationDebugger


logger = logging.getLogger(__name__)


# 术语提取 Prompt
from pathlib import Path

# 获取 prompts 目录的绝对路径
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"
ANALYZER_PROMPT_FILE = PROMPTS_DIR / "analyzer.txt"

# 加载 Analyzer Prompt
if ANALYZER_PROMPT_FILE.exists():
    ANALYZER_PROMPT = ANALYZER_PROMPT_FILE.read_text(encoding="utf-8")
else:
    ANALYZER_PROMPT = "Error: Analyzer prompt file not found."


class MainGraph:
    """主图 - RLM 控制层。
    
    实现完整的长文本翻译流程：
    1. Analyzer: 全书扫描，提取术语和角色
    2. Chunker: 基于章节切分文本
    3. Translation Loop: 循环翻译每个块
    4. Aggregation: 汇总翻译结果
    
    Attributes:
        client_manager: LLM 客户端管理器
        chunker: 文本切分器
        translation_subgraph: 翻译子图
        graph: 编译后的工作流
        debugger: 调试器 (Optional)
    """
    
    def __init__(
        self,
        client_manager: LLMClientManager,
        chunker_config: Optional[ChunkerConfig] = None,
        max_iterations: int = 2,
        source_lang: str = "English",
        target_lang: str = "Simplified Chinese",
        debugger: Optional[TranslationDebugger] = None,
    ):
        """初始化主图。
        
        Args:
            client_manager: LLM 客户端管理器
            chunker_config: 切分器配置
            max_iterations: Agent 协作最大迭代次数
            source_lang: 源语言
            target_lang: 目标语言
        """
        self.client_manager = client_manager
        self.chunker = TextChunker(chunker_config)
        self.scanner = StructureScanner()
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.target_lang = target_lang
        self.max_iterations = max_iterations
        self.debugger = debugger
        
        # 创建 Agent
        self._drafter = DrafterAgent(
            client=client_manager.drafter,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        self._critic = CriticAgent(
            client=client_manager.critic,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        self._editor = EditorAgent(
            client=client_manager.editor,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        # 创建翻译子图
        self.translation_subgraph = TranslationSubgraph(
            drafter=self._drafter,
            critic=self._critic,
            editor=self._editor,
            max_iterations=max_iterations,
        )
        
        # 构建主图
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """构建主图。"""
        
        def analyze_node(state: TranslationState) -> Dict:
            """Analyzer 节点：提取术语和角色。"""
            logger.info("[MainGraph] Analyzing text for terminology...")
            
            raw_text = state["raw_text"]
            
            # 1. 运行结构扫描
            skeleton = self.scanner.scan(raw_text)
            
            # 将骨架转换为字符串供 Prompt 使用 (只取前 2000 个节点以防超限? 
            # 实际上 skeleton 很轻，全书一般也就在 context window 内)
            # 我们将其格式化为 Line: Content
            skeleton_text = ""
            for node in skeleton:
                if node['type'] == 'header_candidate':
                    skeleton_text += f"[L{node['line_number']}] {node['content_preview']}\n"
                elif node['type'] == 'empty':
                     pass # 忽略空行以节省 Prompt? 或者用 [GAP] 标记?
                     # skeleton_text += f"[L{node['line_number']}] [GAP]\n"
                else:
                    # Text block start
                    skeleton_text += f"[L{node['line_number']}] {node['content_preview']} ({node['length']} chars)\n"

            # 2. 构建分析 Prompt
            # 取样正文用于风格分析 (前 5000 字符)
            sample_length = min(len(raw_text), 5000)
            sample_text = raw_text[:sample_length]
            
            prompt = ANALYZER_PROMPT.format(
                text=sample_text,
                structural_skeleton=skeleton_text
            )
            
            # Debug Log: Analyzer Prompt
            if self.debugger:
                self.debugger.save_prompt(
                    agent_name="analyzer",
                    prompt=prompt,
                    chunk_index=-1, # Global analysis
                    metadata={"skeleton_length": len(skeleton_text)}
                )

            # 调用分析器
            response = self.client_manager.analyzer.invoke(prompt)
            content = response.content

            # Debug Log: Analyzer Response
            if self.debugger:
                self.debugger.save_response(
                    agent_name="analyzer",
                    response=str(content),
                    chunk_index=-1
                )
            
            # 处理列表类型的 content (Gemini 3 可能返回多模态格式)
            if isinstance(content, list):
                if all(isinstance(x, str) for x in content):
                    content = "".join(content)
                else:
                    # 尝试从字典列表提取文本 (OpenAI style / Anthropic style compatibility)
                    text_parts = []
                    for part in content:
                        if isinstance(part, str):
                            text_parts.append(part)
                        elif hasattr(part, "get"):
                            text_parts.append(part.get("text", ""))
                    content = "".join(text_parts)
            
            # 解析 JSON 响应
            # 解析 JSON 响应
            try:
                data = {}
                # 1. 优先尝试提取 Markdown 代码块
                code_block_match = re.search(r"```json\s*(.*?)```", content, re.DOTALL)
                if code_block_match:
                     try:
                        data = json.loads(code_block_match.group(1).strip())
                     except json.JSONDecodeError:
                        logger.warning("Failed to parse JSON from markdown code block, falling back to heuristic search.")

                # 2. 如果代码块解析失败，尝试启发式提取
                if not data:
                    # 寻找最外层的 {}
                    # stack based approach or simple greedy search?
                    # Greedy search from first { to last } is usually fine if no other braces exist in preamble
                    json_match = re.search(r'\{[\s\S]*\}', content)
                    if json_match:
                        try:
                            data = json.loads(json_match.group())
                        except json.JSONDecodeError:
                             logger.warning("Failed to parse analyzer response as JSON (heuristic)")
                    else:
                        logger.warning("No JSON object found in analyzer response")

            except Exception as e:
                logger.error(f"Error during JSON parsing: {e}")
                data = {}
                
            # 清洗 Glossary：去除拼音和注释
            cleaned_glossary = {}
            for k, v in data.get("glossary", {}).items():
                # 去除括号及其内容 (e.g., "Term (Pinyin)" -> "Term")
                clean_v = re.sub(r'\s*\(.*?\)', '', v).strip()
                # 转换由 Gemini 可能输出的繁体到简体 (虽然 Prompt 要求了，但为了保险)
                # 这里暂时依赖 LLM 的修正，因为引入专门的简繁转换库可能太重
                # 但去除干扰字符是最重要的
                cleaned_glossary[k] = clean_v

            return {
                "glossary": cleaned_glossary,
                "character_profiles": data.get("characters", {}),
                "book_summary": data.get("book_summary", ""),
                "chunk_plan": data.get("chunks", []),
                "formatted_skeleton": skeleton, # 保存原始骨架供后续映射使用
            }
            
        def chunk_node(state: TranslationState) -> Dict:
            """Chunker 节点：切分文本。
            
            优先尝试执行 Analyzer 的 Semantic Chunk Plan。
            如果 Plan 无效或缺失，回退到 Regex Chunker。
            """
            logger.info("[MainGraph] Chunking text...")
            
            raw_text = state["raw_text"]
            chunk_plan = state.get("chunk_plan", [])
            formatted_skeleton = state.get("formatted_skeleton", [])
            
            chunks = []
            
            if chunk_plan and formatted_skeleton:
                logger.info(f"[MainGraph] Executing Semantic Chunk Plan ({len(chunk_plan)} chunks)...")
                try:
                    # 将 plan (基于 skeleton 行号) 映射回 full_text 的 char offsets
                    lines = raw_text.split('\n')
                    
                    # 预计算每行的起始 offset
                    line_offsets = []
                    current_offset = 0
                    for line in lines:
                        line_offsets.append(current_offset)
                        current_offset += len(line) + 1 # +1 for newline
                        
                    for plan_item in chunk_plan:
                        start_line = int(plan_item.get("start_line", 0))
                        end_line = int(plan_item.get("end_line", len(lines)))
                        
                        # 边界检查
                        start_line = max(0, min(start_line, len(lines) - 1))
                        end_line = max(0, min(end_line, len(lines)))
                        
                        start_char = line_offsets[start_line]
                        # end_line exclude, so calculate end char of end_line - 1
                        if end_line >= len(lines):
                            end_char = len(raw_text)
                        else:
                            end_char = line_offsets[end_line]
                            
                        # 创建 ChunkInfo
                        # 重新计算 Token
                        chunk_text = raw_text[start_char:end_char]
                        token_count = self.chunker.tokenizer(chunk_text)
                        
                        chunks.append(ChunkInfo(
                            title=plan_item.get("title", f"Chunk {len(chunks)+1}"),
                            start=start_char,
                            end=end_char,
                            token_count=token_count
                        ))
                        
                except Exception as e:
                    logger.error(f"[MainGraph] Failed to execute semantic chunk plan: {e}. Fallback to Regex.")
                    chunks = []
            else:
                 logger.warning(f"[MainGraph] Semantic Chunking skipped. Plan exists: {bool(chunk_plan)}, Skeleton exists: {bool(formatted_skeleton)}")
                 if chunk_plan:
                     logger.warning(f"Chunk Plan Preview: {str(chunk_plan)[:200]}")
                 if formatted_skeleton:
                     logger.warning(f"Skeleton Preview: {str(formatted_skeleton)[:200]}")
            
            if not chunks:
                logger.info("[MainGraph] Using Regex Chunker (Fallback)...")
                chunks = self.chunker.plan_chunks(raw_text)
            
            logger.info(f"[MainGraph] Finalized {len(chunks)} chunks")
            
            return {
                "chapter_map": chunks,
                "total_chunks": len(chunks),
                "current_chunk_index": 0,
            }
            
        def peek_chunk_node(state: TranslationState) -> Dict:
            """Peek 节点：获取当前待翻译的块。"""
            chunk_index = state["current_chunk_index"]
            chapter_map = state["chapter_map"]
            
            if chunk_index >= len(chapter_map):
                return {"current_source_text": ""}
                
            chunk = chapter_map[chunk_index]
            raw_text = state["raw_text"]
            
            # RLM Selective Viewing
            current_text = raw_text[chunk["start"]:chunk["end"]]
            
            logger.info(f"[MainGraph] Peeking chunk {chunk_index + 1}/{len(chapter_map)}: {chunk.get('title', 'Unknown')}")
            
            return {
                "current_source_text": current_text,
                "current_chunk_title": chunk.get("title", f"Chunk {chunk_index + 1}"),
                "iteration_count": 0,  # 重置迭代计数
            }
            
        def translate_node(state: TranslationState) -> Dict:
            """Translation 节点：调用翻译子图。"""
            chunk_index = state["current_chunk_index"]
            logger.info(f"[MainGraph] Translating chunk {chunk_index + 1}")
            
            # 调用翻译子图
            result = self.translation_subgraph.invoke(state)
            
            return {
                "final_chunk_translation": result.get("final_chunk_translation", ""),
                "draft_translation": result.get("draft_translation", ""),
                "critique_comments": result.get("critique_comments", ""),
            }
            
        def aggregate_node(state: TranslationState) -> Dict:
            """Aggregation 节点：汇总翻译结果。"""
            chunk_index = state["current_chunk_index"]
            translation = state.get("final_chunk_translation", "")
            chunk_title = state.get("current_chunk_title", "")
            
            logger.info(f"[MainGraph] Aggregating chunk {chunk_index + 1}")
            
            # 追加翻译结果
            completed = list(state.get("completed_translations", []))
            completed.append(translation)
            
            # 追加剧情摘要（简化版）
            summaries = list(state.get("plot_summary", []))
            # 可选：生成章节摘要
            summary = f"{chunk_title}: 已翻译完成"
            summaries.append(summary)
            
            return {
                "completed_translations": completed,
                "plot_summary": summaries,
                "current_chunk_index": chunk_index + 1,
            }
            
        def should_continue(state: TranslationState) -> Literal["continue", "finish"]:
            """判断是否继续翻译下一个块。"""
            current_index = state.get("current_chunk_index", 0)
            total_chunks = state.get("total_chunks", 0)
            
            if current_index < total_chunks:
                return "continue"
            return "finish"
            
        # 构建图
        workflow = StateGraph(TranslationState)
        
        # 添加节点
        workflow.add_node("analyze", analyze_node)
        workflow.add_node("chunk", chunk_node)
        workflow.add_node("peek", peek_chunk_node)
        workflow.add_node("translate", translate_node)
        workflow.add_node("aggregate", aggregate_node)
        
        # 设置入口点
        workflow.set_entry_point("analyze")
        
        # 添加边
        workflow.add_edge("analyze", "chunk")
        workflow.add_edge("chunk", "peek")
        workflow.add_edge("peek", "translate")
        workflow.add_edge("translate", "aggregate")
        
        # 条件边：继续或结束
        workflow.add_conditional_edges(
            "aggregate",
            should_continue,
            {
                "continue": "peek",
                "finish": END,
            }
        )
        
        # 编译图
        return workflow.compile()
        
    def invoke(self, state: TranslationState) -> TranslationState:
        """执行主图。
        
        Args:
            state: 输入状态
            
        Returns:
            最终状态
        """
        return self.graph.invoke(state)
        
    async def ainvoke(self, state: TranslationState) -> TranslationState:
        """异步执行主图。
        
        Args:
            state: 输入状态
            
        Returns:
            最终状态
        """
        return await self.graph.ainvoke(state)
        
    def translate_text(
        self,
        text: str,
        style_guide: str = "",
        target_audience: str = "一般读者",
    ) -> Dict[str, Any]:
        """翻译完整文本。
        
        便捷方法，封装状态创建和结果提取。
        
        Args:
            text: 待翻译文本
            style_guide: 风格指南
            target_audience: 目标读者
            
        Returns:
            翻译结果字典
        """
        # 创建初始状态
        state = create_initial_state(
            raw_text=text,
            style_guide=style_guide,
            target_audience=target_audience,
        )
        
        # 执行翻译
        final_state = self.invoke(state)
        
        # 提取结果
        translations = final_state.get("completed_translations", [])
        
        return {
            "translations": translations,
            "full_translation": "\n\n".join(translations),
            "glossary": final_state.get("glossary", {}),
            "total_chunks": final_state.get("total_chunks", 0),
            "book_summary": final_state.get("book_summary", ""),
        }


def create_main_graph(
    api_key: Optional[str] = None,
    chunker_config: Optional[ChunkerConfig] = None,
    max_iterations: int = 2,
    source_lang: str = "English",
    target_lang: str = "Simplified Chinese",
    debugger: Optional[TranslationDebugger] = None,
) -> MainGraph:
    """创建主图。
    
    Args:
        api_key: Google API 密钥
        chunker_config: 切分器配置
        max_iterations: Agent 协作最大迭代次数
        source_lang: 源语言
        target_lang: 目标语言
        
    Returns:
        MainGraph 实例
    """
    # 加载模型配置
    import yaml
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "config" / "models.yaml"
    
    model_overrides = {}
    timeout = None
    
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                if config:
                    if "models" in config:
                        for role, settings in config["models"].items():
                            if "name" in settings:
                                model_overrides[role] = settings["name"]
                    
                    # 读取 API 超时配置
                    if "api" in config and "request_timeout" in config["api"]:
                        timeout = config["api"]["request_timeout"]
                        logger.info(f"Loaded request timeout: {timeout}s")
                        
            logger.info(f"Loaded model config from {config_path}")
            logger.debug(f"Model overrides: {model_overrides}")
        except Exception as e:
            logger.warning(f"Failed to load model config: {e}")
    else:
        logger.warning(f"Model config not found at {config_path}")
    
    # 传递 model_overrides 和 timeout 给 ClientManager
    client_manager = LLMClientManager(
        api_key=api_key,
        model_overrides=model_overrides,
        timeout=timeout
    )
    
    return MainGraph(
        client_manager=client_manager,
        chunker_config=chunker_config,
        max_iterations=max_iterations,
        source_lang=source_lang,
        target_lang=target_lang,
        debugger=debugger,
    )
