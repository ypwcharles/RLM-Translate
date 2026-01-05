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
from ..core.chunker import TextChunker, ChunkerConfig
from ..core.client import LLMClientManager
from ..agents.drafter import DrafterAgent
from ..agents.critic import CriticAgent
from ..agents.editor import EditorAgent
from .translation_subgraph import TranslationSubgraph


logger = logging.getLogger(__name__)


# 术语提取 Prompt
ANALYZER_PROMPT = """You are an expert literary analyst. Your task is to scan the following text and extract all important terminology, proper nouns, and invented terms that need consistent translation.

## Text to Analyze
{text}

## Instructions
1. Identify ALL proper nouns (character names, place names, organization names)
2. Identify invented terms, fictional concepts, and world-specific vocabulary
3. Identify recurring important phrases that should be translated consistently
4. For each term, suggest an appropriate Chinese translation

## Output Format
Return a JSON object with the following structure:
{{
    "glossary": {{
        "English term": "Chinese translation",
        ...
    }},
    "characters": {{
        "Character Name": "Brief description",
        ...
    }},
    "book_summary": "A 2-3 sentence summary of the text"
}}

Return ONLY the JSON object, no additional text or markdown.
"""


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
    """
    
    def __init__(
        self,
        client_manager: LLMClientManager,
        chunker_config: Optional[ChunkerConfig] = None,
        max_iterations: int = 2,
        source_lang: str = "English",
        target_lang: str = "Chinese",
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
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_iterations = max_iterations
        
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
            
            # 构建分析 Prompt
            # 对于超长文本，可能需要分段分析
            sample_length = min(len(raw_text), 100000)  # 取样分析
            sample_text = raw_text[:sample_length]
            
            prompt = ANALYZER_PROMPT.format(text=sample_text)
            
            # 调用分析器
            response = self.client_manager.analyzer.invoke(prompt)
            content = response.content
            
            # 解析 JSON 响应
            try:
                # 尝试提取 JSON
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    data = {}
            except json.JSONDecodeError:
                logger.warning("Failed to parse analyzer response as JSON")
                data = {}
                
            return {
                "glossary": data.get("glossary", {}),
                "character_profiles": data.get("characters", {}),
                "book_summary": data.get("book_summary", ""),
            }
            
        def chunk_node(state: TranslationState) -> Dict:
            """Chunker 节点：切分文本。"""
            logger.info("[MainGraph] Chunking text...")
            
            raw_text = state["raw_text"]
            chunks = self.chunker.plan_chunks(raw_text)
            
            logger.info(f"[MainGraph] Created {len(chunks)} chunks")
            
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
    target_lang: str = "Chinese",
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
    client_manager = LLMClientManager(api_key=api_key)
    
    return MainGraph(
        client_manager=client_manager,
        chunker_config=chunker_config,
        max_iterations=max_iterations,
        source_lang=source_lang,
        target_lang=target_lang,
    )
