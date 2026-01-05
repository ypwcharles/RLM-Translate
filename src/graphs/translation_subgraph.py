"""
翻译子图 (Translation Subgraph)

实现 Agent 协作层的 LangGraph 工作流。
Drafter → Critic → Editor 循环协作。
"""

from typing import Dict, Any, Literal, Optional
import logging

from langgraph.graph import StateGraph, END

from ..core.state import TranslationState, update_state
from ..agents.drafter import DrafterAgent
from ..agents.critic import CriticAgent
from ..agents.editor import EditorAgent


logger = logging.getLogger(__name__)


class TranslationSubgraph:
    """翻译子图。
    
    实现单个文本块的 Agent 协作翻译流程。
    基于 TransAgents 的 Addition-by-Subtraction 策略。
    
    工作流：
    1. Drafter 产出初稿
    2. Critic 审查反馈
    3. 判断是否收敛
    4. 未收敛则重复 1-3
    5. Editor 产出最终定稿
    
    Attributes:
        drafter: 初翻 Agent
        critic: 评审 Agent
        editor: 润色 Agent
        max_iterations: 最大迭代次数
        graph: 编译后的工作流
    """
    
    def __init__(
        self,
        drafter: DrafterAgent,
        critic: CriticAgent,
        editor: EditorAgent,
        max_iterations: int = 2,
    ):
        """初始化翻译子图。
        
        Args:
            drafter: 初翻 Agent
            critic: 评审 Agent
            editor: 润色 Agent
            max_iterations: 最大迭代次数
        """
        self.drafter = drafter
        self.critic = critic
        self.editor = editor
        self.max_iterations = max_iterations
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 工作流。"""
        
        # 定义节点函数
        def draft_node(state: TranslationState) -> Dict:
            """Drafter 节点：产出初稿。"""
            logger.info(f"[Subgraph] Drafter processing chunk {state.get('current_chunk_index', 0)}")
            
            context = {
                "glossary": self._format_glossary(state.get("glossary", {})),
                "book_summary": state.get("book_summary", ""),
                "plot_summary": self._format_summaries(state.get("plot_summary", [])),
                "style_guide": state.get("style_guide", ""),
                "target_audience": state.get("target_audience", "一般读者"),
                "character_profiles": self._format_characters(state.get("character_profiles", {})),
            }
            
            draft = self.drafter.process_sync(
                source_text=state["current_source_text"],
                context=context,
            )
            
            return {
                "draft_translation": draft,
                "iteration_count": state.get("iteration_count", 0) + 1,
            }
            
        def critique_node(state: TranslationState) -> Dict:
            """Critic 节点：审查反馈。"""
            logger.info(f"[Subgraph] Critic reviewing chunk {state.get('current_chunk_index', 0)}")
            
            context = {
                "glossary": self._format_glossary(state.get("glossary", {})),
                "style_guide": state.get("style_guide", ""),
            }
            
            critique = self.critic.process_sync(
                source_text=state["current_source_text"],
                context=context,
                draft_translation=state["draft_translation"],
            )
            
            return {
                "critique_comments": critique,
            }
            
        def edit_node(state: TranslationState) -> Dict:
            """Editor 节点：精准修缮 (Surgical Edit)。"""
            logger.info(f"[Subgraph] Editor applying fixes to chunk {state.get('current_chunk_index', 0)}")
            
            context = {
                "glossary": self._format_glossary(state.get("glossary", {})),
                "style_guide": state.get("style_guide", ""),
                "target_audience": state.get("target_audience", "一般读者"),
            }
            
            # Editor takes draft and critique, produces better draft
            patched_text = self.editor.process_sync(
                source_text=state["current_source_text"],
                context=context,
                draft_translation=state["draft_translation"],
                critique_comments=state["critique_comments"],
            )
            
            return {
                "draft_translation": patched_text, # Update draft for next iteration
                "final_chunk_translation": patched_text, # Also update final candidate
                "iteration_count": state.get("iteration_count", 0) + 1,
            }
            
        def should_continue(state: TranslationState) -> Literal["continue", "end"]:
            """判断是否继续迭代。"""
            critique = state.get("critique_comments", "")
            iteration = state.get("iteration_count", 0)
            
            # 检查收敛条件
            if self.critic.check_convergence(critique):
                logger.info(f"[Subgraph] Converged at iteration {iteration}")
                return "end"
                
            # 检查迭代次数
            if iteration >= self.max_iterations:
                logger.info(f"[Subgraph] Max iterations reached ({iteration})")
                return "end"
            
            # 如果未收敛且未达最大次数，继续修缮
            return "continue"
            
        # 构建图
        workflow = StateGraph(TranslationState)
        
        # 添加节点
        workflow.add_node("draft", draft_node)
        workflow.add_node("critique", critique_node)
        workflow.add_node("edit", edit_node)
        
        # 设置入口点
        workflow.set_entry_point("draft")
        
        # 添加边
        workflow.add_edge("draft", "critique")
        workflow.add_edge("edit", "critique") # Edit 后再次审查
        
        workflow.add_conditional_edges(
            "critique",
            should_continue,
            {
                "continue": "edit",
                "end": END,
            }
        )
        
        # 编译图
        return workflow.compile()
        
    def _format_glossary(self, glossary: Dict[str, str]) -> str:
        """格式化术语表。"""
        if not glossary:
            return ""
        return "\n".join(f"- {src} → {tgt}" for src, tgt in glossary.items())
        
    def _format_summaries(self, summaries: list, count: int = 10) -> str:
        """格式化剧情摘要。"""
        if not summaries:
            return ""
        return "\n\n".join(summaries[-count:])
        
    def _format_characters(self, characters: Dict[str, str]) -> str:
        """格式化角色信息。"""
        if not characters:
            return ""
        return "\n".join(f"- {name}: {desc}" for name, desc in characters.items())
        
    def invoke(self, state: TranslationState) -> TranslationState:
        """执行翻译子图。
        
        Args:
            state: 输入状态
            
        Returns:
            更新后的状态
        """
        return self.graph.invoke(state)
        
    async def ainvoke(self, state: TranslationState) -> TranslationState:
        """异步执行翻译子图。
        
        Args:
            state: 输入状态
            
        Returns:
            更新后的状态
        """
        return await self.graph.ainvoke(state)


def create_translation_subgraph(
    drafter: DrafterAgent,
    critic: CriticAgent,
    editor: EditorAgent,
    max_iterations: int = 2,
) -> TranslationSubgraph:
    """创建翻译子图。
    
    Args:
        drafter: 初翻 Agent
        critic: 评审 Agent
        editor: 润色 Agent
        max_iterations: 最大迭代次数
        
    Returns:
        TranslationSubgraph 实例
    """
    return TranslationSubgraph(
        drafter=drafter,
        critic=critic,
        editor=editor,
        max_iterations=max_iterations,
    )
