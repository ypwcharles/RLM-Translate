"""
Agent 协作模块

实现 TransAgents 论文中的 Addition-by-Subtraction 协作策略。
"""

from typing import Dict, Any, Optional, List, Generator
import logging

from .drafter import DrafterAgent
from .critic import CriticAgent
from .editor import EditorAgent
from ..memory.short_term import ShortTermMemory
from ..utils.debugger import TranslationDebugger


logger = logging.getLogger(__name__)


class TranslationCollaboration:
    """基于 TransAgents 的 Addition-by-Subtraction 协作策略。
    
    协作流程：
    1. Addition Agent (Drafter) 产出详尽初稿
    2. Subtraction Agent (Critic) 审查并反馈冗余/错误
    3. 迭代直到收敛或达到最大次数
    4. Editor 结合所有反馈产出最终定稿
    
    Attributes:
        drafter: 初翻 Agent
        critic: 评审 Agent
        editor: 润色 Agent
        max_iterations: 最大迭代次数
        
    Example:
        >>> collab = TranslationCollaboration(drafter, critic, editor)
        >>> result = await collab.translate(source_text, context)
    """
    
    MAX_ITERATIONS = 2  # TransAgents 论文默认值
    
    def __init__(
        self,
        drafter: DrafterAgent,
        critic: CriticAgent,
        editor: EditorAgent,
        max_iterations: int = MAX_ITERATIONS,
        debugger: Optional[TranslationDebugger] = None,
    ):
        """初始化协作器。
        
        Args:
            drafter: 初翻 Agent (Addition)
            critic: 评审 Agent (Subtraction)
            editor: 润色 Agent
            max_iterations: 最大迭代次数
            debugger: 调试器（可选）
        """
        self.drafter = drafter
        self.critic = critic
        self.editor = editor
        self.max_iterations = max_iterations
        self.debugger = debugger
        
    async def translate(
        self,
        source_text: str,
        context: Dict[str, Any],
        chunk_index: int = 0,
    ) -> Dict[str, Any]:
        """执行完整的协作翻译流程。
        
        Args:
            source_text: 源文本
            context: 上下文信息
            chunk_index: 当前块索引（用于调试）
            
        Returns:
            包含最终翻译和元数据的字典
        """
        memory = ShortTermMemory()
        
        # Phase 1: Addition-by-Subtraction 迭代
        draft = ""
        critique = ""
        iterations_used = 0
        
        for iteration in range(self.max_iterations):
            iterations_used = iteration + 1
            
            # Step 1: Drafter 产出初稿 (或使用 Patch 后的稿件)
            logger.info(f"Chunk {chunk_index}: Drafter iteration {iteration + 1}")
            
            non_local_draft_update = False
            
            # 检查是否有 Patch 后的 Draft (logic implemented inside loop end)
            # 这里需要一个状态变量，但由于 python 作用域，我们直接检查 draft 是否被修改且标记
            
            if locals().get("use_patched_draft", False):
                logger.info(f"Chunk {chunk_index}: Using patched draft, skipping regeneration.")
                # draft is already updated
                use_patched_draft = False
            else:
                history = [t.content for t in memory.get_history()]
                draft = await self.drafter.process(
                    source_text=source_text,
                    context=context,
                    history=history if history else None,
                )
            
            
            memory.add_turn("drafter", draft)
            
            # 保存调试信息
            if self.debugger:
                self.debugger.save_response(
                    agent_name="drafter",
                    response=draft,
                    chunk_index=chunk_index,
                    iteration=iteration,
                )
                
            # Step 2: Critic 审查
            logger.info(f"Chunk {chunk_index}: Critic iteration {iteration + 1}")
            
            critique = await self.critic.process(
                source_text=source_text,
                context=context,
                draft_translation=draft,
            )
            
            memory.add_turn("critic", critique)
            
            # 保存调试信息
            if self.debugger:
                self.debugger.save_response(
                    agent_name="critic",
                    response=critique,
                    chunk_index=chunk_index,
                    iteration=iteration,
                )
                
            # Early Exit: 如果无需更多修改
            if self.critic.check_convergence(critique):
                logger.info(f"Chunk {chunk_index}: Converged at iteration {iteration + 1}")
                break

                
        # Phase 2: Editor 最终润色
        logger.info(f"Chunk {chunk_index}: Editor finalizing")
        
        final_translation = await self.editor.process(
            source_text=source_text,
            context=context,
            draft_translation=draft,
            critique_comments=critique,
        )
        
        memory.add_turn("editor", final_translation)
        
        # 保存调试信息
        if self.debugger:
            self.debugger.save_response(
                agent_name="editor",
                response=final_translation,
                chunk_index=chunk_index,
                iteration=0,
            )
            
        return {
            "translation": final_translation,
            "draft": draft,
            "critique": critique,
            "iterations": iterations_used,
            "converged": self.critic.check_convergence(critique),
            "history": memory.to_dict(),
        }
        
    def translate_sync(
        self,
        source_text: str,
        context: Dict[str, Any],
        chunk_index: int = 0,
    ) -> Dict[str, Any]:
        """同步执行协作翻译流程。
        
        Args:
            source_text: 源文本
            context: 上下文信息
            chunk_index: 当前块索引
            
        Returns:
            包含最终翻译和元数据的字典
        """
        memory = ShortTermMemory()
        
        draft = ""
        critique = ""
        iterations_used = 0
        
        for iteration in range(self.max_iterations):
            iterations_used = iteration + 1
            
            logger.info(f"Chunk {chunk_index}: Drafter iteration {iteration + 1}")
            
            if locals().get("use_patched_draft", False):
                logger.info(f"Chunk {chunk_index}: Using patched draft, skipping regeneration.")
                use_patched_draft = False
            else:
                history = [t.content for t in memory.get_history()]
                draft = self.drafter.process_sync(
                    source_text=source_text,
                    context=context,
                    history=history if history else None,
                )
            
            
            memory.add_turn("drafter", draft)
            
            if self.debugger:
                self.debugger.save_response(
                    agent_name="drafter",
                    response=draft,
                    chunk_index=chunk_index,
                    iteration=iteration,
                )
                
            logger.info(f"Chunk {chunk_index}: Critic iteration {iteration + 1}")
            
            critique = self.critic.process_sync(
                source_text=source_text,
                context=context,
                draft_translation=draft,
            )
            
            memory.add_turn("critic", critique)
            
            if self.debugger:
                self.debugger.save_response(
                    agent_name="critic",
                    response=critique,
                    chunk_index=chunk_index,
                    iteration=iteration,
                )
                
            if self.critic.check_convergence(critique):
                logger.info(f"Chunk {chunk_index}: Converged at iteration {iteration + 1}")
                break
                
        logger.info(f"Chunk {chunk_index}: Editor finalizing")
        
        final_translation = self.editor.process_sync(
            source_text=source_text,
            context=context,
            draft_translation=draft,
            critique_comments=critique,
        )
        
        memory.add_turn("editor", final_translation)
        
        if self.debugger:
            self.debugger.save_response(
                agent_name="editor",
                response=final_translation,
                chunk_index=chunk_index,
                iteration=0,
            )
            
        return {
            "translation": final_translation,
            "draft": draft,
            "critique": critique,
            "iterations": iterations_used,
            "converged": self.critic.check_convergence(critique),
            "history": memory.to_dict(),
        }
        
    async def translate_stream(
        self,
        source_text: str,
        context: Dict[str, Any],
        chunk_index: int = 0,
    ) -> Generator[Dict[str, Any], None, None]:
        """流式执行协作翻译，返回中间状态。
        
        Args:
            source_text: 源文本
            context: 上下文信息
            chunk_index: 当前块索引
            
        Yields:
            每个阶段的状态更新
        """
        memory = ShortTermMemory()
        
        for iteration in range(self.max_iterations):
            # Drafter
            yield {
                "stage": "drafter",
                "iteration": iteration + 1,
                "status": "processing",
            }
            
            history = [t.content for t in memory.get_history()]
            draft = await self.drafter.process(
                source_text=source_text,
                context=context,
                history=history if history else None,
            )
            
            memory.add_turn("drafter", draft)
            
            yield {
                "stage": "drafter",
                "iteration": iteration + 1,
                "status": "completed",
                "content": draft[:200] + "..." if len(draft) > 200 else draft,
            }
            
            # Critic
            yield {
                "stage": "critic",
                "iteration": iteration + 1,
                "status": "processing",
            }
            
            critique = await self.critic.process(
                source_text=source_text,
                context=context,
                draft_translation=draft,
            )
            
            memory.add_turn("critic", critique)
            
            converged = self.critic.check_convergence(critique)
            
            yield {
                "stage": "critic",
                "iteration": iteration + 1,
                "status": "completed",
                "content": critique[:200] + "..." if len(critique) > 200 else critique,
                "converged": converged,
            }
            
            if converged:
                break
                
        # Editor
        yield {
            "stage": "editor",
            "status": "processing",
        }
        
        final_translation = await self.editor.process(
            source_text=source_text,
            context=context,
            draft_translation=draft,
            critique_comments=critique,
        )
        
        yield {
            "stage": "editor",
            "status": "completed",
            "content": final_translation,
            "final": True,
        }
