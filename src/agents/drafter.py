"""
Drafter Agent - 初翻 Agent (Addition Agent)

基于 TransAgents 论文的 Addition Agent 设计，负责产出详尽的初稿翻译。
"""

from typing import Dict, Any, Optional, List

from .base import BaseAgent
from ..core.client import ChatGoogleGenerativeAI


from pathlib import Path

# 获取 prompts 目录的绝对路径
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

class DrafterAgent(BaseAgent):
    """初翻 Agent (Addition Agent)。
    
    负责产出详尽的初稿翻译。遵循 Addition 原则：
    宁可多翻译一些内容，也不要遗漏。
    
    Attributes:
        source_lang: 源语言
        target_lang: 目标语言
        
    Example:
        >>> drafter = DrafterAgent(client)
        >>> translation = await drafter.process(source_text, context)
    """
    
    def __init__(
        self,
        client: ChatGoogleGenerativeAI,
        source_lang: str = "English",
        target_lang: str = "Chinese",
        prompt_template: Optional[str] = None,
        prompt_file: Optional[str] = None,
    ):
        """初始化 Drafter Agent。
        
        Args:
            client: LLM 客户端
            source_lang: 源语言
            target_lang: 目标语言
            prompt_template: 自定义 Prompt 模板
            prompt_file: Prompt 模板文件 (默认为 prompts/drafter.txt)
        """
        # 如果未指定 prompt_file 且没有 template，则使用默认文件
        if not prompt_template and not prompt_file:
            prompt_file = str(PROMPTS_DIR / "drafter.txt")
            
        super().__init__(
            name="drafter",
            client=client,
            prompt_template=prompt_template,
            prompt_file=prompt_file,
        )
        self.source_lang = source_lang
        self.target_lang = target_lang
        
    def _get_default_template(self) -> str:
        """获取默认 Prompt 模板。
        
        注意：现在默认通过 prompt_file 加载，此方法作为后备。
        """
        return "Error: Prompt template not found."
        
    async def process(
        self,
        source_text: str,
        context: Dict[str, Any],
        history: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """翻译源文本。
        
        Args:
            source_text: 源文本
            context: 上下文，包含 glossary, plot_summary 等
            history: 协作历史（用于多轮迭代）
            **kwargs: 其他参数
            
        Returns:
            翻译结果
        """
        # 格式化历史
        history_text = ""
        if history:
            history_text = "\n\n".join(history)
            
        # 构建 Prompt
        prompt = self.build_prompt(
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            source_text=source_text,
            glossary=context.get("glossary", ""),
            book_summary=context.get("book_summary", ""),
            plot_summary=context.get("plot_summary", ""),
            style_guide=context.get("style_guide", ""),
            target_audience=context.get("target_audience", "一般读者"),
            character_profiles=context.get("character_profiles", ""),
            history=history_text,
        )
        
        # 调用 LLM
        return await self.invoke(prompt)
        
    def process_sync(
        self,
        source_text: str,
        context: Dict[str, Any],
        history: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """同步翻译源文本。
        
        Args:
            source_text: 源文本
            context: 上下文
            history: 协作历史
            **kwargs: 其他参数
            
        Returns:
            翻译结果
        """
        # 格式化历史
        history_text = ""
        if history:
            history_text = "\n\n".join(history)
            
        # 构建 Prompt
        prompt = self.build_prompt(
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            source_text=source_text,
            glossary=context.get("glossary", ""),
            book_summary=context.get("book_summary", ""),
            plot_summary=context.get("plot_summary", ""),
            style_guide=context.get("style_guide", ""),
            target_audience=context.get("target_audience", "一般读者"),
            character_profiles=context.get("character_profiles", ""),
            history=history_text,
        )
        
        # 调用 LLM
        return self.invoke_sync(prompt)
