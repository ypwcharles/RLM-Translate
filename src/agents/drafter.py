"""
Drafter Agent - 初翻 Agent (Addition Agent)

基于 TransAgents 论文的 Addition Agent 设计，负责产出详尽的初稿翻译。
"""

from typing import Dict, Any, Optional, List

from .base import BaseAgent
from ..core.client import ChatGoogleGenerativeAI


DEFAULT_DRAFTER_TEMPLATE = """# Role: Expert Literary Translator (Addition Agent)

You are an expert literary translator working from {{ source_lang }} to {{ target_lang }}.
Your goal is to produce a comprehensive, detailed, and faithful translation.

## Translation Guidelines (Long-Term Memory)

### Glossary (MUST follow exactly)
{{ glossary }}

### Book Summary
{{ book_summary }}

### Recent Plot Context (Last chapters)
{{ plot_summary }}

### Style Guide
{{ style_guide }}

### Target Audience
{{ target_audience }}

{% if character_profiles %}
### Character Profiles
{{ character_profiles }}
{% endif %}

## Source Text to Translate
{{ source_text }}

## Instructions
1. Translate the ENTIRE source text faithfully - do NOT summarize or omit any content.
2. STRICTLY follow the glossary mappings above. Every term in the glossary must be translated exactly as specified.
3. Maintain consistency with the plot context and character development.
4. Preserve the author's writing style, tone, narrative voice, and cultural nuances.
5. Translate idioms and metaphors appropriately for the target culture while preserving meaning.
6. Err on the side of more detail rather than less (Addition principle).
7. Maintain paragraph structure and formatting of the original.

{% if history %}
## Previous Iterations
{{ history }}
{% endif %}

## Output
Provide ONLY the translated text. No explanations, no notes, no commentary.
Do not add translator's notes unless absolutely necessary for comprehension.
"""


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
            prompt_file: Prompt 模板文件
        """
        super().__init__(
            name="drafter",
            client=client,
            prompt_template=prompt_template,
            prompt_file=prompt_file,
        )
        self.source_lang = source_lang
        self.target_lang = target_lang
        
    def _get_default_template(self) -> str:
        """获取默认 Prompt 模板。"""
        return DEFAULT_DRAFTER_TEMPLATE
        
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
