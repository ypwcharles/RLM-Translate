"""
Editor Agent - 润色 Agent

负责结合评审反馈产出最终定稿翻译。
"""

from typing import Dict, Any, Optional

from .base import BaseAgent
from ..core.client import ChatGoogleGenerativeAI


DEFAULT_EDITOR_TEMPLATE = """# Role: Expert Literary Editor

You are a senior literary editor finalizing a translation from {{ source_lang }} to {{ target_lang }}.
Your task is to take the draft translation and the reviewer's feedback to produce a polished final version.

## Translation Guidelines

### Glossary (MUST follow exactly)
{{ glossary }}

### Style Guide
{{ style_guide }}

### Target Audience
{{ target_audience }}

## Original Source Text
{{ source_text }}

## Draft Translation
{{ draft_translation }}

## Reviewer Feedback
{{ critique_comments }}

## Your Task

1. Carefully review the draft translation alongside the reviewer's feedback
2. Address ALL issues identified by the reviewer
3. Ensure every glossary term is translated exactly as specified
4. Polish the language for natural flow in {{ target_lang }}
5. Maintain the author's voice and style
6. Preserve the original meaning and nuances
7. Do NOT introduce new errors while fixing identified issues

## Output

Provide ONLY the final polished translation.
No explanations, notes, or meta-commentary.
Maintain the original paragraph structure.
"""


class EditorAgent(BaseAgent):
    """润色 Agent。
    
    负责结合评审反馈产出最终定稿翻译。
    是 Addition-by-Subtraction 协作的最终环节。
    
    Attributes:
        source_lang: 源语言
        target_lang: 目标语言
        
    Example:
        >>> editor = EditorAgent(client)
        >>> final = await editor.process(
        ...     source_text, context, 
        ...     draft=draft, feedback=critique
        ... )
    """
    
    def __init__(
        self,
        client: ChatGoogleGenerativeAI,
        source_lang: str = "English",
        target_lang: str = "Chinese",
        prompt_template: Optional[str] = None,
        prompt_file: Optional[str] = None,
    ):
        """初始化 Editor Agent。
        
        Args:
            client: LLM 客户端
            source_lang: 源语言
            target_lang: 目标语言
            prompt_template: 自定义 Prompt 模板
            prompt_file: Prompt 模板文件
        """
        super().__init__(
            name="editor",
            client=client,
            prompt_template=prompt_template,
            prompt_file=prompt_file,
        )
        self.source_lang = source_lang
        self.target_lang = target_lang
        
    def _get_default_template(self) -> str:
        """获取默认 Prompt 模板。"""
        return DEFAULT_EDITOR_TEMPLATE
        
    async def process(
        self,
        source_text: str,
        context: Dict[str, Any],
        draft_translation: str = "",
        critique_comments: str = "",
        **kwargs,
    ) -> str:
        """润色翻译。
        
        Args:
            source_text: 源文本
            context: 上下文，包含 glossary, style_guide 等
            draft_translation: 初稿翻译
            critique_comments: 评审意见
            **kwargs: 其他参数
            
        Returns:
            润色后的最终翻译
        """
        prompt = self.build_prompt(
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            source_text=source_text,
            draft_translation=draft_translation,
            critique_comments=critique_comments,
            glossary=context.get("glossary", ""),
            style_guide=context.get("style_guide", ""),
            target_audience=context.get("target_audience", "一般读者"),
        )
        
        return await self.invoke(prompt)
        
    def process_sync(
        self,
        source_text: str,
        context: Dict[str, Any],
        draft_translation: str = "",
        critique_comments: str = "",
        **kwargs,
    ) -> str:
        """同步润色翻译。
        
        Args:
            source_text: 源文本
            context: 上下文
            draft_translation: 初稿翻译
            critique_comments: 评审意见
            **kwargs: 其他参数
            
        Returns:
            润色后的最终翻译
        """
        prompt = self.build_prompt(
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            source_text=source_text,
            draft_translation=draft_translation,
            critique_comments=critique_comments,
            glossary=context.get("glossary", ""),
            style_guide=context.get("style_guide", ""),
            target_audience=context.get("target_audience", "一般读者"),
        )
        
        return self.invoke_sync(prompt)
