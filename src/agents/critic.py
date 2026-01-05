"""
Critic Agent - 评审 Agent (Subtraction Agent)

基于 TransAgents 论文的 Subtraction Agent 设计，负责审查翻译并指出问题。
"""

from typing import Dict, Any, Optional

from .base import BaseAgent
from ..core.client import ChatGoogleGenerativeAI


DEFAULT_CRITIC_TEMPLATE = """# Role: Senior Editor (Subtraction Agent)

You are a senior editor reviewing a translation from {{ source_lang }} to {{ target_lang }}.
Your role is to identify and flag redundant, inaccurate, or inconsistent content.
Be critical but constructive.

## Translation Guidelines Reference

### Glossary (translations MUST match exactly)
{{ glossary }}

{% if style_guide %}
### Style Guide
{{ style_guide }}
{% endif %}

## Original Source Text
{{ source_text }}

## Draft Translation to Review
{{ draft_translation }}

## Review Criteria

### 1. Glossary Compliance (Critical)
- Check if EVERY term from the glossary is translated correctly
- Flag any deviations with the exact term and expected translation

### 2. Accuracy
- Identify mistranslations, semantic errors, or altered meanings
- Check for omissions - any content missing from the original
- Verify numerical values, names, and proper nouns

### 3. Fluency & Style
- Flag translation artifacts (translationese, awkward phrasing)
- Check for consistent character voice and tone
- Identify overly literal translations that sound unnatural

### 4. Redundancy
- Identify verbose or repetitive passages not in the original
- Flag unnecessary additions by the translator

### 5. Cultural Fidelity
- Check for lost cultural nuances or imagery
- Verify idioms and metaphors are appropriately adapted

## Output Format

If there are issues to address, structure your feedback as:

### Glossary Issues
- [Term]: Expected "X", found "Y" (Location: quote the context)

### Accuracy Issues
- Issue description (Location: quote the problematic text)
- Suggested fix: ...

### Style Issues
- Issue description
- Suggestion: ...

If the translation is excellent with no significant issues, respond ONLY with:
"No further changes required."

Remember: Be specific and actionable. Quote the exact problematic text.
"""


class CriticAgent(BaseAgent):
    """评审 Agent (Subtraction Agent)。
    
    负责审查翻译质量，按照 Subtraction 原则：
    识别并移除冗余、不准确或不一致的内容。
    
    Attributes:
        source_lang: 源语言
        target_lang: 目标语言
        
    Example:
        >>> critic = CriticAgent(client)
        >>> feedback = await critic.process(source_text, context, draft=translation)
    """
    
    def __init__(
        self,
        client: ChatGoogleGenerativeAI,
        source_lang: str = "English",
        target_lang: str = "Chinese",
        prompt_template: Optional[str] = None,
        prompt_file: Optional[str] = None,
    ):
        """初始化 Critic Agent。
        
        Args:
            client: LLM 客户端
            source_lang: 源语言
            target_lang: 目标语言
            prompt_template: 自定义 Prompt 模板
            prompt_file: Prompt 模板文件
        """
        super().__init__(
            name="critic",
            client=client,
            prompt_template=prompt_template,
            prompt_file=prompt_file,
        )
        self.source_lang = source_lang
        self.target_lang = target_lang
        
    def _get_default_template(self) -> str:
        """获取默认 Prompt 模板。"""
        return DEFAULT_CRITIC_TEMPLATE
        
    async def process(
        self,
        source_text: str,
        context: Dict[str, Any],
        draft_translation: str = "",
        **kwargs,
    ) -> str:
        """审查翻译。
        
        Args:
            source_text: 源文本
            context: 上下文，包含 glossary, style_guide 等
            draft_translation: 待审查的翻译稿
            **kwargs: 其他参数
            
        Returns:
            审查反馈
        """
        prompt = self.build_prompt(
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            source_text=source_text,
            draft_translation=draft_translation,
            glossary=context.get("glossary", ""),
            style_guide=context.get("style_guide", ""),
        )
        
        return await self.invoke(prompt)
        
    def process_sync(
        self,
        source_text: str,
        context: Dict[str, Any],
        draft_translation: str = "",
        **kwargs,
    ) -> str:
        """同步审查翻译。
        
        Args:
            source_text: 源文本
            context: 上下文
            draft_translation: 待审查的翻译稿
            **kwargs: 其他参数
            
        Returns:
            审查反馈
        """
        prompt = self.build_prompt(
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            source_text=source_text,
            draft_translation=draft_translation,
            glossary=context.get("glossary", ""),
            style_guide=context.get("style_guide", ""),
        )
        
        return self.invoke_sync(prompt)
        
    def check_convergence(self, feedback: str) -> bool:
        """检查是否达到收敛条件。
        
        Args:
            feedback: 审查反馈
            
        Returns:
            是否收敛（无需更多修改）
        """
        convergence_indicators = [
            "no further changes required",
            "no further changes needed",
            "no significant issues",
            "translation is excellent",
            "无需进一步修改",
            "翻译质量优秀",
        ]
        
        feedback_lower = feedback.lower()
        return any(indicator in feedback_lower for indicator in convergence_indicators)
