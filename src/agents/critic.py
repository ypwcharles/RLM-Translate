"""
Critic Agent - 评审 Agent (Subtraction Agent)

基于 TransAgents 论文的 Subtraction Agent 设计，负责审查翻译并指出问题。
"""

from typing import Dict, Any, Optional, List
import json
import re

from .base import BaseAgent
from ..core.client import ChatGoogleGenerativeAI


from pathlib import Path

# 获取 prompts 目录的绝对路径
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

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
            prompt_file: Prompt 模板文件 (默认为 prompts/critic.txt)
        """
        # 如果未指定 prompt_file 且没有 template，则使用默认文件
        if not prompt_template and not prompt_file:
            prompt_file = str(PROMPTS_DIR / "critic.txt")
            
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
        return "Error: Prompt template not found."
        
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
        
        对于 JSON 格式反馈：
        1. 如果是列表 []，则收敛。
        2. 如果是字典 {"issues": []}，则收敛。
        
        Args:
            feedback: 审查反馈
            
        Returns:
            是否收敛（无需更多修改）
        """
        # 尝试解析 JSON
        try:
            data = self.parse_json_response(feedback)
            # Case 1: List (Legacy or simple format)
            if isinstance(data, list) and len(data) == 0:
                return True
            # Case 2: Dict (New format)
            if isinstance(data, dict):
                issues = data.get("issues", [])
                if isinstance(issues, list) and len(issues) == 0:
                    return True
        except Exception:
            pass

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

    def parse_json_response(self, response: str) -> Any:
        """从 LLM 响应中解析 JSON。
        
        Args:
            response: LLM 响应文本
            
        Returns:
            解析后的 JSON 对象 (List 或 Dict)
        """
        # 1. 尝试直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
            
        # 2. 尝试提取 Markdown 代码块 ```json ... ```
        match = re.search(r"```json\s*(.*?)```", response, re.DOTALL)
        if match:
            try:
                content = match.group(1).strip()
                return json.loads(content)
            except json.JSONDecodeError:
                pass
                
        # 3. 尝试提取可能被包裹的 JSON 对象 {...} 或 [...]
        # 优先寻找最外层的 {} 或 []
        
        # 尝试找 {...}
        match_dict = re.search(r"\{[\s\S]*\}", response)
        if match_dict:
             try:
                content = match_dict.group(0).strip()
                return json.loads(content)
             except json.JSONDecodeError:
                pass

        # 尝试找 [...]
        match_list = re.search(r"\[[\s\S]*\]", response)
        if match_list:
             try:
                content = match_list.group(0).strip()
                return json.loads(content)
             except json.JSONDecodeError:
                pass
                
        return []
