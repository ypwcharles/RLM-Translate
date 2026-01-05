"""
Agent 基类

定义所有翻译 Agent 的基础接口和共享功能。
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from pathlib import Path

from jinja2 import Template

from ..core.client import ChatGoogleGenerativeAI


class BaseAgent(ABC):
    """Agent 基类。
    
    定义所有翻译 Agent 的共同接口和基础功能。
    
    Attributes:
        name: Agent 名称
        client: LLM 客户端
        prompt_template: Prompt 模板
    """
    
    def __init__(
        self,
        name: str,
        client: ChatGoogleGenerativeAI,
        prompt_template: Optional[str] = None,
        prompt_file: Optional[str] = None,
    ):
        """初始化 Agent。
        
        Args:
            name: Agent 名称
            client: LLM 客户端
            prompt_template: Prompt 模板字符串
            prompt_file: Prompt 模板文件路径
        """
        self.name = name
        self.client = client
        
        if prompt_template:
            self.prompt_template = prompt_template
        elif prompt_file:
            self.prompt_template = Path(prompt_file).read_text(encoding="utf-8")
        else:
            self.prompt_template = self._get_default_template()
            
    @abstractmethod
    def _get_default_template(self) -> str:
        """获取默认 Prompt 模板。
        
        子类必须实现此方法。
        
        Returns:
            默认 Prompt 模板
        """
        pass
        
    def build_prompt(self, **kwargs) -> str:
        """构建 Prompt。
        
        Args:
            **kwargs: 模板变量
            
        Returns:
            填充后的 Prompt
        """
        template = Template(self.prompt_template)
        return template.render(**kwargs)
        
    def _extract_content(self, content: Any) -> str:
        """从响应中提取文本内容。
        
        Args:
            content: 原始响应内容 (str 或 list)
            
        Returns:
            提取后的文本字符串
        """
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    parts.append(part["text"])
                elif hasattr(part, "text"): # For objects
                    parts.append(part.text)
            return "".join(parts)
            
        return str(content)

    async def invoke(self, prompt: str) -> str:
        """调用 LLM。
        
        Args:
            prompt: Prompt 内容
            
        Returns:
            LLM 响应
        """
        response = await self.client.ainvoke(prompt)
        return self._extract_content(response.content)
        
    def invoke_sync(self, prompt: str) -> str:
        """同步调用 LLM。
        
        Args:
            prompt: Prompt 内容
            
        Returns:
            LLM 响应
        """
        response = self.client.invoke(prompt)
        return self._extract_content(response.content)
        
    @abstractmethod
    async def process(
        self,
        source_text: str,
        context: Dict[str, Any],
        **kwargs,
    ) -> str:
        """处理文本。
        
        子类必须实现此方法。
        
        Args:
            source_text: 源文本
            context: 上下文信息（包含 glossary, summary 等）
            **kwargs: 其他参数
            
        Returns:
            处理结果
        """
        pass
