"""
LLM 客户端封装

配置 Gemini 3 系列模型，安全设置为 BLOCK_NONE 以支持文学翻译。
"""

import os
from typing import Optional, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types import HarmCategory, HarmBlockThreshold


# === 默认模型配置 ===
DEFAULT_MODELS = {
    "analyzer": "gemini-2.0-flash",
    "drafter": "gemini-2.0-flash",
    "critic": "gemini-2.0-flash",
    "editor": "gemini-2.0-flash",
}

DEFAULT_TEMPERATURE = 1.0


def get_safety_settings() -> Dict[HarmCategory, HarmBlockThreshold]:
    """获取文学翻译所需的安全设置。
    
    所有类别设为 BLOCK_NONE，以防止文学内容触发审查。
    
    Returns:
        安全设置字典
    """
    return {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }


def create_llm_client(
    model_name: str,
    temperature: float = DEFAULT_TEMPERATURE,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> ChatGoogleGenerativeAI:
    """创建配置好安全设置的 LLM 客户端。
    
    基于 PRD 要求：使用 Gemini 3 系列，Temperature 1.0，
    安全设置为 BLOCK_NONE。
    
    Args:
        model_name: 模型名称（如 gemini-3-pro, gemini-3-flash）
        temperature: 温度参数，默认 1.0
        api_key: Google API 密钥（可选，默认从环境变量读取）
        **kwargs: 传递给 ChatGoogleGenerativeAI 的其他参数
        
    Returns:
        配置好的 LLM 客户端
        
    Example:
        >>> client = create_llm_client("gemini-3-flash")
        >>> response = await client.ainvoke("Hello, world!")
    """
    if api_key is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key,
        safety_settings=get_safety_settings(),
        **kwargs,
    )


def create_analyzer_client(
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> ChatGoogleGenerativeAI:
    """创建术语分析器客户端。
    
    使用 Gemini 3 Pro，利用其超长上下文窗口进行全书扫描。
    
    Args:
        api_key: Google API 密钥
        **kwargs: 其他参数
        
    Returns:
        分析器客户端
    """
    return create_llm_client(
        model_name=DEFAULT_MODELS["analyzer"],
        api_key=api_key,
        **kwargs,
    )


def create_drafter_client(
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> ChatGoogleGenerativeAI:
    """创建初翻器客户端（Addition Agent）。
    
    使用 Gemini 3 Flash 进行高速初翻。
    
    Args:
        api_key: Google API 密钥
        **kwargs: 其他参数
        
    Returns:
        初翻器客户端
    """
    return create_llm_client(
        model_name=DEFAULT_MODELS["drafter"],
        api_key=api_key,
        **kwargs,
    )


def create_critic_client(
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> ChatGoogleGenerativeAI:
    """创建评审器客户端（Subtraction Agent）。
    
    使用 Gemini 3 Pro 进行深度审查。
    
    Args:
        api_key: Google API 密钥
        **kwargs: 其他参数
        
    Returns:
        评审器客户端
    """
    return create_llm_client(
        model_name=DEFAULT_MODELS["critic"],
        api_key=api_key,
        **kwargs,
    )


def create_editor_client(
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> ChatGoogleGenerativeAI:
    """创建润色器客户端。
    
    使用 Gemini 3 Flash 进行快速润色。
    
    Args:
        api_key: Google API 密钥
        **kwargs: 其他参数
        
    Returns:
        润色器客户端
    """
    return create_llm_client(
        model_name=DEFAULT_MODELS["editor"],
        api_key=api_key,
        **kwargs,
    )


class LLMClientManager:
    """LLM 客户端管理器。
    
    集中管理所有 Agent 所需的 LLM 客户端实例。
    
    Attributes:
        analyzer: 术语分析器客户端
        drafter: 初翻器客户端
        critic: 评审器客户端
        editor: 润色器客户端
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_overrides: Optional[Dict[str, str]] = None,
    ):
        """初始化客户端管理器。
        
        Args:
            api_key: Google API 密钥
            model_overrides: 模型名称覆盖配置
        """
        self.api_key = api_key
        self.model_overrides = model_overrides or {}
        
        self._analyzer: Optional[ChatGoogleGenerativeAI] = None
        self._drafter: Optional[ChatGoogleGenerativeAI] = None
        self._critic: Optional[ChatGoogleGenerativeAI] = None
        self._editor: Optional[ChatGoogleGenerativeAI] = None
        
    def _get_model_name(self, role: str) -> str:
        """获取角色对应的模型名称。"""
        return self.model_overrides.get(role, DEFAULT_MODELS[role])
        
    @property
    def analyzer(self) -> ChatGoogleGenerativeAI:
        """获取分析器客户端（懒加载）。"""
        if self._analyzer is None:
            self._analyzer = create_llm_client(
                model_name=self._get_model_name("analyzer"),
                api_key=self.api_key,
            )
        return self._analyzer
        
    @property
    def drafter(self) -> ChatGoogleGenerativeAI:
        """获取初翻器客户端（懒加载）。"""
        if self._drafter is None:
            self._drafter = create_llm_client(
                model_name=self._get_model_name("drafter"),
                api_key=self.api_key,
            )
        return self._drafter
        
    @property
    def critic(self) -> ChatGoogleGenerativeAI:
        """获取评审器客户端（懒加载）。"""
        if self._critic is None:
            self._critic = create_llm_client(
                model_name=self._get_model_name("critic"),
                api_key=self.api_key,
            )
        return self._critic
        
    @property
    def editor(self) -> ChatGoogleGenerativeAI:
        """获取润色器客户端（懒加载）。"""
        if self._editor is None:
            self._editor = create_llm_client(
                model_name=self._get_model_name("editor"),
                api_key=self.api_key,
            )
        return self._editor
