"""
LLM 客户端封装

支持 DMXAPI 中转站调用 Gemini 模型。
配置安全设置为 BLOCK_NONE 以支持文学翻译。
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

# === DMXAPI 中转站配置 ===
DMXAPI_BASE_URL = "https://www.dmxapi.cn"



def get_safety_settings() -> Dict[str, str]:
    """获取文学翻译所需的安全设置。
    
    所有类别设为 BLOCK_NONE，以防止文学内容触发审查。
    为了兼容性，使用字符串形式。
    
    Returns:
        安全设置字典
    """
    return {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }


def create_llm_client(
    model_name: str,
    temperature: float = DEFAULT_TEMPERATURE,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    use_dmxapi: bool = True,
    **kwargs: Any,
) -> ChatGoogleGenerativeAI:
    """创建配置好安全设置的 LLM 客户端。
    
    支持 DMXAPI 中转站和原生 Google API 两种模式。
    
    Args:
        model_name: 模型名称（如 gemini-2.0-flash, gemini-3-flash-preview）
        temperature: 温度参数，默认 1.0
        api_key: API 密钥（DMXAPI 密钥或 Google API 密钥）
        base_url: 自定义 API 基础 URL（优先级高于 use_dmxapi）
        use_dmxapi: 是否使用 DMXAPI 中转站（默认 True）
        **kwargs: 传递给 ChatGoogleGenerativeAI 的其他参数
        
    Returns:
        配置好的 LLM 客户端
    """
    if api_key is None:
        # 优先使用 DMXAPI 密钥
        api_key = os.environ.get("DMXAPI_KEY") or os.environ.get("GOOGLE_API_KEY")
        
    # 确定 base_url
    if base_url is None and use_dmxapi:
        base_url = os.environ.get("DMXAPI_BASE_URL", DMXAPI_BASE_URL)
        
    # 构建客户端参数
    client_kwargs = {
        "model": model_name,
        "temperature": temperature,
        "google_api_key": api_key,
        "safety_settings": get_safety_settings(),
        **kwargs,
    }
    
    # 如果使用 DMXAPI 中转站
    if base_url:
        # 设置 client_options 以支持自定义 endpoint
        # 注意: ChatGoogleGenerativeAI 可能会将 client_options 传递给底层客户端
        if "client_options" not in client_kwargs:
            client_kwargs["client_options"] = {"api_endpoint": base_url}
        else:
            client_kwargs["client_options"]["api_endpoint"] = base_url
            
        # 移除显式的 transport 参数，因为它会导致 Warning 且通常不是必需的
        # client_kwargs["transport"] = "rest"
        
        # 同时设置环境变量作为后备
        os.environ["GOOGLE_AI_STUDIO_API_ENDPOINT"] = base_url
        
    return ChatGoogleGenerativeAI(**client_kwargs)


def create_genai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """创建原生 Google GenAI 客户端（支持 DMXAPI）。
    
    使用 google.genai.Client，更好地支持 DMXAPI 中转站。
    
    Args:
        api_key: DMXAPI 或 Google API 密钥
        base_url: API 基础 URL
        
    Returns:
        google.genai.Client 实例
    """
    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "请安装 google-genai 包: pip install google-genai"
        )
        
    if api_key is None:
        api_key = os.environ.get("DMXAPI_KEY") or os.environ.get("GOOGLE_API_KEY")
        
    if base_url is None:
        base_url = os.environ.get("DMXAPI_BASE_URL", DMXAPI_BASE_URL)
        
    return genai.Client(
        api_key=api_key,
        http_options={"base_url": base_url}
    )


def create_analyzer_client(
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> ChatGoogleGenerativeAI:
    """创建术语分析器客户端。
    
    使用 Gemini 模型，利用其超长上下文窗口进行全书扫描。
    
    Args:
        api_key: API 密钥
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
    
    使用 Gemini Flash 进行高速初翻。
    
    Args:
        api_key: API 密钥
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
    
    使用 Gemini 模型进行深度审查。
    
    Args:
        api_key: API 密钥
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
    
    使用 Gemini Flash 进行快速润色。
    
    Args:
        api_key: API 密钥
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
    支持 DMXAPI 中转站。
    
    Attributes:
        analyzer: 术语分析器客户端
        drafter: 初翻器客户端
        critic: 评审器客户端
        editor: 润色器客户端
        
    Example:
        # 使用 DMXAPI 中转站（默认）
        >>> manager = LLMClientManager(api_key="sk-xxx")
        
        # 使用原生 Google API
        >>> manager = LLMClientManager(api_key="xxx", use_dmxapi=False)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        use_dmxapi: bool = True,
        model_overrides: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ):
        """初始化客户端管理器。
        
        Args:
            api_key: API 密钥（DMXAPI 或 Google）
            base_url: 自定义 API 基础 URL
            use_dmxapi: 是否使用 DMXAPI 中转站
            model_overrides: 模型名称覆盖配置
        """
        self.api_key = api_key
        self.base_url = base_url
        self.use_dmxapi = use_dmxapi
        self.model_overrides = model_overrides or {}
        self.timeout = timeout
        
        self._analyzer: Optional[ChatGoogleGenerativeAI] = None
        self._drafter: Optional[ChatGoogleGenerativeAI] = None
        self._critic: Optional[ChatGoogleGenerativeAI] = None
        self._editor: Optional[ChatGoogleGenerativeAI] = None
        
    def _get_model_name(self, role: str) -> str:
        """获取角色对应的模型名称。"""
        return self.model_overrides.get(role, DEFAULT_MODELS[role])
        
    def _create_client(self, role: str) -> ChatGoogleGenerativeAI:
        """创建指定角色的客户端。"""
        return create_llm_client(
            model_name=self._get_model_name(role),
            api_key=self.api_key,
            base_url=self.base_url,
            use_dmxapi=self.use_dmxapi,
            request_timeout=self.timeout,
        )
        
    @property
    def analyzer(self) -> ChatGoogleGenerativeAI:
        """获取分析器客户端（懒加载）。"""
        if self._analyzer is None:
            self._analyzer = self._create_client("analyzer")
        return self._analyzer
        
    @property
    def drafter(self) -> ChatGoogleGenerativeAI:
        """获取初翻器客户端（懒加载）。"""
        if self._drafter is None:
            self._drafter = self._create_client("drafter")
        return self._drafter
        
    @property
    def critic(self) -> ChatGoogleGenerativeAI:
        """获取评审器客户端（懒加载）。"""
        if self._critic is None:
            self._critic = self._create_client("critic")
        return self._critic
        
    @property
    def editor(self) -> ChatGoogleGenerativeAI:
        """获取润色器客户端（懒加载）。"""
        if self._editor is None:
            self._editor = self._create_client("editor")
        return self._editor
