"""
DMXAPI 原生客户端

直接使用 google-genai SDK 调用 DMXAPI 中转站。
"""

import os
import json
import requests
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


# === DMXAPI 配置 ===
DMXAPI_BASE_URL = "https://www.dmxapi.cn"
DMXAPI_GENERATE_ENDPOINT = "/v1beta/models/{model}:generateContent"


@dataclass
class GenerateResponse:
    """生成响应"""
    text: str
    usage: Dict[str, int]
    raw: Dict[str, Any]


class DMXAPIClient:
    """DMXAPI 中转站客户端。
    
    直接调用 Gemini 原生 API 格式。
    
    Attributes:
        api_key: DMXAPI 密钥
        base_url: API 基础 URL
        model: 默认模型
        
    Example:
        >>> client = DMXAPIClient(api_key="sk-xxx")
        >>> response = client.generate("Hello, world!")
        >>> print(response.text)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        temperature: float = 1.0,
        timeout: int = 120,
    ):
        """初始化 DMXAPI 客户端。
        
        Args:
            api_key: DMXAPI 密钥（或从环境变量 DMXAPI_KEY 读取）
            base_url: API 基础 URL
            model: 默认模型名称
            temperature: 温度参数
            timeout: 请求超时时间（秒）
        """
        self.api_key = api_key or os.environ.get("DMXAPI_KEY")
        if not self.api_key:
            raise ValueError(
                "DMXAPI 密钥未设置。请设置环境变量 DMXAPI_KEY 或传入 api_key 参数。"
            )
            
        self.base_url = base_url or os.environ.get("DMXAPI_BASE_URL", DMXAPI_BASE_URL)
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        
    def _build_url(self, model: Optional[str] = None) -> str:
        """构建 API URL。"""
        model = model or self.model
        endpoint = DMXAPI_GENERATE_ENDPOINT.format(model=model)
        return f"{self.base_url}{endpoint}"
        
    def _build_headers(self) -> Dict[str, str]:
        """构建请求头。"""
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        system_instruction: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
    ) -> GenerateResponse:
        """生成文本。
        
        Args:
            prompt: 用户提示
            model: 模型名称（覆盖默认）
            temperature: 温度参数
            system_instruction: 系统指令
            max_output_tokens: 最大输出 token 数
            
        Returns:
            GenerateResponse 对象
        """
        url = self._build_url(model)
        headers = self._build_headers()
        
        # 构建请求体
        payload: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ]
        }
        
        # 生成配置
        generation_config: Dict[str, Any] = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        elif self.temperature is not None:
            generation_config["temperature"] = self.temperature
        if max_output_tokens is not None:
            generation_config["maxOutputTokens"] = max_output_tokens
            
        if generation_config:
            payload["generationConfig"] = generation_config
            
        # 系统指令
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }
            
        # 发送请求
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        
        data = response.json()
        
        # 解析响应
        text = ""
        if "candidates" in data and len(data["candidates"]) > 0:
            candidate = data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                text = "".join(part.get("text", "") for part in parts)
                
        usage = data.get("usageMetadata", {})
        
        return GenerateResponse(
            text=text,
            usage={
                "prompt_tokens": usage.get("promptTokenCount", 0),
                "completion_tokens": usage.get("candidatesTokenCount", 0),
                "total_tokens": usage.get("totalTokenCount", 0),
            },
            raw=data,
        )
        
    def generate_with_history(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        system_instruction: Optional[str] = None,
    ) -> GenerateResponse:
        """带历史记录的生成。
        
        Args:
            messages: 消息列表，每个消息包含 role 和 content
            model: 模型名称
            temperature: 温度参数
            system_instruction: 系统指令
            
        Returns:
            GenerateResponse 对象
        """
        url = self._build_url(model)
        headers = self._build_headers()
        
        # 转换消息格式
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            # Gemini 使用 "user" 和 "model"
            if role == "assistant":
                role = "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg.get("content", "")}]
            })
            
        payload: Dict[str, Any] = {"contents": contents}
        
        # 生成配置
        generation_config: Dict[str, Any] = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        elif self.temperature is not None:
            generation_config["temperature"] = self.temperature
            
        if generation_config:
            payload["generationConfig"] = generation_config
            
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }
            
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        
        data = response.json()
        
        text = ""
        if "candidates" in data and len(data["candidates"]) > 0:
            candidate = data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                text = "".join(part.get("text", "") for part in parts)
                
        usage = data.get("usageMetadata", {})
        
        return GenerateResponse(
            text=text,
            usage={
                "prompt_tokens": usage.get("promptTokenCount", 0),
                "completion_tokens": usage.get("candidatesTokenCount", 0),
                "total_tokens": usage.get("totalTokenCount", 0),
            },
            raw=data,
        )
        
    def invoke(self, prompt: str) -> str:
        """简单调用接口（兼容 LangChain 风格）。
        
        Args:
            prompt: 用户提示
            
        Returns:
            生成的文本
        """
        response = self.generate(prompt)
        return response.text
        
    async def ainvoke(self, prompt: str) -> str:
        """异步调用接口。
        
        注意：当前实现是同步的，未来可以使用 aiohttp 实现真正的异步。
        
        Args:
            prompt: 用户提示
            
        Returns:
            生成的文本
        """
        return self.invoke(prompt)


class DMXAPIClientManager:
    """DMXAPI 客户端管理器。
    
    管理不同角色的 DMXAPI 客户端。
    
    Example:
        >>> manager = DMXAPIClientManager(api_key="sk-xxx")
        >>> response = manager.drafter.generate("翻译以下文本...")
    """
    
    DEFAULT_MODELS = {
        "analyzer": "gemini-2.0-flash",
        "drafter": "gemini-2.0-flash",
        "critic": "gemini-2.0-flash",
        "editor": "gemini-2.0-flash",
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_overrides: Optional[Dict[str, str]] = None,
        temperature: float = 1.0,
    ):
        """初始化管理器。
        
        Args:
            api_key: DMXAPI 密钥
            base_url: API 基础 URL
            model_overrides: 模型覆盖配置
            temperature: 默认温度参数
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model_overrides = model_overrides or {}
        self.temperature = temperature
        
        self._clients: Dict[str, DMXAPIClient] = {}
        
    def _get_model(self, role: str) -> str:
        """获取角色对应的模型。"""
        return self.model_overrides.get(role, self.DEFAULT_MODELS[role])
        
    def _get_client(self, role: str) -> DMXAPIClient:
        """获取或创建角色客户端。"""
        if role not in self._clients:
            self._clients[role] = DMXAPIClient(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self._get_model(role),
                temperature=self.temperature,
            )
        return self._clients[role]
        
    @property
    def analyzer(self) -> DMXAPIClient:
        """分析器客户端。"""
        return self._get_client("analyzer")
        
    @property
    def drafter(self) -> DMXAPIClient:
        """初翻器客户端。"""
        return self._get_client("drafter")
        
    @property
    def critic(self) -> DMXAPIClient:
        """评审器客户端。"""
        return self._get_client("critic")
        
    @property
    def editor(self) -> DMXAPIClient:
        """润色器客户端。"""
        return self._get_client("editor")
