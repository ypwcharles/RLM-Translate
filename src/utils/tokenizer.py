"""
Token 计数工具

基于 tiktoken 的精确 Token 计数，用于控制文本切分和 API 调用。
"""

import re
from typing import Optional
from functools import lru_cache

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


# Gemini 默认使用类似 cl100k_base 的编码
DEFAULT_ENCODING = "cl100k_base"


@lru_cache(maxsize=4)
def _get_encoding(encoding_name: str):
    """获取 tiktoken 编码器（带缓存）。"""
    if not TIKTOKEN_AVAILABLE:
        return None
    return tiktoken.get_encoding(encoding_name)


def count_tokens(
    text: str,
    encoding_name: str = DEFAULT_ENCODING,
) -> int:
    """计算文本的 Token 数量。
    
    使用 tiktoken 进行精确计数。如果 tiktoken 不可用，
    则使用简单估算（英文按空格分词 + 中文按字符）。
    
    Args:
        text: 要计数的文本
        encoding_name: tiktoken 编码名称
        
    Returns:
        Token 数量
        
    Example:
        >>> count_tokens("Hello, world!")
        4
    """
    if not text:
        return 0
        
    encoding = _get_encoding(encoding_name)
    
    if encoding is not None:
        return len(encoding.encode(text))
    else:
        # Fallback: 简单估算
        return _estimate_tokens(text)


def _estimate_tokens(text: str) -> int:
    """简单的 Token 估算。
    
    英文按空格分词，中文/日文/韩文按字符计数。
    这是一个粗略估算，实际使用时应优先使用 tiktoken。
    
    Args:
        text: 要估算的文本
        
    Returns:
        估算的 Token 数
    """
    # 英文单词数
    word_count = len(text.split())
    
    # CJK 字符数（中文、日文、韩文）
    cjk_pattern = r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]'
    cjk_count = len(re.findall(cjk_pattern, text))
    
    # 标点符号
    punct_count = len(re.findall(r'[^\w\s]', text))
    
    # 估算：英文词 + CJK 字符 + 标点按 0.5 计算
    return word_count + cjk_count + int(punct_count * 0.5)


class TokenCounter:
    """Token 计数器类。
    
    提供更灵活的 Token 计数功能，支持批量计数和缓存。
    
    Attributes:
        encoding_name: tiktoken 编码名称
        use_cache: 是否使用缓存
        
    Example:
        >>> counter = TokenCounter()
        >>> counter.count("Hello, world!")
        4
        >>> counter.count_batch(["Hello", "World"])
        [1, 1]
    """
    
    def __init__(
        self,
        encoding_name: str = DEFAULT_ENCODING,
        use_cache: bool = True,
    ):
        self.encoding_name = encoding_name
        self.use_cache = use_cache
        self._cache = {} if use_cache else None
        
    def count(self, text: str) -> int:
        """计算单个文本的 Token 数。
        
        Args:
            text: 要计数的文本
            
        Returns:
            Token 数量
        """
        if not text:
            return 0
            
        if self.use_cache:
            # 使用文本哈希作为缓存键
            cache_key = hash(text)
            if cache_key in self._cache:
                return self._cache[cache_key]
                
        token_count = count_tokens(text, self.encoding_name)
        
        if self.use_cache:
            self._cache[cache_key] = token_count
            
        return token_count
        
    def count_batch(self, texts: list) -> list:
        """批量计算 Token 数。
        
        Args:
            texts: 文本列表
            
        Returns:
            Token 数量列表
        """
        return [self.count(text) for text in texts]
        
    def clear_cache(self):
        """清空缓存。"""
        if self._cache:
            self._cache.clear()
            
    def estimate_cost(
        self,
        text: str,
        price_per_1m_tokens: float = 0.15,
    ) -> float:
        """估算 API 调用成本。
        
        Args:
            text: 要计数的文本
            price_per_1m_tokens: 每百万 Token 价格（美元）
            
        Returns:
            估算成本（美元）
        """
        token_count = self.count(text)
        return (token_count / 1_000_000) * price_per_1m_tokens
        
    def fits_in_context(
        self,
        text: str,
        context_window: int = 2_000_000,
    ) -> bool:
        """检查文本是否适合上下文窗口。
        
        Args:
            text: 要检查的文本
            context_window: 上下文窗口大小
            
        Returns:
            是否适合
        """
        return self.count(text) <= context_window
