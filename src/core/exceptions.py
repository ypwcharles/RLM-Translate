"""
自定义异常类

定义翻译流程中可能出现的各种异常情况。
"""


class TranslationError(Exception):
    """翻译流程基础异常。
    
    所有翻译相关异常的基类。
    """
    pass


class TokenLimitExceeded(TranslationError):
    """Token 超限异常。
    
    当文本块超过 Token 限制时抛出。
    
    Attributes:
        chunk_index: 超限块的索引
        token_count: 实际 Token 数
        limit: Token 限制
    """
    
    def __init__(self, chunk_index: int, token_count: int, limit: int):
        self.chunk_index = chunk_index
        self.token_count = token_count
        self.limit = limit
        super().__init__(
            f"Chunk {chunk_index} exceeds token limit: {token_count} > {limit}"
        )


class GlossaryViolation(TranslationError):
    """术语表违规异常。
    
    当翻译结果违反术语表约束时抛出。
    
    Attributes:
        term: 原文术语
        expected: 期望的译文
        actual: 实际的译文
    """
    
    def __init__(self, term: str, expected: str, actual: str):
        self.term = term
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Glossary violation: '{term}' should be '{expected}', got '{actual}'"
        )


class RLMContextError(TranslationError):
    """RLM 上下文操作异常。
    
    当 RLM 上下文操作（peek, search 等）失败时抛出。
    """
    pass


class CollaborationConvergenceError(TranslationError):
    """Addition-by-Subtraction 协作未收敛异常。
    
    当协作迭代达到最大次数但翻译质量仍不满足要求时抛出。
    
    Attributes:
        max_iterations: 最大迭代次数
        last_feedback: 最后一次评审反馈
    """
    
    def __init__(self, max_iterations: int, last_feedback: str = ""):
        self.max_iterations = max_iterations
        self.last_feedback = last_feedback
        super().__init__(
            f"Collaboration did not converge after {max_iterations} iterations. "
            f"Last feedback: {last_feedback[:200]}..."
        )


class APIRateLimitError(TranslationError):
    """API 限流异常。
    
    当 API 调用遇到速率限制时抛出。
    
    Attributes:
        retry_after: 建议的重试等待时间（秒）
    """
    
    def __init__(self, message: str = "", retry_after: int = 60):
        self.retry_after = retry_after
        super().__init__(
            f"API rate limit exceeded. {message} Retry after {retry_after}s."
        )


class ChunkingError(TranslationError):
    """文本切分异常。
    
    当文本切分失败时抛出。
    """
    pass


class PromptTemplateError(TranslationError):
    """Prompt 模板异常。
    
    当 Prompt 模板渲染失败时抛出。
    """
    pass


class CheckpointError(TranslationError):
    """检查点异常。
    
    当检查点保存或加载失败时抛出。
    """
    pass


class StateValidationError(TranslationError):
    """状态验证异常。
    
    当 TranslationState 验证失败时抛出。
    """
    pass
