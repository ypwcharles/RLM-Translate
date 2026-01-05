"""Core module for DeepTrans-RLM."""

from .state import TranslationState
from .rlm_context import RLMContext
from .chunker import TextChunker
from .client import create_llm_client
from .exceptions import (
    TranslationError,
    TokenLimitExceeded,
    GlossaryViolation,
    RLMContextError,
    APIRateLimitError,
    CollaborationConvergenceError,
)

__all__ = [
    "TranslationState",
    "RLMContext",
    "TextChunker",
    "create_llm_client",
    "TranslationError",
    "TokenLimitExceeded",
    "GlossaryViolation",
    "RLMContextError",
    "APIRateLimitError",
    "CollaborationConvergenceError",
]
