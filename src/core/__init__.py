"""Core module for DeepTrans-RLM."""

from .state import TranslationState, ChunkInfo, create_initial_state, update_state
from .rlm_context import RLMContext, SearchResult
from .chunker import TextChunker, ChunkerConfig
from .client import (
    create_llm_client,
    create_analyzer_client,
    create_drafter_client,
    create_critic_client,
    create_editor_client,
    LLMClientManager,
)
from .dmxapi_client import DMXAPIClient, DMXAPIClientManager, GenerateResponse
from .exceptions import (
    TranslationError,
    TokenLimitExceeded,
    GlossaryViolation,
    RLMContextError,
    APIRateLimitError,
    CollaborationConvergenceError,
)

__all__ = [
    # State
    "TranslationState",
    "ChunkInfo",
    "create_initial_state",
    "update_state",
    # RLM Context
    "RLMContext",
    "SearchResult",
    # Chunker
    "TextChunker",
    "ChunkerConfig",
    # Clients
    "create_llm_client",
    "create_analyzer_client",
    "create_drafter_client",
    "create_critic_client",
    "create_editor_client",
    "LLMClientManager",
    # DMXAPI
    "DMXAPIClient",
    "DMXAPIClientManager",
    "GenerateResponse",
    # Exceptions
    "TranslationError",
    "TokenLimitExceeded",
    "GlossaryViolation",
    "RLMContextError",
    "APIRateLimitError",
    "CollaborationConvergenceError",
]
