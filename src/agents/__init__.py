"""Agents module for DeepTrans-RLM."""

from .base import BaseAgent
from .drafter import DrafterAgent
from .critic import CriticAgent
from .editor import EditorAgent
from .collaboration import TranslationCollaboration

__all__ = [
    "BaseAgent",
    "DrafterAgent",
    "CriticAgent",
    "EditorAgent",
    "TranslationCollaboration",
]
