# models/llm/__init__.py
"""
LLM-based forecasting models (Direct Prompt & LLMP).
Mirrors structure of AutoARIMA package.
"""

from .direct_prompt import DirectPrompt
from .llm_processes import LLMPForecaster

__all__ = ["DirectPrompt", "LLMPForecaster"]
