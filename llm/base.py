"""LLM provider abstraction. Two roles are configured independently
(connection-finding vs. GraphRAG answering) so they can be different
models/providers, per the project's "different LLM for each stage" design.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Optional

import config


class LLMProvider(ABC):
    model_name: str

    @abstractmethod
    def complete(self, prompt: str, system: Optional[str] = None, json_mode: bool = False) -> str:
        ...

    def complete_json(self, prompt: str, system: Optional[str] = None, max_retries: int = 3) -> Optional[dict]:
        """Calls the model with JSON mode/schema hints and validates the result parses.
        Retries with an increasingly blunt reminder on failure; returns None if it
        never produces valid JSON (callers must handle that -- never fabricate a result)."""
        last_raw = ""
        for attempt in range(max_retries):
            reminder = "" if attempt == 0 else "\n\nYour previous reply was not valid JSON. Reply with ONLY a single valid JSON object, no prose, no markdown fences."
            last_raw = self.complete(prompt + reminder, system=system, json_mode=True)
            cleaned = _strip_code_fences(last_raw)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue
        return None


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def get_provider(role: str) -> LLMProvider:
    """role: 'connection' or 'answer'."""
    if config.LLM_PROVIDER == "gemini":
        from llm.gemini_provider import GeminiProvider
        model = config.GEMINI_CONNECTION_MODEL if role == "connection" else config.GEMINI_ANSWER_MODEL
        return GeminiProvider(model)
    from llm.ollama_provider import OllamaProvider
    model = config.OLLAMA_CONNECTION_MODEL if role == "connection" else config.OLLAMA_ANSWER_MODEL
    return OllamaProvider(model)
