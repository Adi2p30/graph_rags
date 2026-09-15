from __future__ import annotations

from typing import Optional

import google.generativeai as genai

import config
from llm.base import LLMProvider


class GeminiProvider(LLMProvider):
    def __init__(self, model_name: str):
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set but LLM_PROVIDER=gemini")
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model_name = model_name

    def complete(self, prompt: str, system: Optional[str] = None, json_mode: bool = False) -> str:
        model = genai.GenerativeModel(
            self.model_name,
            system_instruction=system,
            generation_config={
                "temperature": 0.2,
                "response_mime_type": "application/json" if json_mode else "text/plain",
            },
        )
        response = model.generate_content(prompt)
        return response.text
