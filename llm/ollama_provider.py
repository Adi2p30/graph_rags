from __future__ import annotations

import time
from typing import Optional

import httpx
import ollama

import config
from llm.base import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = ollama.Client(host=config.OLLAMA_HOST)

    def complete(self, prompt: str, system: Optional[str] = None, json_mode: bool = False) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # The local Ollama server subprocess occasionally drops the connection
        # mid-request under sustained load (observed: `RemoteProtocolError:
        # Server disconnected` during a long batch run) and auto-restarts --
        # worth one retry rather than aborting a whole multi-topic run.
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    format="json" if json_mode else None,
                    options={"temperature": 0.2},
                )
                return response["message"]["content"]
            except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout) as e:
                last_error = e
                time.sleep(3 * (attempt + 1))
        raise last_error
