from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

import requests

from app.model_control import load_model_sampling_config


@dataclass
class LlamaServerConfig:
    base_url: str = field(default_factory=lambda: load_model_sampling_config().llama_base_url)
    default_slot_id: int | None = None
    timeout_seconds: int = 120


class LlamaServerClient:
    """Thin requests-based client for llama.cpp's llama-server."""

    def __init__(self, config: LlamaServerConfig | None = None) -> None:
        self.config = config or LlamaServerConfig()

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def props(self) -> dict[str, Any]:
        return self._get("/props")

    def completion(
        self,
        *,
        prompt: Any,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"prompt": prompt}
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if top_k is not None:
            payload["top_k"] = top_k
        if stop:
            payload["stop"] = stop
        if self.config.default_slot_id is not None:
            payload["id_slot"] = self.config.default_slot_id
        if extra_payload:
            payload.update(extra_payload)
        return self._post("/completion", payload)

    def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        response_format: dict[str, Any] | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if top_k is not None:
            payload["top_k"] = top_k
        if response_format:
            payload["response_format"] = response_format
        if extra_payload:
            payload.update(extra_payload)
        return self._post("/v1/chat/completions", payload)

    def stream_chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        response_format: dict[str, Any] | None = None,
        extra_payload: dict[str, Any] | None = None,
    ):
        payload: dict[str, Any] = {"messages": messages, "stream": True}
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if top_k is not None:
            payload["top_k"] = top_k
        if response_format:
            payload["response_format"] = response_format
        if extra_payload:
            payload.update(extra_payload)
        yield from self._post_stream("/v1/chat/completions", payload)

    def embeddings(
        self,
        *,
        input_text: str | list[str],
        extra_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"input": input_text}
        if extra_payload:
            payload.update(extra_payload)
        return self._post("/v1/embeddings", payload)

    def tokenize(self, content: str, add_special: bool = False, parse_special: bool = False) -> dict[str, Any]:
        payload = {
            "content": content,
            "add_special": add_special,
            "parse_special": parse_special,
        }
        return self._post("/tokenize", payload)

    def detokenize(self, tokens: list[int]) -> dict[str, Any]:
        return self._post("/detokenize", {"tokens": tokens})

    def _get(self, path: str) -> dict[str, Any]:
        response = requests.get(
            f"{self.config.base_url}{path}",
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(
            f"{self.config.base_url}{path}",
            json=payload,
            timeout=self.config.timeout_seconds,
        )
        if not response.ok:
            body = ""
            try:
                body = response.text.strip()
            except Exception:
                body = ""
            message = f"{response.status_code} Client Error for url: {response.url}"
            if body:
                message = f"{message} | Response body: {body}"
            raise requests.HTTPError(message, response=response)
        return response.json()

    def _post_stream(self, path: str, payload: dict[str, Any]):
        with requests.post(
            f"{self.config.base_url}{path}",
            json=payload,
            timeout=self.config.timeout_seconds,
            stream=True,
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                delta = choices[0].get("delta", {})
                reasoning = (
                    delta.get("reasoning")
                    or delta.get("reasoning_content")
                    or delta.get("thinking")
                    or delta.get("thinking_content")
                )
                if reasoning:
                    yield {"type": "reasoning", "text": str(reasoning)}
                content = delta.get("content")
                if content:
                    yield {"type": "content", "text": str(content)}
