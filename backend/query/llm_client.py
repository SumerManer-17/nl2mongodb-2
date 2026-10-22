"""Local Ollama client for LLM-based MongoDB query generation.

Configuration is read from environment variables so the model can be
switched without changing code:

    OLLAMA_URL        - Ollama chat endpoint (default http://localhost:11434/api/chat)
    OLLAMA_MODEL      - model name (default qwen2.5:0.5b)
    OLLAMA_TIMEOUT    - request timeout in seconds (default 120)

The small ``qwen2.5:0.5b`` model is known to be unreliable for structured
query generation (it wraps output in markdown, adds a ``{"filter": ...}``
wrapper, and uses non-standard operator names).  A larger local model such as
``qwen2.5:3b`` is recommended when available.  Set ``OLLAMA_MODEL`` to use it.
"""

from __future__ import annotations

import os
import json
from typing import Optional

import requests


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))


class OllamaError(RuntimeError):
    """Raised when the local Ollama service is unavailable or returns an error."""


def generate_response(
    prompt: str,
    *,
    system: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    timeout: Optional[float] = None,
    use_json_mode: bool = True,
) -> str:
    """Send a chat request to the local Ollama API and return the message text.

    Parameters
    ----------
    prompt:
        The user prompt to send.
    system:
        Optional system instruction.  When omitted a generic assistant system
        message is used.
    model:
        Ollama model name.  Defaults to ``OLLAMA_MODEL`` env var or
        ``qwen2.5:0.5b``.
    temperature:
        Sampling temperature (0 = deterministic).
    timeout:
        Request timeout in seconds.  Defaults to ``OLLAMA_TIMEOUT``.
    use_json_mode:
        Request ``format: "json"`` from Ollama when supported.  The model is
        still not trusted to produce valid JSON; the caller must parse and
        validate the output.
    """
    if system is None:
        system = (
            "You are a helpful assistant that follows instructions precisely. "
            "Return only the requested output, no extra commentary."
        )

    payload = {
        "model": model or MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }

    if use_json_mode:
        payload["format"] = "json"

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=timeout or TIMEOUT,
        )
    except requests.RequestException as exc:
        raise OllamaError(
            f"Could not reach Ollama at {OLLAMA_URL}. "
            "Make sure Ollama is running (run `ollama serve`)."
        ) from exc

    if response.status_code >= 400:
        raise OllamaError(
            f"Ollama returned HTTP {response.status_code}: {response.text[:200]}"
        )

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise OllamaError("Ollama returned invalid JSON.") from exc

    if "message" not in data or "content" not in data["message"]:
        raise OllamaError("Ollama response did not contain a message.")

    return data["message"]["content"]


def is_available() -> bool:
    """Return True if the Ollama /api/tags endpoint responds."""
    try:
        tags_url = OLLAMA_URL.rsplit("/api/chat", 1)[0] + "/api/tags"
        response = requests.get(tags_url, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    print("Model:", MODEL_NAME)
    print("Available:", is_available())
    try:
        print(generate_response("Say hello in one word."))
    except OllamaError as exc:
        print("Ollama error:", exc)
