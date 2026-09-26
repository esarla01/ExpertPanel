"""Thin wrapper around the Vertex AI Gemini client.

All model calls go through here so retry logic, caching and error
handling live in one place.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_client: Optional[genai.Client] = None

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CACHE_DIR = Path(os.getenv("CACHE_DIR", ".cache"))


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY environment variable is not set")
        _client = genai.Client(vertexai=True, api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Cache helpers (development only — avoids burning quota on repeated runs)
# ---------------------------------------------------------------------------

def _cache_key(system: str, prompt: str) -> str:
    raw = f"{MODEL}::{system}::{prompt}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _read_cache(key: str) -> Optional[str]:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["text"]
    return None


def _write_cache(key: str, text: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.json").write_text(json.dumps({"text": text}))


# ---------------------------------------------------------------------------
# Main call
# ---------------------------------------------------------------------------

def generate(
    prompt: str,
    *,
    system: str = "",
    use_cache: bool = True,
    max_retries: int = 3,
    temperature: float = 1.0,
) -> str:
    """Send a prompt to Gemini and return the text response.

    Retries on transient failures with exponential backoff.
    Caches responses keyed on (model, system, prompt) to save quota
    during development.  Disable with use_cache=False.
    """
    # Check cache first
    if use_cache:
        key = _cache_key(system, prompt)
        cached = _read_cache(key)
        if cached is not None:
            return cached

    client = _get_client()

    config = types.GenerateContentConfig(
        temperature=temperature,
    )
    if system:
        config.system_instruction = system

    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=config,
            )
            text = response.text
            if text is None:
                raise ValueError("Model returned an empty response")

            if use_cache:
                _write_cache(key, text)
            return text

        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)

    raise RuntimeError(
        f"Gemini call failed after {max_retries} attempts: {last_error}"
    )
