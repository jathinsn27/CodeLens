"""
Unified LLM interface: local GGUF (dev) or remote API (prod).

Set LLM_MODE=local (default) to use llama-cpp-python with GGUF files.
Set LLM_MODE=remote to call an OpenAI-compatible API.

Environment variables:
    LLM_MODE        - "local" or "remote" (default: local)
    LLM_API_URL     - API base URL for remote mode
    LLM_API_KEY     - API key for remote mode
    LLM_MODEL_NAME  - Model name for remote mode (e.g. "qwen3-4b")
"""

import os
import requests

_llama_cpp = None
_local_model = None
_mode = None
_model_name = None


def _get_mode():
    return os.getenv("LLM_MODE", "local")


def init_llm(model_paths: list[tuple[str, str]] | None = None):
    """
    Initialize the LLM backend.
    For local mode, pass a list of (path, name) tuples to try in order.
    For remote mode, no model loading needed.
    """
    global _local_model, _llama_cpp, _mode, _model_name
    _mode = _get_mode()

    if _mode == "remote":
        _model_name = os.getenv("LLM_MODEL_NAME", "remote")
        print(f"LLM mode: remote ({os.getenv('LLM_API_URL', 'not set')})")
        return

    # Local mode: load GGUF via llama-cpp-python
    try:
        from llama_cpp import Llama
        _llama_cpp = Llama
    except ImportError:
        print("WARNING: llama-cpp-python not installed. LLM features disabled.")
        return

    if model_paths is None:
        model_paths = []

    for path, name in model_paths:
        if os.path.exists(path):
            print(f"Loading {name} LLM from {path}...")
            _local_model = Llama(
                model_path=path,
                n_ctx=4096,
                n_batch=512,
                n_threads=4,
                verbose=False,
            )
            _model_name = name
            print(f"{name} LLM loaded.")
            return

    print("WARNING: No LLM model found. LLM features disabled.")


def get_llm_response(system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
    """Generate a chat completion using either local or remote LLM."""
    mode = _mode or _get_mode()

    if mode == "remote":
        return _remote_completion(system_prompt, user_prompt, max_tokens)
    return _local_completion(system_prompt, user_prompt, max_tokens)


def _local_completion(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    if _local_model is None:
        return "No local LLM loaded. Set LLM_MODE=remote or place GGUF files in models/."
    
    response = _local_model.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return response["choices"][0]["message"]["content"]


def _remote_completion(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    api_url = os.getenv("LLM_API_URL")
    api_key = os.getenv("LLM_API_KEY", "")
    model_name = os.getenv("LLM_MODEL_NAME", "qwen3-4b")

    if not api_url:
        return "LLM_API_URL not set. Configure a remote LLM endpoint."

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }

    try:
        r = requests.post(
            f"{api_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Remote LLM error: {e}"


def is_available() -> bool:
    """Check if LLM is ready (local model loaded or remote configured)."""
    mode = _mode or _get_mode()
    if mode == "remote":
        return bool(os.getenv("LLM_API_URL"))
    return _local_model is not None


def get_model_name() -> str:
    """Return the name of the currently loaded model."""
    if _model_name:
        return _model_name
    mode = _mode or _get_mode()
    if mode == "remote":
        return os.getenv("LLM_MODEL_NAME", "remote")
    return "none"
