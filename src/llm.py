"""Thin wrapper around a local GGUF model (Qwen2.5-7B-Instruct, Q4_K_M) via
llama-cpp-python. One loaded model is reused for answer composition,
relation-typing and classification — only the prompt changes.
"""
import json
import re

from llama_cpp import Llama

_llm = None


def load_model(model_path: str, n_gpu_layers: int = -1, n_ctx: int = 8192) -> Llama:
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,  # -1 = offload all layers to GPU
            n_ctx=n_ctx,
            verbose=False,
        )
    return _llm


def chat(system: str, user: str, max_tokens: int = 1024, temperature: float = 0.1) -> str:
    if _llm is None:
        raise RuntimeError("Call load_model() first")
    result = _llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return result["choices"][0]["message"]["content"]


def chat_json(system: str, user: str, max_tokens: int = 512) -> dict:
    """Ask for JSON and parse it defensively — small local models don't
    always respect 'return only JSON' instructions."""
    raw = chat(system, user, max_tokens=max_tokens, temperature=0.0)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
