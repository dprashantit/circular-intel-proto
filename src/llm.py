"""LLM backend wrapper with two interchangeable backends behind the same
chat()/chat_json() interface:

- 'llama_cpp': a local GGUF model via llama-cpp-python. Needs a GPU-linked
  wheel to be fast; used when a Colab GPU is available.
- 'transformers': a HF model loaded directly (no GGUF conversion, no C++
  compile step). Used as the CPU fallback — llama-cpp-python's source build
  is too slow/unreliable on a fresh Python version with no prebuilt wheel.

relations.py and answer.py only call chat()/chat_json() and don't care which
backend is active.
"""
import json
import re

_backend = None
_llm = None  # llama_cpp.Llama instance
_hf_model = None
_hf_tokenizer = None


def load_model(backend: str, model_path: str, n_gpu_layers: int = -1, n_ctx: int = 8192):
    global _backend, _llm, _hf_model, _hf_tokenizer
    _backend = backend

    if backend == "llama_cpp":
        from llama_cpp import Llama

        if _llm is None:
            _llm = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, verbose=False)
        return _llm

    if backend == "transformers":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if _hf_model is None:
            _hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
            _hf_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
        return _hf_model

    raise ValueError(f"Unknown backend: {backend!r} (use 'llama_cpp' or 'transformers')")


def _chat_llama_cpp(system: str, user: str, max_tokens: int, temperature: float) -> str:
    result = _llm.create_chat_completion(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return result["choices"][0]["message"]["content"]


def _chat_transformers(system: str, user: str, max_tokens: int, temperature: float) -> str:
    import torch

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = _hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _hf_tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = _hf_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01),
            pad_token_id=_hf_tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return _hf_tokenizer.decode(new_tokens, skip_special_tokens=True)


def chat(system: str, user: str, max_tokens: int = 1024, temperature: float = 0.1) -> str:
    if _backend is None:
        raise RuntimeError("Call load_model() first")
    if _backend == "llama_cpp":
        return _chat_llama_cpp(system, user, max_tokens, temperature)
    return _chat_transformers(system, user, max_tokens, temperature)


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
