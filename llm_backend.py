"""
Shared llama-cpp-python backend for ComfyUI-Slarti-LLM-Nodes.

Handles:
  - Loading / caching of Llama instances (text-only and vision)
  - Resolution of the correct multimodal chat handler for the
    installed llama-cpp-python version
  - Conversion of ComfyUI IMAGE tensors to base64 data URIs
  - Chat completion with context guard, think-block stripping and
    automatic recovery from KV-cache exhaustion

Model discovery, download and the models/LLM folder live in model_catalog.
"""

import base64
import gc
import io
import os

import numpy as np

from . import model_catalog
from . import llama_bootstrap

model_catalog.register_llm_folder()


# ---------------------------------------------------------------------------
# Chat handler resolution (vision support)
# ---------------------------------------------------------------------------

# Ordered by preference: newest architectures first. Which of these exist
# depends on the installed llama-cpp-python version/fork.
_HANDLER_CANDIDATES = [
    "Qwen3VLChatHandler",
    "Qwen25VLChatHandler",
    "MiniCPMv26ChatHandler",
    "Llava16ChatHandler",
    "Llava15ChatHandler",
]

# Which handler to try FIRST for a given model family (matched against the
# lower-cased model file name). Mistral Small 3.x uses a Pixtral projector;
# recent forks expose it through the generic Llava-style handlers.
_HANDLER_HINTS = [
    ("qwen3", ["Qwen3VLChatHandler"]),
    ("qwen2.5", ["Qwen25VLChatHandler"]),
    ("qwen2_5", ["Qwen25VLChatHandler"]),
    ("minicpm", ["MiniCPMv26ChatHandler"]),
    ("mistral", ["Llava16ChatHandler", "Llava15ChatHandler"]),
    ("pixtral", ["Llava16ChatHandler", "Llava15ChatHandler"]),
]


def _resolve_chat_handler(mmproj_path, model_path="", verbose=False):
    """Find a multimodal chat handler class supported by the installed
    llama-cpp-python and instantiate it with the given mmproj (clip) model."""
    from llama_cpp import llama_chat_format

    candidates = list(_HANDLER_CANDIDATES)
    name_l = os.path.basename(model_path).lower()
    for needle, preferred in _HANDLER_HINTS:
        if needle in name_l:
            candidates = preferred + [c for c in candidates if c not in preferred]
            break

    last_error = None
    for name in candidates:
        handler_cls = getattr(llama_chat_format, name, None)
        if handler_cls is None:
            continue
        try:
            handler = handler_cls(clip_model_path=mmproj_path, verbose=verbose)
            print(f"[LocalLLM] Using multimodal chat handler: {name}")
            return handler
        except Exception as e:  # handler exists but failed to init
            last_error = e
            continue

    raise RuntimeError(
        "No compatible multimodal chat handler found in your llama-cpp-python "
        "installation. Vision needs a recent build (JamePeng fork). Delete the "
        "cached wheel in ComfyUI/models/LLM/.wheels, restart ComfyUI and run "
        "the node again to fetch a newer build - or install one manually "
        "(see README, Troubleshooting).\n"
        f"Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Llama instance cache
# ---------------------------------------------------------------------------

_CACHE = {}  # key -> Llama instance


def _cache_key(model_path, mmproj_path, n_ctx, n_gpu_layers):
    return (os.path.abspath(model_path),
            os.path.abspath(mmproj_path) if mmproj_path else None,
            int(n_ctx), int(n_gpu_layers))


def get_llama(model_path, mmproj_path=None, n_ctx=8192,
              n_gpu_layers=-1, verbose=False):
    """Load (or fetch from cache) a Llama instance from absolute paths.

    mmproj_path: None for text-only, otherwise the vision projector .gguf
    belonging to the same model family.
    """
    llama_bootstrap.ensure_llama_cpp()
    from llama_cpp import Llama

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    key = _cache_key(model_path, mmproj_path, n_ctx, n_gpu_layers)
    if key in _CACHE:
        return _CACHE[key]

    # A 27B model doesn't fit twice in VRAM -> evict everything else first.
    unload_all()

    chat_handler = None
    if mmproj_path is not None:
        chat_handler = _resolve_chat_handler(mmproj_path, model_path, verbose=verbose)

    size_gb = os.path.getsize(model_path) / 1e9
    print(f"[LocalLLM] Loading {os.path.basename(model_path)} "
          f"({size_gb:.2f} GB, n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers}, "
          f"vision={'yes' if chat_handler else 'no'}) ...")

    try:
        # verbose=True during load so llama.cpp prints the REAL reason
        # (unknown architecture, bad magic, truncated file, OOM, ...)
        # to the ComfyUI console instead of a generic ValueError.
        llm = Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_ctx=int(n_ctx),
            n_gpu_layers=int(n_gpu_layers),
            logits_all=False,
            verbose=True,
        )
    except ValueError as e:
        raise RuntimeError(
            f"llama.cpp could not load '{os.path.basename(model_path)}' "
            f"({size_gb:.2f} GB). Check the ComfyUI console above this "
            "message for the actual llama.cpp error. Most common causes:\n"
            "  1. Incomplete download - re-verify the file size against "
            "the HuggingFace repo.\n"
            "  2. Your llama-cpp-python's native library is too old for "
            "this model architecture. Delete ComfyUI/models/LLM/.wheels, "
            "restart ComfyUI and run again to auto-install a newer build "
            "(or see README, Troubleshooting).\n"
            "  3. Not enough free (V)RAM - lower n_gpu_layers or close "
            "other models first."
        ) from e
    _CACHE[key] = llm
    return llm


def unload_all():
    """Free every cached Llama instance and reclaim VRAM."""
    global _CACHE
    if not _CACHE:
        return
    for llm in _CACHE.values():
        try:
            llm.close()
        except Exception:
            pass
    _CACHE = {}
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    print("[LocalLLM] All models unloaded.")


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def comfy_image_to_data_uris(image_tensor, max_side=1568):
    """Convert a ComfyUI IMAGE tensor [B, H, W, C] (float 0..1) into a list
    of base64 PNG data URIs, one per batch item. Large images are downscaled
    to keep vision token counts sane."""
    from PIL import Image

    uris = []
    arr = image_tensor.cpu().numpy() if hasattr(image_tensor, "cpu") else np.asarray(image_tensor)
    if arr.ndim == 3:  # single image without batch dim
        arr = arr[None, ...]

    for i in range(arr.shape[0]):
        img = np.clip(arr[i] * 255.0, 0, 255).astype(np.uint8)
        pil = Image.fromarray(img)
        if max(pil.size) > max_side:
            scale = max_side / max(pil.size)
            new_size = (max(1, int(pil.width * scale)),
                        max(1, int(pil.height * scale)))
            pil = pil.resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        uris.append(f"data:image/png;base64,{b64}")
    return uris


# ---------------------------------------------------------------------------
# Chat completion helper
# ---------------------------------------------------------------------------

def _clear_memory(llm):
    """Best-effort reset of the model's KV/recurrent state between runs.
    Hybrid (SSM+attention) models can leak cache slots across generations,
    eventually causing 'failed to find a memory slot' errors."""
    try:
        llm.reset()
    except Exception:
        pass
    ctx = getattr(llm, "_ctx", None)
    if ctx is not None:
        for name in ("memory_clear", "kv_cache_clear", "kv_self_clear"):
            fn = getattr(ctx, name, None)
            if fn is not None:
                try:
                    fn()
                    break
                except Exception:
                    continue


def chat(llm, messages, max_tokens=512, temperature=0.7, top_p=0.9,
         top_k=40, repeat_penalty=1.1, seed=-1):
    """Run a chat completion and return the assistant text."""
    _clear_memory(llm)

    # CONTEXT BUDGET GUARD: prompt tokens + generated tokens must fit in
    # n_ctx. If max_tokens alone (nearly) fills the context, generation
    # slams into the context wall mid-answer and surfaces as a cryptic
    # 'KV slots full' error. Clamp and warn instead. We reserve a rough
    # prompt allowance; llama.cpp stops earlier anyway if the prompt is
    # larger than that.
    max_tokens = int(max_tokens)
    try:
        n_ctx = int(llm.n_ctx())
    except Exception:
        n_ctx = 0
    if n_ctx > 0 and max_tokens > n_ctx - 1024:
        clamped = max(256, n_ctx - 1024)
        print(f"[LocalLLM] WARNING: max_tokens={max_tokens} does not leave "
              f"room for the prompt inside n_ctx={n_ctx}. Clamping "
              f"max_tokens to {clamped}. For long generations raise n_ctx "
              f"instead (e.g. n_ctx=8192 with max_tokens=4096).")
        max_tokens = clamped

    kwargs = dict(
        messages=messages,
        max_tokens=max_tokens,
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        repeat_penalty=float(repeat_penalty),
    )
    if seed is not None and int(seed) >= 0:
        kwargs["seed"] = int(seed)

    # If this llama-cpp-python build supports disabling reasoning at the
    # API level (e.g. JamePeng fork's reasoning budget control), use it.
    import inspect
    try:
        params = inspect.signature(llm.create_chat_completion).parameters
        if "reasoning_budget" in params:
            kwargs["reasoning_budget"] = 0
        elif "enable_thinking" in params:
            kwargs["enable_thinking"] = False
    except (ValueError, TypeError):
        pass

    result = llm.create_chat_completion(**kwargs)
    text = result["choices"][0]["message"]["content"] or ""
    return extract_final_answer(text).strip()


def chat_with_recovery(model_path, mmproj_path, n_ctx, n_gpu_layers,
                       messages, **sampling):
    """chat() with automatic recovery: if the hybrid KV cache is exhausted
    ('Failed completely even with batch size 1'), unload, reload the model
    fresh and retry ONCE. Fixes intermittent cache-fragmentation failures
    on hybrid SSM models at the cost of one reload."""
    llm = get_llama(model_path, mmproj_path=mmproj_path,
                    n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
    try:
        return chat(llm, messages, **sampling)
    except RuntimeError as e:
        msg = str(e)
        if "batch size 1" not in msg and "memory slot" not in msg \
                and "KV slots" not in msg:
            raise
        print("[LocalLLM] KV cache exhausted - reloading model and "
              "retrying once...")
        unload_all()
        llm = get_llama(model_path, mmproj_path=mmproj_path,
                        n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
        return chat(llm, messages, **sampling)


def extract_final_answer(text):
    """Isolate the model's actual answer from any reasoning noise.

    Priority 1: closed <prompt>...</prompt> -> content of the LAST block.
    Priority 2: an OPEN <prompt> with no closing tag (max_tokens ran out
    mid-answer) -> everything after the last <prompt>, with any think
    block ahead of it removed first. This recovers a near-complete prompt
    instead of returning raw reasoning.
    Priority 3: no <prompt> markers at all -> fall back to stripping
    Qwen3 <think> blocks (closed, lone-closer, or unclosed)."""
    import re
    # First, remove any reasoning that sits BEFORE the answer, so a stray
    # </think> inside reasoning can't confuse the prompt extraction.
    body = text
    if "</think>" in body:
        body = body.rsplit("</think>", 1)[1]

    closed = re.findall(r"<prompt>(.*?)</prompt>", body, flags=re.DOTALL)
    if closed:
        return closed[-1]
    if "<prompt>" in body:
        # open tag, never closed -> take everything after the last one
        return body.rsplit("<prompt>", 1)[1]

    # No markers: strip any remaining unclosed think opener and return.
    body = re.sub(r"<think>.*$", "", body, flags=re.DOTALL)
    return body


def strip_think_blocks(text):
    """Backward-compatible alias, see extract_final_answer."""
    return extract_final_answer(text)
