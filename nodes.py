"""
ComfyUI-LocalLLM-Nodes

  1. LocalLLM Prompt Generator     - text -> text (SD/Flux/video prompt writing)
  2. LocalVLM Image Caption        - image(+prompt) -> text (captioning)
  3. LocalLLM System Prompt        - pick a template from prompts/ -> STRING
  4. LocalLLM Unload (passthrough) - free VRAM before heavy sampling

Models are chosen from a catalog (models.json / custom_models.json) with
their download size in the label and are fetched from HuggingFace on first
use. Files placed manually in ComfyUI/models/LLM show up as "Local: ...".
"""

import os
import re

from . import llm_backend as backend
from . import model_catalog as catalog

_HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(_HERE, "prompts")

DEFAULT_PROMPT_SYSTEM = (
    "You are an expert prompt engineer for text-to-image diffusion models "
    "(Stable Diffusion, SDXL, Flux). Turn the user's idea into ONE detailed, "
    "comma-separated image prompt. Describe subject, style, lighting, "
    "composition, camera/lens and mood. Output ONLY the prompt itself - "
    "no explanations, no quotes, no markdown."
)

DEFAULT_CAPTION_PROMPT = (
    "Describe this image in detail. Mention the main subject, setting, "
    "style, lighting, colors and composition. Answer with the description "
    "only."
)

# Sampling defaults (tuned by the author for Mistral Small 3.2 / Krea2 use)
_SAMPLING = {
    "max_tokens": ("INT", {"default": 6144, "min": 16, "max": 32768,
                           "tooltip": "Reasoning models spend tokens thinking BEFORE answering. "
                                      "Long video prompts need 4096+."}),
    "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
    "top_p": ("FLOAT", {"default": 0.98, "min": 0.0, "max": 1.0, "step": 0.01}),
    "top_k": ("INT", {"default": 40, "min": 0, "max": 200}),
    "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 0.8, "max": 2.0, "step": 0.01}),
    "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
}

_LOADING = {
    "n_ctx": ("INT", {"default": 8192, "min": 512, "max": 131072, "step": 512,
                      "tooltip": "Context window. Vision tokens are large - keep >= 8192 for captioning."}),
    "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 999,
                             "tooltip": "-1 = offload all layers to GPU. Lower it if you run out of VRAM."}),
}


def _select_system_prompt(widget_text, socket_text):
    """Socket input (from the System Prompt node) wins over the widget."""
    if socket_text is not None and socket_text.strip():
        return socket_text
    return widget_text or ""


def _validate_model_choice(model):
    """Custom validation so that v1.x workflows (bare file names) and
    catalog labels both pass ComfyUI's combo check."""
    if model in (None, "", "none"):
        return "No model selected."
    if catalog.find_entry(model) is not None:
        return True
    rel = model[len(catalog.LOCAL_PREFIX):] if model.startswith(catalog.LOCAL_PREFIX) else model
    if os.path.isfile(catalog.local_path(rel)):
        return True
    return (f"Model '{model}' is not in the catalog and not in "
            f"{catalog.llm_dir()}. Re-select it in the node.")


class LocalLLMPromptGenerator:
    """Generate or rewrite text-to-image / video prompts with a local GGUF LLM."""

    CATEGORY = "LocalLLM"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (catalog.model_choices(), {
                    "tooltip": "Catalog models are downloaded automatically on first use. "
                               "Size in the label = download size; VRAM need is ~10% more plus context."}),
                "system_prompt": ("STRING", {"multiline": True, "default": DEFAULT_PROMPT_SYSTEM,
                                             "tooltip": "Ignored when the system_prompt_input socket is connected."}),
                "user_prompt": ("STRING", {"multiline": True, "default": ""}),
                **_SAMPLING,
                **_LOADING,
                "suppress_thinking": ("BOOLEAN", {"default": True,
                                                  "tooltip": "Appends /no_think to the user message (Qwen3). "
                                                             "Harmless for other models."}),
                "unload_after_run": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "text_input": ("STRING", {"forceInput": True,
                                          "tooltip": "Text from another node (e.g. a caption). Appended to user_prompt."}),
                "system_prompt_input": ("STRING", {"forceInput": True,
                                                   "tooltip": "Connect the 'LocalLLM System Prompt' node here."}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, model):
        return _validate_model_choice(model)

    def generate(self, model, system_prompt, user_prompt, max_tokens,
                 temperature, top_p, top_k, repeat_penalty, seed, n_ctx,
                 n_gpu_layers, suppress_thinking, unload_after_run,
                 text_input=None, system_prompt_input=None):
        model_path, entry = catalog.resolve_model(model)
        if entry:
            warn = catalog.vram_warning(entry["size_gb"], n_ctx)
            if warn:
                print(warn)

        system_prompt = _select_system_prompt(system_prompt, system_prompt_input)

        prompt = user_prompt or ""
        if text_input:
            prompt = f"{prompt}\n\n{text_input}" if prompt.strip() else text_input
        if not prompt.strip():
            raise RuntimeError("user_prompt is empty and nothing is connected to text_input.")
        if suppress_thinking:
            prompt = f"{prompt} /no_think"

        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = backend.chat_with_recovery(
            model_path, None, n_ctx, n_gpu_layers, messages,
            max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            top_k=top_k, repeat_penalty=repeat_penalty, seed=seed)

        if unload_after_run:
            backend.unload_all()
        return (text,)


class LocalVLMImageCaption:
    """Caption images with a local multimodal GGUF model (model + mmproj)."""

    CATEGORY = "LocalLLM"
    FUNCTION = "caption"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)

    @classmethod
    def INPUT_TYPES(cls):
        sampling = dict(_SAMPLING)
        sampling["max_tokens"] = ("INT", {"default": 2048, "min": 16, "max": 16384})
        sampling["temperature"] = ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05,
                                             "tooltip": "Captioning benefits from a low temperature."})
        return {
            "required": {
                "model": (catalog.model_choices(vision_only=True), {
                    "tooltip": "Only vision-capable catalog families are listed (plus local files)."}),
                "mmproj": (catalog.mmproj_choices(), {
                    "tooltip": "'auto' downloads/uses the projector that belongs to the catalog model. "
                               "For local models pick the matching mmproj-*.gguf."}),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": DEFAULT_CAPTION_PROMPT}),
                "system_prompt": ("STRING", {"multiline": True,
                                             "default": "You are a precise image captioning assistant.",
                                             "tooltip": "Ignored when the system_prompt_input socket is connected."}),
                **sampling,
                **_LOADING,
                "batch_separator": ("STRING", {"default": "\\n---\\n",
                                               "tooltip": "Separator between captions when batching"}),
                "unload_after_run": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "system_prompt_input": ("STRING", {"forceInput": True,
                                                   "tooltip": "Connect the 'LocalLLM System Prompt' node here."}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, model):
        return _validate_model_choice(model)

    def caption(self, model, mmproj, image, prompt, system_prompt,
                max_tokens, temperature, top_p, top_k, repeat_penalty,
                seed, n_ctx, n_gpu_layers, batch_separator,
                unload_after_run, system_prompt_input=None):
        model_path, entry = catalog.resolve_model(model)
        mmproj_path = catalog.resolve_mmproj(mmproj, entry)
        if entry:
            warn = catalog.vram_warning(entry["size_gb"] + 1.0, n_ctx)
            if warn:
                print(warn)

        system_prompt = _select_system_prompt(system_prompt, system_prompt_input)
        uris = backend.comfy_image_to_data_uris(image)
        separator = batch_separator.replace("\\n", "\n")

        captions = []
        for uri in uris:
            messages = []
            if system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt})
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": uri}},
                    {"type": "text", "text": prompt or DEFAULT_CAPTION_PROMPT},
                ],
            })
            text = backend.chat_with_recovery(
                model_path, mmproj_path, n_ctx, n_gpu_layers, messages,
                max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                top_k=top_k, repeat_penalty=repeat_penalty, seed=seed)
            captions.append(text)

        if unload_after_run:
            backend.unload_all()
        return (separator.join(captions),)


# ---------------------------------------------------------------------------
# Prompt library
# ---------------------------------------------------------------------------

def _load_prompt_templates():
    """Return {display name: prompt text} from prompts/*.md.

    Each .md file has a '# Title' line and the actual system prompt inside
    the first ``` fenced block. Files without a fenced block are used as-is.
    """
    templates = {}
    if not os.path.isdir(PROMPTS_DIR):
        return templates
    for fn in sorted(os.listdir(PROMPTS_DIR)):
        if not fn.lower().endswith(".md") or fn.lower() == "readme.md":
            continue
        try:
            with open(os.path.join(PROMPTS_DIR, fn), "r", encoding="utf-8") as fh:
                raw = fh.read()
        except Exception:
            continue
        m = re.search(r"^#\s+(.+?)\s*$", raw, flags=re.MULTILINE)
        title = m.group(1).strip() if m else os.path.splitext(fn)[0]
        blocks = re.findall(r"```[^\n]*\n(.*?)```", raw, flags=re.DOTALL)
        body = blocks[0].strip() if blocks else raw.strip()
        name = f"{title}  [{os.path.splitext(fn)[0]}]"
        templates[name] = body
    return templates


class LocalLLMSystemPrompt:
    """Pick a ready-made system prompt from the prompts/ library."""

    CATEGORY = "LocalLLM"
    FUNCTION = "get"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)

    @classmethod
    def INPUT_TYPES(cls):
        names = list(_load_prompt_templates().keys()) or ["(no templates found in prompts/)"]
        return {
            "required": {
                "template": (names, {"tooltip": "Templates are the .md files in the prompts/ folder. "
                                                "Add your own there and press 'R'."}),
                "append": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "Optional extra instructions appended to the template."}),
            },
        }

    def get(self, template, append=""):
        text = _load_prompt_templates().get(template)
        if text is None:
            raise RuntimeError(f"Template '{template}' not found in {PROMPTS_DIR}. Press 'R' to refresh.")
        if append and append.strip():
            text = f"{text}\n\n{append.strip()}"
        return (text,)


class LocalLLMUnload:
    """Utility: free the cached LLM from VRAM (e.g. before a big KSampler)."""

    CATEGORY = "LocalLLM"
    FUNCTION = "unload"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"forceInput": True})}}

    def unload(self, text):
        backend.unload_all()
        return (text,)


NODE_CLASS_MAPPINGS = {
    "LocalLLMPromptGenerator": LocalLLMPromptGenerator,
    "LocalVLMImageCaption": LocalVLMImageCaption,
    "LocalLLMSystemPrompt": LocalLLMSystemPrompt,
    "LocalLLMUnload": LocalLLMUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LocalLLMPromptGenerator": "LocalLLM Prompt Generator (GGUF)",
    "LocalVLMImageCaption": "LocalVLM Image Caption (GGUF + mmproj)",
    "LocalLLMSystemPrompt": "LocalLLM System Prompt (Library)",
    "LocalLLMUnload": "LocalLLM Unload (passthrough)",
}
