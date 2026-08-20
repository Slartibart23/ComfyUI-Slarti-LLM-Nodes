"""
ComfyUI-LocalLLM-Nodes

Two nodes built on a shared llama.cpp backend:

  1. LocalLLM Prompt Generator  - text -> text (e.g. SD/Flux prompt writing)
  2. LocalVLM Image Caption     - image(+prompt) -> text (captioning)

Both nodes share one model cache, so switching between them does not
reload the model as long as model/mmproj/ctx settings are identical.
"""

from . import llm_backend as backend


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


class LocalLLMPromptGenerator:
    """Generate or rewrite text-to-image prompts with a local GGUF LLM."""

    CATEGORY = "LocalLLM"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (backend.list_gguf_models(),),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_PROMPT_SYSTEM,
                }),
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": "a cozy cabin in a snowy forest at night",
                }),
                "max_tokens": ("INT", {"default": 2048, "min": 16, "max": 8192,
                                       "tooltip": "Reasoning models spend tokens thinking BEFORE answering. "
                                                  "For long prompts (e.g. video prompts) use 4096."}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 200}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 0.8, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "n_ctx": ("INT", {"default": 8192, "min": 512, "max": 131072, "step": 512}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 999,
                                         "tooltip": "-1 = offload all layers to GPU"}),
                "suppress_thinking": ("BOOLEAN", {"default": True,
                                                  "tooltip": "Appends /no_think to the user message. "
                                                             "Qwen3 models read this switch from the user turn, "
                                                             "not the system prompt. Harmless for other models."}),
                "unload_after_run": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                # Lets you chain text from other nodes (e.g. a caption)
                # into the user prompt.
                "text_input": ("STRING", {"forceInput": True}),
            },
        }

    def generate(self, model, system_prompt, user_prompt, max_tokens,
                 temperature, top_p, top_k, repeat_penalty, seed, n_ctx,
                 n_gpu_layers, suppress_thinking, unload_after_run,
                 text_input=None):
        if model == "none":
            raise RuntimeError(
                "No GGUF model found. Put your model into ComfyUI/models/LLM/."
            )

        prompt = user_prompt
        if text_input:
            prompt = f"{user_prompt}\n\n{text_input}" if user_prompt.strip() else text_input
        if suppress_thinking:
            prompt = f"{prompt} /no_think"

        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = backend.chat_with_recovery(
            model, None, n_ctx, n_gpu_layers, messages,
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
        return {
            "required": {
                "model": (backend.list_gguf_models(),),
                "mmproj": (backend.list_mmproj_files(),),
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_CAPTION_PROMPT,
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a precise image captioning assistant.",
                }),
                "max_tokens": ("INT", {"default": 512, "min": 16, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 200}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 0.8, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "n_ctx": ("INT", {"default": 8192, "min": 2048, "max": 131072, "step": 512,
                                  "tooltip": "Vision tokens are large - keep >= 8192"}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 999}),
                "batch_separator": ("STRING", {"default": "\\n---\\n",
                                               "tooltip": "Separator between captions when batching"}),
                "unload_after_run": ("BOOLEAN", {"default": False}),
            },
        }

    def caption(self, model, mmproj, image, prompt, system_prompt,
                max_tokens, temperature, top_p, top_k, repeat_penalty,
                seed, n_ctx, n_gpu_layers, batch_separator,
                unload_after_run):
        if model == "none":
            raise RuntimeError(
                "No GGUF model found. Put your model into ComfyUI/models/LLM/."
            )
        if mmproj == "none":
            raise RuntimeError(
                "This node needs a vision projector. Select the mmproj-*.gguf "
                "that belongs to your model (same HuggingFace repo) and place "
                "it in ComfyUI/models/LLM/ as well."
            )

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
                    {"type": "text", "text": prompt},
                ],
            })
            text = backend.chat_with_recovery(
                model, mmproj, n_ctx, n_gpu_layers, messages,
                max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                top_k=top_k, repeat_penalty=repeat_penalty, seed=seed)
            captions.append(text)

        if unload_after_run:
            backend.unload_all()

        return (separator.join(captions),)


class LocalLLMUnload:
    """Utility: free the cached LLM from VRAM (e.g. before a big KSampler)."""

    CATEGORY = "LocalLLM"
    FUNCTION = "unload"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
        }

    def unload(self, text):
        backend.unload_all()
        return (text,)


NODE_CLASS_MAPPINGS = {
    "LocalLLMPromptGenerator": LocalLLMPromptGenerator,
    "LocalVLMImageCaption": LocalVLMImageCaption,
    "LocalLLMUnload": LocalLLMUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LocalLLMPromptGenerator": "LocalLLM Prompt Generator (GGUF)",
    "LocalVLMImageCaption": "LocalVLM Image Caption (GGUF + mmproj)",
    "LocalLLMUnload": "LocalLLM Unload (passthrough)",
}
