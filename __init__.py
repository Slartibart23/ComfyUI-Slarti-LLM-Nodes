"""ComfyUI-LocalLLM-Nodes: local GGUF LLM/VLM nodes via llama-cpp-python."""

__version__ = "1.2.0"

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

print(f"[LocalLLM] ComfyUI-LocalLLM-Nodes v{__version__} loaded "
      f"({len(NODE_CLASS_MAPPINGS)} nodes)")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
