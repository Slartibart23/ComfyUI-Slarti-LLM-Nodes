"""ComfyUI-Slarti-LLM-Nodes: local GGUF LLM/VLM nodes via llama-cpp-python,
with model catalog, auto-download and self-installing GPU backend."""

__version__ = "2.0.1"

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from . import model_catalog as _catalog
from . import llama_bootstrap as _bootstrap

print(f"[LocalLLM] ComfyUI-Slarti-LLM-Nodes v{__version__} loaded "
      f"({len(NODE_CLASS_MAPPINGS)} nodes) - {_bootstrap.status_line()}")
try:
    _catalog.print_vram_overview()
except Exception:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
