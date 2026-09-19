"""Run by ComfyUI-Manager after 'Install via Git URL' / update.
Installs a GPU-enabled llama-cpp-python into ComfyUI's Python.
Safe to run manually:  <comfyui-python> install.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import llama_bootstrap
    llama_bootstrap.ensure_llama_cpp()
except Exception as e:  # never break the Manager's install flow
    print(f"[LocalLLM] install.py: {e}")
    print("[LocalLLM] The nodes will retry the installation on first use.")
