"""
Self-contained installer for a GPU-enabled llama-cpp-python.

Called lazily (first node execution) and by install.py (ComfyUI-Manager).

Strategy, in order:
  0. Already importable with GPU offload  -> nothing to do.
  1. Wheel cached in models/LLM/.wheels   -> pip install (offline, seconds).
  2. Prebuilt CUDA wheel from the JamePeng fork releases on GitHub
     (matched by OS / Python / CUDA)      -> download, cache, install.
  3. abetlen's prebuilt CUDA wheel index  -> pip install --extra-index-url.
  4. Source build with CMAKE_ARGS=-DGGML_CUDA=on (needs CUDA toolkit).

Everything is installed into the SAME interpreter that runs ComfyUI
(sys.executable), which on portable Windows installs is python_embeded.

Set LOCALLLM_NO_AUTOINSTALL=1 to disable all of this.
"""

import glob
import importlib
import json
import os
import platform
import re
import subprocess
import sys
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
FORK_REPO = "JamePeng/llama-cpp-python"
ABETLEN_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/{cu}"
_STATE = {"checked": False, "ok": False, "restart_needed": False}


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

def py_tag():
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def platform_tag():
    sysname = platform.system().lower()
    if sysname == "windows":
        return "win_amd64", "win"
    if sysname == "linux":
        return "linux_x86_64", "linux"
    if sysname == "darwin":
        return "macosx", "macos"
    return sysname, sysname


def cuda_tag():
    """'cu128' style tag from torch, or None if no CUDA."""
    try:
        import torch
        if not torch.cuda.is_available() or not torch.version.cuda:
            return None
        major, minor = torch.version.cuda.split(".")[:2]
        return f"cu{major}{minor}"
    except Exception:
        return None


def wheel_cache_dir():
    try:
        from . import model_catalog
        base = model_catalog.llm_dir()
    except Exception:
        base = os.path.join(_HERE, "models", "LLM")
    d = os.path.join(base, ".wheels")
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Status check
# ---------------------------------------------------------------------------

_BACKENDS_LOADED = False


def _copy_cuda_runtime_from_torch(lib_dir):
    """Windows CUDA wheels of llama-cpp-python may ship ggml-cuda.dll without
    the CUDA runtime DLLs it depends on (cudart, cublas, cublasLt). If they
    cannot be found, ggml silently falls back to CPU. PyTorch CUDA builds
    bundle exactly these DLLs in torch/lib, so copy missing or outdated ones
    next to ggml-cuda.dll. Copying new files works even while ComfyUI runs."""
    if not os.path.isfile(os.path.join(lib_dir, "ggml-cuda.dll")):
        return
    try:
        import importlib.util
        spec = importlib.util.find_spec("torch")
        if not spec or not spec.origin:
            return
        torch_lib = os.path.join(os.path.dirname(spec.origin), "lib")
    except Exception:
        return
    import shutil
    for pattern in ("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll"):
        for src in glob.glob(os.path.join(torch_lib, pattern)):
            dst = os.path.join(lib_dir, os.path.basename(src))
            try:
                if os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(src):
                    continue
                shutil.copy2(src, dst)
                print(f"[LocalLLM] Copied CUDA runtime {os.path.basename(src)} from torch/lib")
            except Exception as e:
                print(f"[LocalLLM] Could not copy {os.path.basename(src)}: {e}")


def _load_ggml_backends(llama_cpp):
    """PATCH: Newer llama.cpp builds load their compute backends (CPU/CUDA)
    lazily as separate DLLs. Right after import no device is registered yet,
    so llama_supports_gpu_offload() wrongly reports False. Load the backends
    from llama_cpp/lib once, but only if none are registered yet."""
    global _BACKENDS_LOADED
    if _BACKENDS_LOADED:
        return
    _BACKENDS_LOADED = True
    try:
        import ctypes
        lib_dir = os.path.join(os.path.dirname(llama_cpp.__file__), "lib")
        name = {"windows": "ggml.dll", "darwin": "libggml.dylib"}.get(
            platform.system().lower(), "libggml.so")
        ggml_path = os.path.join(lib_dir, name)
        if not os.path.isfile(ggml_path):
            return
        if platform.system().lower() == "windows":
            _copy_cuda_runtime_from_torch(lib_dir)
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(lib_dir)
        g = ctypes.CDLL(ggml_path)
        g.ggml_backend_dev_count.restype = ctypes.c_size_t
        if g.ggml_backend_dev_count() == 0:
            g.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
            g.ggml_backend_load_all_from_path(lib_dir.encode())
    except Exception as e:
        print(f"[LocalLLM] Backend preload skipped: {e}")


def _gpu_build_present():
    """(importable, gpu_offload) for the currently installed llama_cpp."""
    try:
        import llama_cpp
    except Exception:
        return False, False
    _load_ggml_backends(llama_cpp)
    try:
        gpu = bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:
        gpu = False
    return True, gpu


def _pip(args, env=None):
    cmd = [sys.executable, "-m", "pip", "install", "--no-input",
           "--disable-pip-version-check"] + args
    print("[LocalLLM] " + " ".join(cmd))
    return subprocess.call(cmd, env=env)


# ---------------------------------------------------------------------------
# Wheel discovery on GitHub
# ---------------------------------------------------------------------------

def _github_json(url):
    req = urllib.request.Request(url, headers={
        "User-Agent": "ComfyUI-Slarti-LLM-Nodes",
        "Accept": "application/vnd.github+json"})
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _cuda_candidates(cu):
    """Preferred CUDA tags: exact first, then lower minor versions of the
    same major (a newer driver runs older CUDA runtimes fine)."""
    m = re.match(r"cu(\d+?)(\d)$", cu)  # cu128 -> ('12','8'), cu130 -> ('13','0')
    if not m:
        return [cu]
    major, minor = m.group(1), int(m.group(2))
    return [f"cu{major}{i}" for i in range(minor, -1, -1)]


def find_fork_wheel(cu):
    """Return (download_url, asset_name) of the best matching wheel in the
    JamePeng releases, or None."""
    pyt = py_tag()
    plat_asset, plat_tag = platform_tag()
    releases = _github_json(
        f"https://api.github.com/repos/{FORK_REPO}/releases?per_page=60")
    cu_pref = _cuda_candidates(cu)

    best = None
    for rel in releases:
        tag = (rel.get("tag_name") or "").lower()
        if plat_tag not in tag:
            continue
        cu_rank = next((i for i, c in enumerate(cu_pref) if c in tag), None)
        if cu_rank is None:
            continue
        # Prefer generic AVX2 builds over exotic ones if several exist.
        flavor_rank = 0 if "avx2" in tag else 1
        for asset in rel.get("assets", []):
            name = asset.get("name", "")
            if not name.endswith(".whl"):
                continue
            if pyt not in name or plat_asset not in name:
                continue
            published = rel.get("published_at", "")
            key = (cu_rank, flavor_rank, -_ts(published))
            if best is None or key < best[0]:
                best = (key, asset["browser_download_url"], name)
    if best:
        return best[1], best[2]
    return None


def _ts(iso):
    try:
        import datetime
        return datetime.datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").timestamp()
    except Exception:
        return 0.0


def _download(url, dest):
    print(f"[LocalLLM] Downloading wheel {os.path.basename(dest)} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-Slarti-LLM-Nodes"})
    tmp = dest + ".part"
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as fh:
        total = r.headers.get("Content-Length")
        total = int(total) if total else 0
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total:
                sys.stdout.write(f"\r[LocalLLM]   {done / total * 100:5.1f}%  "
                                 f"{done / 1e6:.0f} / {total / 1e6:.0f} MB   ")
                sys.stdout.flush()
    sys.stdout.write("\n")
    os.replace(tmp, dest)
    return dest


# ---------------------------------------------------------------------------
# Install strategies
# ---------------------------------------------------------------------------

def _install_cached_wheel(cu):
    cache = wheel_cache_dir()
    pyt = py_tag()
    plat_asset, _ = platform_tag()
    for whl in sorted(glob.glob(os.path.join(cache, "*.whl")), reverse=True):
        n = os.path.basename(whl)
        if pyt in n and plat_asset in n:
            print(f"[LocalLLM] Installing cached wheel {n}")
            if _pip(["--force-reinstall", "--no-deps", whl]) == 0:
                _pip(["numpy", "typing-extensions", "diskcache", "jinja2"])
                return True
    return False


def _install_fork_wheel(cu):
    try:
        found = find_fork_wheel(cu)
    except Exception as e:
        print(f"[LocalLLM] GitHub release lookup failed: {e}")
        return False
    if not found:
        print(f"[LocalLLM] No prebuilt wheel for {py_tag()}/{platform_tag()[0]}/{cu} "
              f"in {FORK_REPO} releases.")
        return False
    url, name = found
    dest = os.path.join(wheel_cache_dir(), name)
    if not os.path.isfile(dest):
        _download(url, dest)
    if _pip(["--force-reinstall", "--no-deps", dest]) != 0:
        return False
    _pip(["numpy", "typing-extensions", "diskcache", "jinja2"])
    return True


def _install_abetlen_index(cu):
    for c in _cuda_candidates(cu):
        if _pip(["--upgrade", "llama-cpp-python",
                 "--extra-index-url", ABETLEN_INDEX.format(cu=c)]) == 0:
            return True
    return False


def _install_from_source():
    env = dict(os.environ)
    env["CMAKE_ARGS"] = "-DGGML_CUDA=on"
    env["FORCE_CMAKE"] = "1"
    return _pip(["--no-cache-dir", "--force-reinstall",
                 f"git+https://github.com/{FORK_REPO}.git"], env=env) == 0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ensure_llama_cpp(require_gpu=True, force=False):
    """Make sure a usable llama_cpp is importable. Returns True when it is
    usable in THIS process. Raises RuntimeError with instructions when the
    install succeeded but a ComfyUI restart is required, or when every
    strategy failed."""
    if _STATE["checked"] and _STATE["ok"] and not force:
        return True
    if _STATE["restart_needed"]:
        raise RuntimeError(
            "llama-cpp-python was (re)installed. Please restart ComfyUI once, "
            "then run the workflow again.")

    importable, gpu = _gpu_build_present()
    cu = cuda_tag()
    want_gpu = require_gpu and cu is not None

    if importable and (gpu or not want_gpu):
        _STATE.update(checked=True, ok=True)
        return True

    if os.environ.get("LOCALLLM_NO_AUTOINSTALL"):
        raise RuntimeError(
            "llama-cpp-python is missing or has no GPU support, and auto-install "
            "is disabled (LOCALLLM_NO_AUTOINSTALL). Install it manually - see README.")

    was_imported = "llama_cpp" in sys.modules
    print("[LocalLLM] ---------------------------------------------------------")
    print(f"[LocalLLM] llama-cpp-python {'not installed' if not importable else 'has no GPU support'}."
          f" Installing automatically for Python {sys.version_info.major}."
          f"{sys.version_info.minor} / {platform_tag()[0]} / {cu or 'CPU'} ...")
    print(f"[LocalLLM] Target interpreter: {sys.executable}")
    print("[LocalLLM] ---------------------------------------------------------")

    ok = False
    if want_gpu:
        ok = (_install_cached_wheel(cu)
              or _install_fork_wheel(cu)
              or _install_abetlen_index(cu)
              or _install_from_source())
    else:
        ok = _pip(["--upgrade", "llama-cpp-python"]) == 0 or _install_from_source()

    if not ok:
        raise RuntimeError(
            "Automatic installation of llama-cpp-python failed. See the console "
            "above for pip output. Manual fallback: download a wheel for your "
            f"OS/Python/CUDA from https://github.com/{FORK_REPO}/releases and run\n"
            f"  \"{sys.executable}\" -m pip install --force-reinstall <wheel>.whl")

    if was_imported:
        # A native library that was already loaded cannot be swapped in-process.
        _STATE.update(checked=True, ok=False, restart_needed=True)
        raise RuntimeError(
            "llama-cpp-python was installed successfully. Please restart ComfyUI "
            "once so the new GPU build is loaded, then run the workflow again.")

    importlib.invalidate_caches()
    importable, gpu = _gpu_build_present()
    if importable and (gpu or not want_gpu):
        print("[LocalLLM] llama-cpp-python is ready"
              f"{' (GPU offload available)' if gpu else ''}.")
        _STATE.update(checked=True, ok=True)
        return True

    _STATE.update(checked=True, ok=False, restart_needed=True)
    raise RuntimeError(
        "llama-cpp-python was installed, but could not be loaded in the running "
        "process. Please restart ComfyUI once and run the workflow again.")


def status_line():
    importable, gpu = _gpu_build_present()
    if not importable:
        return "llama-cpp-python: not installed (will be installed on first run)"
    try:
        import llama_cpp
        ver = getattr(llama_cpp, "__version__", "?")
    except Exception:
        ver = "?"
    return f"llama-cpp-python {ver}: {'GPU' if gpu else 'CPU-only'}"


if __name__ == "__main__":  # manual invocation: python llama_bootstrap.py
    ensure_llama_cpp()
