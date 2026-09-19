"""
Model catalog and auto-download for ComfyUI-LocalLLM-Nodes.

- Reads models.json (built-in) and custom_models.json (user) next to this file.
- Builds the dropdown entries shown in the nodes. Every entry carries its
  download size so the user can pick a quant that fits the GPU.
- Downloads the selected GGUF (and its mmproj, if any) from HuggingFace on
  first use, with resume support and a console progress bar.
- Scans ComfyUI/models/LLM for files placed there manually.

Dropdown labels are deliberately STABLE (no "downloaded" markers, no VRAM
markers): ComfyUI stores the selected label in the workflow JSON and refuses
values that are no longer in the list, so anything that changes over time
must not be part of the label.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

try:
    import folder_paths
except ImportError:  # outside ComfyUI (tests)
    folder_paths = None

_HERE = os.path.dirname(os.path.abspath(__file__))
LLM_FOLDER_NAME = "LLM"
LOCAL_PREFIX = "Local: "
SEP = " \u00b7 "  # " · "


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def llm_dir():
    """ComfyUI/models/LLM (created on demand). Outside ComfyUI: ./models/LLM."""
    if folder_paths is not None:
        base = os.path.join(folder_paths.models_dir, LLM_FOLDER_NAME)
    else:
        base = os.path.join(_HERE, "models", LLM_FOLDER_NAME)
    os.makedirs(base, exist_ok=True)
    return base


def register_llm_folder():
    """Make ComfyUI aware of models/LLM (idempotent)."""
    if folder_paths is None:
        return
    path = llm_dir()
    try:
        folder_paths.add_model_folder_path(LLM_FOLDER_NAME, path)
    except Exception:
        if LLM_FOLDER_NAME not in folder_paths.folder_names_and_paths:
            folder_paths.folder_names_and_paths[LLM_FOLDER_NAME] = ([path], {".gguf"})


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------

_CATALOG_CACHE = None


def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None
    except Exception as e:  # malformed user file -> don't kill node import
        print(f"[LocalLLM] WARNING: could not read {os.path.basename(path)}: {e}")
        return None


def load_catalog(force=False):
    """Return a list of catalog entries (dicts), built-in first, then custom.

    Entry fields:
      label       stable dropdown label
      family      family label
      quant       quant name
      size_gb     download size of the main file
      repo        HF repo id
      filename    file name inside repo (== local file name)
      vision      bool
      mmproj      dict(repo, filename, size_gb) or None
      recommended bool
    """
    global _CATALOG_CACHE
    if _CATALOG_CACHE is not None and not force:
        return _CATALOG_CACHE

    entries = []
    for src in ("models.json", "custom_models.json"):
        data = _load_json(os.path.join(_HERE, src))
        if not data:
            continue
        for fam in data.get("families", []):
            fam_label = fam.get("label") or fam.get("id") or "model"
            mmproj = fam.get("mmproj")
            for q in fam.get("quants", []):
                try:
                    size = float(q["size_gb"])
                except (KeyError, TypeError, ValueError):
                    size = 0.0
                label = f"{fam_label}{SEP}{q.get('quant', '?')}{SEP}{size:g} GB"
                entries.append({
                    "label": label,
                    "family": fam_label,
                    "quant": q.get("quant", "?"),
                    "size_gb": size,
                    "repo": q["repo"],
                    "filename": q["filename"],
                    "vision": bool(fam.get("vision", False)),
                    "mmproj": mmproj,
                    "recommended": bool(q.get("recommended", False)),
                    "source": src,
                })
    _CATALOG_CACHE = entries
    return entries


# ---------------------------------------------------------------------------
# Local files
# ---------------------------------------------------------------------------

def _catalog_filenames():
    names = set()
    for e in load_catalog():
        names.add(e["filename"])
        if e.get("mmproj"):
            names.add(e["mmproj"]["filename"])
    return names


def list_local_gguf():
    """All .gguf files in models/LLM (recursively), relative paths."""
    base = llm_dir()
    out = []
    for root, _dirs, files in os.walk(base):
        for f in files:
            if f.lower().endswith(".gguf"):
                rel = os.path.relpath(os.path.join(root, f), base)
                out.append(rel.replace("\\", "/"))
    return sorted(out, key=str.lower)


def list_local_models():
    """Local main-model files that are NOT part of the catalog."""
    known = _catalog_filenames()
    return [f for f in list_local_gguf()
            if "mmproj" not in os.path.basename(f).lower()
            and os.path.basename(f) not in known]


def list_local_mmproj():
    return [f for f in list_local_gguf() if "mmproj" in os.path.basename(f).lower()]


# ---------------------------------------------------------------------------
# Dropdown construction / resolution
# ---------------------------------------------------------------------------

def model_choices(vision_only=False):
    """Dropdown entries: catalog first (optionally vision families only),
    then any extra local files prefixed with 'Local: '."""
    choices = [e["label"] for e in load_catalog() if (e["vision"] or not vision_only)]
    choices += [LOCAL_PREFIX + f for f in list_local_models()]
    return choices or ["none"]


def mmproj_choices():
    return ["auto"] + list_local_mmproj()


def find_entry(label):
    for e in load_catalog():
        if e["label"] == label:
            return e
    return None


def local_path(filename):
    return os.path.join(llm_dir(), filename)


def is_downloaded(entry):
    return os.path.isfile(local_path(entry["filename"]))


def resolve_model(label):
    """Turn a dropdown label into (absolute path, entry-or-None).
    Downloads catalog entries that are missing locally."""
    if label in (None, "", "none"):
        raise RuntimeError("No model selected. Pick one in the 'model' dropdown.")

    if label.startswith(LOCAL_PREFIX):
        rel = label[len(LOCAL_PREFIX):]
        path = local_path(rel)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Local model '{rel}' not found in {llm_dir()}. "
                "Press 'R' in ComfyUI to refresh the list.")
        return path, None

    entry = find_entry(label)
    if entry is None:
        # Backwards compatibility: v1.x workflows stored bare file names.
        path = local_path(label)
        if os.path.isfile(path):
            return path, None
        raise RuntimeError(
            f"Unknown model '{label}'. It is neither in the catalog nor in "
            f"{llm_dir()}. Re-select the model in the node.")

    path = local_path(entry["filename"])
    if not os.path.isfile(path):
        print(f"[LocalLLM] Model not present locally -> downloading "
              f"{entry['family']} {entry['quant']} ({entry['size_gb']:g} GB)")
        download_file(entry["repo"], entry["filename"], path)
    return path, entry


def resolve_mmproj(mmproj_label, entry):
    """Return absolute path of the mmproj to use, or raise with a clear
    message. 'auto' takes the mmproj that belongs to the catalog entry."""
    if mmproj_label and mmproj_label != "auto":
        path = local_path(mmproj_label)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"mmproj '{mmproj_label}' not found in {llm_dir()}.")
        return path

    if entry is None:
        raise RuntimeError(
            "mmproj is set to 'auto', but the selected model is a local file "
            "without catalog information. Select the matching mmproj-*.gguf "
            "from the mmproj dropdown (place it in ComfyUI/models/LLM/).")
    if not entry.get("vision") or not entry.get("mmproj"):
        raise RuntimeError(
            f"'{entry['family']}' is a text-only model and cannot see images. "
            "Use a vision model (e.g. Mistral-Small-3.2-24B abliterated) for "
            "image captioning.")

    mm = entry["mmproj"]
    path = local_path(mm["filename"])
    if not os.path.isfile(path):
        print(f"[LocalLLM] Vision projector not present -> downloading "
              f"{mm['filename']} ({mm.get('size_gb', '?')} GB)")
        download_file(mm["repo"], mm["filename"], path)
    return path


# ---------------------------------------------------------------------------
# VRAM helper
# ---------------------------------------------------------------------------

def free_vram_gb():
    """Free VRAM of the current CUDA device in GB, or None."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return free / 1e9
    except Exception:
        return None


def vram_warning(size_gb, n_ctx=8192):
    """Return a warning string if the model will likely not fit, else None."""
    free = free_vram_gb()
    if free is None or not size_gb:
        return None
    # ~10% overhead for llama.cpp buffers + a rough KV-cache allowance.
    need = size_gb * 1.10 + max(0.5, n_ctx / 8192 * 1.0)
    if need > free:
        return (f"[LocalLLM] WARNING: model needs ~{need:.1f} GB but only "
                f"{free:.1f} GB VRAM are free. Lower n_gpu_layers, pick a "
                f"smaller quant, or unload other models first.")
    return None


def print_vram_overview():
    """Console overview at startup: which catalog entries fit right now."""
    free = free_vram_gb()
    if free is None:
        print("[LocalLLM] No CUDA device detected - models will run on CPU.")
        return
    fits = [e for e in load_catalog() if e["size_gb"] * 1.10 + 1.0 <= free]
    largest = max(fits, key=lambda e: e["size_gb"]) if fits else None
    msg = f"[LocalLLM] Free VRAM: {free:.1f} GB."
    if largest:
        msg += (f" Largest catalog model that fits: "
                f"{largest['family']} {largest['quant']} ({largest['size_gb']:g} GB).")
    print(msg)


# ---------------------------------------------------------------------------
# Downloader (stdlib only, resumable)
# ---------------------------------------------------------------------------

def hf_url(repo, filename):
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def _headers(resume_from=0):
    h = {"User-Agent": "ComfyUI-LocalLLM-Nodes"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    if resume_from > 0:
        h["Range"] = f"bytes={resume_from}-"
    return h


def _fmt_gb(n):
    return f"{n / 1e9:.2f} GB"


def download_file(repo, filename, dest, retries=3):
    """Download repo/filename from HuggingFace to dest (atomic via .part,
    resumable, progress printed to console)."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    part = dest + ".part"
    url = hf_url(repo, filename)

    for attempt in range(1, retries + 1):
        have = os.path.getsize(part) if os.path.isfile(part) else 0
        try:
            req = urllib.request.Request(url, headers=_headers(have))
            with urllib.request.urlopen(req, timeout=60) as resp:
                status = getattr(resp, "status", 200)
                if have and status != 206:
                    have = 0  # server ignored Range -> start over
                total = resp.headers.get("Content-Length")
                total = int(total) + have if total else None
                mode = "ab" if have else "wb"
                print(f"[LocalLLM] Downloading {filename}"
                      f"{' (resuming)' if have else ''} -> {dest}")
                done = have
                t0 = time.time()
                last = 0.0
                with open(part, mode) as fh:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        done += len(chunk)
                        now = time.time()
                        if now - last >= 2.0 or (total and done >= total):
                            last = now
                            speed = (done - have) / max(now - t0, 1e-6)
                            if total:
                                pct = done / total * 100
                                eta = (total - done) / max(speed, 1)
                                sys.stdout.write(
                                    f"\r[LocalLLM]   {pct:5.1f}%  {_fmt_gb(done)} / "
                                    f"{_fmt_gb(total)}  {speed / 1e6:6.1f} MB/s  "
                                    f"ETA {int(eta) // 60:02d}:{int(eta) % 60:02d}   ")
                            else:
                                sys.stdout.write(
                                    f"\r[LocalLLM]   {_fmt_gb(done)}  {speed / 1e6:6.1f} MB/s   ")
                            sys.stdout.flush()
                sys.stdout.write("\n")
            if total and os.path.getsize(part) != total:
                raise IOError(f"size mismatch after download "
                              f"({os.path.getsize(part)} != {total})")
            os.replace(part, dest)
            print(f"[LocalLLM] Download complete: {dest}")
            return dest
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise RuntimeError(
                    f"HuggingFace refused the download of {repo}/{filename} "
                    f"(HTTP {e.code}). If the repo is gated, set the HF_TOKEN "
                    "environment variable to a token with access.") from e
            if e.code == 404:
                raise RuntimeError(
                    f"File not found on HuggingFace: {url}\n"
                    "The repo may have renamed or removed this quant. Check "
                    "the repo page and update models.json / custom_models.json.") from e
            err = e
        except Exception as e:  # network hiccup -> retry with resume
            err = e
        print(f"[LocalLLM] Download attempt {attempt}/{retries} failed: {err}")
        time.sleep(3)

    raise RuntimeError(
        f"Could not download {filename} after {retries} attempts. Partial "
        f"file kept at {part} - simply run the node again to resume.")
