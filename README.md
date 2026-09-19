# ComfyUI-LocalLLM-Nodes

ComfyUI custom nodes for running **local GGUF language / vision-language models** via [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) — with a built-in **model catalog**, **automatic model download**, a **self-installing GPU backend** and a **prompt library** for Krea2 / FLUX2 images and MiniMax H3 video.

> ⚠️ The catalog ships with uncensored / "abliterated" models that have no built-in safety behavior. You are responsible for the content you generate and for complying with the model licenses. The nodes themselves are model-agnostic.

| Node | Purpose |
| --- | --- |
| **LocalLLM Prompt Generator (GGUF)** | Text → text. Turn a short idea into a detailed image / video prompt, rewrite or translate prompts, chain any text task. |
| **LocalVLM Image Caption (GGUF + mmproj)** | Image → text. Caption single images or batches with a vision model. |
| **LocalLLM System Prompt (Library)** | Pick a ready-made system prompt from `prompts/` and feed it into either node. |
| **LocalLLM Unload (passthrough)** | Frees the LLM from VRAM before heavy sampling. Passes its text through unchanged. |

## What's new in 2.0

- **Zero-setup installation.** On first use the node pack installs a CUDA-enabled `llama-cpp-python` that matches your OS, Python and CUDA version into ComfyUI's own Python — works on RunPod and on portable ComfyUI for Windows. The wheel is cached in `models/LLM/.wheels/`, so a RunPod pod restart only needs seconds, not a new download.
- **Model catalog with auto-download.** Pick a model in the dropdown — the label shows the quant and its download size — and it is fetched from HuggingFace when the node runs for the first time. Downloads resume after interruptions. Built in: **Mistral-Small-3.2-24B abliterated** (vision-capable) and **Qwen3-4B abliterated** (text-only, tiny) in all sensible quants from 3-bit upwards.
- **Vision projector handled for you.** For vision models the matching `mmproj` is downloaded automatically (`mmproj` = `auto`).
- **Custom models.** Add your own HuggingFace repos in `custom_models.json` — they appear in the dropdown and download the same way. Files you drop into `models/LLM/` manually show up as `Local: …`.
- **Prompt library node.** No more copy-pasting: select a template from `prompts/` and plug it into the `system_prompt_input` socket.
- **Tuned defaults**: `max_tokens` 6144, `temperature` 1.0, `top_p` 0.98, `top_k` 40, `repeat_penalty` 1.1, `n_ctx` 8192.

---

## Installation

### With ComfyUI-Manager (recommended)

**Manager → Install via Git URL** → paste

```
https://github.com/Slartibart23/ComfyUI-LocalLLM-Nodes-v1.2.0
```

The Manager runs `install.py`, which installs the GPU backend right away. Restart ComfyUI. Done.

### Manually (RunPod / JupyterLab terminal, or any shell)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Slartibart23/ComfyUI-LocalLLM-Nodes-v1.2.0.git
```

Restart ComfyUI. Nothing else to install — the first time you run a node it installs `llama-cpp-python` and downloads the selected model. Watch the ComfyUI console for progress (installation takes about 1–2 minutes the first time, the model download depends on its size and your connection).

> **Portable ComfyUI on Windows:** works out of the box. The backend is installed into `python_embeded`, the same Python that runs ComfyUI.
>
> **RunPod:** if ComfyUI's Python environment is not on the persistent `/workspace` volume, the backend is gone after a pod restart. That's fine — the cached wheel in `models/LLM/.wheels/` is re-installed automatically within seconds on the next run.

To disable auto-installation entirely, set the environment variable `LOCALLLM_NO_AUTOINSTALL=1` and install `llama-cpp-python` yourself (see Troubleshooting).

---

## Usage

### Prompt generation (text → text)

```
[LocalLLM System Prompt] ──system_prompt──┐
                                          ▼
[Your text node] ──text_input──> [LocalLLM Prompt Generator] ──text──> [CLIP Text Encode] ──> [KSampler]
```

- **model** – pick a catalog entry. The size in the label is the download size; count on ~10 % more VRAM plus context. A warning is printed to the console if the model probably won't fit in the free VRAM.
- **system_prompt** – used unless something is connected to `system_prompt_input`.
- **user_prompt** / **text_input** – your idea, either typed into the node or piped in from another node (e.g. a caption). Both are combined if both are present.
- **suppress_thinking** – appends `/no_think` (Qwen3). Harmless for Mistral.

### Image captioning (image → text)

```
[Load Image] ──image──> [LocalVLM Image Caption] ──caption──> [Show Text / Prompt Generator text_input]
```

- Only vision-capable families are listed in the model dropdown (Mistral-Small-3.2 from the built-in catalog; Qwen3-4B cannot see images).
- **mmproj** = `auto` downloads and uses the projector that belongs to the chosen catalog model. For `Local: …` models select the matching `mmproj-*.gguf` from the dropdown.
- Batches are supported: every image is captioned separately, results are joined with `batch_separator`.
- Large images are downscaled to 1568 px on the long side before encoding.

### Prompt library

The **LocalLLM System Prompt (Library)** node lists every `.md` file in `prompts/`: Krea2 / FLUX2 image captioning, captioning with person replacement, idea-to-prompt with described persons, and six MiniMax H3 video modes (t2v, i2v, four reference-to-video combinations). The optional `append` field adds extra instructions. Add your own `.md` files (first ` ``` ` block = prompt, first `# heading` = name) and press `R` in ComfyUI. See [`prompts/README.md`](prompts/README.md) — note the H3 prompts hard-code **German** as dialogue language.

### VRAM management

A 24B model (14–19 GB) plus Krea2 / FLUX2 usually does not fit in VRAM at the same time. Options:

1. Insert **LocalLLM Unload** between the text output and your sampler.
2. Set **unload_after_run** = `true` on the LLM/VLM node.
3. Lower **n_gpu_layers** (e.g. 40) to keep part of the model in RAM.
4. Use **Qwen3-4B** (2–4 GB) when the LLM has to share the card with a big image model.

Both LLM nodes share one model cache: as long as model, mmproj, `n_ctx` and `n_gpu_layers` are identical, the model is loaded once and reused.

---

## Model catalog

`models.json` contains the built-in families. Sizes in the dropdown are download sizes.

| Family | Vision | Quants |
| --- | --- | --- |
| Mistral-Small-3.2-24B abliterated ([i1-GGUF](https://huggingface.co/mradermacher/Huihui-Mistral-Small-3.2-24B-Instruct-2506-abliterated-llamacppfixed-i1-GGUF), [static](https://huggingface.co/mradermacher/Huihui-Mistral-Small-3.2-24B-Instruct-2506-abliterated-llamacppfixed-GGUF)) | yes (mmproj from the static repo) | i1-IQ3_XXS 9.4 GB … i1-Q4_K_M 14.4 GB … i1-Q6_K 19.4 GB, Q8_0 25.2 GB |
| Qwen3-4B abliterated ([Mungert](https://huggingface.co/Mungert/Qwen3-4B-abliterated-GGUF)) | no | IQ3_XXS 1.7 GB … Q4_K_M 2.5 GB … Q8_0 4.3 GB, BF16 8.1 GB |

Rule of thumb: the largest quant that leaves ~2 GB free after your image model is loaded, or use `unload_after_run`. 1- and 2-bit quants are intentionally left out.

### Adding your own models

Copy `custom_models.example.json` to `custom_models.json` and add families/quants (HF repo + file name + rough size; `mmproj` optional for vision models). Restart ComfyUI. `custom_models.json` is git-ignored, so it survives updates of the node pack.

Models you already have: drop the `.gguf` into `ComfyUI/models/LLM/` (mmproj files too) and press `R` — they appear as `Local: <file>` / in the mmproj dropdown.

Gated HuggingFace repos: set the `HF_TOKEN` environment variable.

---

## Node parameters

| Parameter | Meaning |
| --- | --- |
| `n_ctx` | Context window. Vision tokens are large — keep ≥ 8192 for captioning. |
| `n_gpu_layers` | `-1` = offload everything to GPU. Lower it if you run out of VRAM. |
| `max_tokens` | Generation budget. Reasoning models spend tokens thinking first. |
| `temperature` / `top_p` / `top_k` / `repeat_penalty` | Sampling. Captioning defaults to temperature 0.4, prompt writing to 1.0. |
| `seed` | Reproducible outputs (passed to llama.cpp). |
| `suppress_thinking` | Appends `/no_think` to the user message (Qwen3). |
| `unload_after_run` | Free VRAM immediately after generation. |

`<think>…</think>` blocks and `<prompt>…</prompt>` wrappers are handled automatically: only the final answer is returned.

---

## Troubleshooting

**"llama-cpp-python was installed. Please restart ComfyUI once"** — expected on the very first run when a CPU-only build was already loaded. Restart ComfyUI and run again.

**Auto-install fails (no network, exotic Python/CUDA)** — install a wheel yourself into ComfyUI's Python:

1. Check your setup: `<comfy-python> -c "import sys,torch;print(sys.version, torch.version.cuda)"` (portable Windows: `python_embeded\python.exe`).
2. Go to <https://github.com/JamePeng/llama-cpp-python/releases>, pick the newest release for your **OS** and **CUDA** (`cu124`/`cu126`/`cu128`/…; RTX 50xx needs `cu128`+), expand **Assets**, download the `.whl` for your Python (`cp312` = 3.12).
3. `<comfy-python> -m pip install --force-reinstall <downloaded>.whl`, restart ComfyUI.

Fallbacks the auto-installer also tries, in this order: cached wheel → JamePeng release → `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/<cuXXX>` → source build (`CMAKE_ARGS="-DGGML_CUDA=on"`, needs the CUDA toolkit).

**Model fails to load: `unknown model architecture` / `missing tensor` / generic `Failed to load model`** — the installed llama.cpp is older than the model. Delete `ComfyUI/models/LLM/.wheels/`, restart ComfyUI and run again to fetch a newer build.

**"No compatible multimodal chat handler found"** — same cause, same fix. The backend tries `Qwen3VLChatHandler`, `Qwen25VLChatHandler`, `MiniCPMv26ChatHandler`, `Llava16ChatHandler`, `Llava15ChatHandler` (order depends on the model name).

**Model loads but ignores the image / hallucinates** — mismatched or missing mmproj. With catalog models keep `mmproj` = `auto`; for local models use the mmproj from the *same* HuggingFace repo.

**Download stops / is slow** — just run the node again; partial `.part` files are resumed. Check free disk space in `models/LLM/`.

**DLL errors on startup (`cudart64_*.dll not found`)** — the wheel's CUDA version doesn't match your driver. Delete `.wheels/`, update the NVIDIA driver or install a wheel for a lower CUDA version manually.

**Out of memory** — lower `n_gpu_layers`, lower `n_ctx`, choose a smaller quant, or use the Unload node before sampling.

**Slow first run** — includes installation, download and model load. Subsequent runs reuse the cached instance.

*pip may print "dependency conflict" warnings from unrelated packages — harmless for this node pack.*

---

## Upgrading from 1.x

Existing workflows keep working: the old bare file-name values in the `model` dropdown are still accepted and resolved as long as the file is in `models/LLM/`. To switch to the catalog (auto-download, mmproj `auto`), simply re-select the model. The `prompts/` library is unchanged.

## License

MIT — see [LICENSE](LICENSE).
