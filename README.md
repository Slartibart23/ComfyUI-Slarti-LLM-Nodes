# ComfyUI-LocalLLM-Nodes

Two ComfyUI custom nodes for running **local GGUF language / vision-language models** via [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) — plus a small VRAM-unload utility. Built and tested for [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF) (main model + separate `mmproj` vision projector), but works with any llama.cpp-compatible GGUF.

> ⚠️ Note: the model this pack was tested with is an uncensored/"abliterated" model with no built-in safety behavior. You are responsible for the content you generate. The nodes themselves are model-agnostic.

| Node | Purpose |
| --- | --- |
| **LocalLLM Prompt Generator (GGUF)** | Text → text. Turn a short idea into a detailed SD/SDXL/Flux prompt, rewrite/translate prompts, or chain any text task. |
| **LocalVLM Image Caption (GGUF + mmproj)** | Image → text. Caption single images or whole batches with a multimodal GGUF model. |
| **LocalLLM Unload (passthrough)** | Frees the LLM from VRAM before heavy sampling steps. Passes its text input through unchanged. |

Both main nodes share one model cache: as long as model, mmproj, context size and GPU layers are identical, the model is loaded **once** and reused.

## 📚 Prompt library

The [`prompts/`](prompts/) folder contains ready-to-use system prompts. For **image models** (Krea2 / FLUX2 / FLUX2 Klein Base): image captioning, captioning with person replacement, and idea-to-prompt expansion with freely described persons. For **MiniMax H3** video: text-to-video, image-to-video, and four reference-to-video combinations (person/background images + voice audio references). Paste them into the node's `system_prompt` field. **Note:** the H3 video prompts hard-code **German** as the spoken dialogue language by convention — [`prompts/README.md`](prompts/README.md) explains the setup and how to switch languages.

---

## Installation

### 1. Install the node pack

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOURNAME/ComfyUI-LocalLLM-Nodes.git
```

### 2. Install llama-cpp-python (with GPU support!)

A plain `pip install llama-cpp-python` gives you a **CPU-only** build. For NVIDIA GPUs, install a CUDA wheel into ComfyUI's Python environment:

```bash
# Prebuilt CUDA wheels (adjust cu124 to your CUDA version):
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

**For very recent architectures (Qwen3-VL etc.)** the upstream package may lag behind. In that case build a current fork from source:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir \
    git+https://github.com/JamePeng/llama-cpp-python.git
```

> **Windows note:** building from source requires Visual Studio Build Tools + CUDA Toolkit. Prefer prebuilt wheels when available.

### 3. Add your models

Download a quant **and** the matching `mmproj` file (needed for image input) from the same repo, e.g. [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF/tree/main). Rough VRAM guide for the 27B: Q4_K_M ≈ 16.5 GB, Q5_K_M ≈ 19 GB, Q6_K ≈ 22 GB, Q8_0 ≈ 29 GB (plus context). Place the files in:

```
ComfyUI/models/LLM/
├── Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf
└── mmproj-Qwen3.8-27B-ABLITERATED-F16.gguf
```

The folder is created automatically on first start. Files containing `mmproj` in their name appear in the **mmproj** dropdown; everything else appears in the **model** dropdown.

### 4. Restart ComfyUI

The nodes appear under the **LocalLLM** category.

---

## Usage

### Prompt generation (text → text)

```
[LocalLLM Prompt Generator] ──text──> [CLIP Text Encode] ──> [KSampler]
```

- **system_prompt** ships with a sensible SD/Flux prompt-engineering default — edit freely.
- **user_prompt** is your idea ("a cozy cabin in a snowy forest").
- The optional **text_input** socket lets you pipe text from other nodes (e.g. a caption) into the request — great for caption → prompt-rewrite chains.

### Image captioning (image → text)

```
[Load Image] ──image──> [LocalVLM Image Caption] ──caption──> [Show Text / Save Text]
```

- Select **both** the main model *and* the matching **mmproj** file. Without the mmproj the model cannot see images and the node will tell you so.
- Image **batches** are supported: each image is captioned separately and results are joined with `batch_separator`.
- Large images are downscaled to max 1568 px on the long side before encoding to keep vision token counts reasonable.

### Caption → new prompt (chaining both nodes)

```
[Load Image] ──> [LocalVLM Image Caption] ──caption──> [LocalLLM Prompt Generator (text_input)] ──> [CLIP Text Encode]
```

Because both nodes share the cache, this chain loads the model **once** (load the caption node's model *with* mmproj; the prompt node can reuse a text-only instance — note that switching between with/without mmproj or different `n_ctx` values triggers a reload, since a 27B model does not fit in VRAM twice).

### VRAM management

A 27B Q4_K_M (~16.5 GB) plus SDXL/Flux may not fit in VRAM simultaneously. Options:

1. Insert **LocalLLM Unload** between the text output and your sampler — it frees the LLM right before sampling.
2. Or set **unload_after_run** to `true` directly on the LLM/VLM node.
3. Or lower **n_gpu_layers** (e.g. 40) to keep part of the model in RAM.

---

## Node parameters (both nodes)

| Parameter | Meaning |
| --- | --- |
| `n_ctx` | Context window. Vision tokens are large — keep ≥ 8192 for captioning. |
| `n_gpu_layers` | `-1` = offload everything to GPU. Lower it if you run out of VRAM. |
| `temperature` / `top_p` / `top_k` / `repeat_penalty` | Standard sampling controls. Captioning defaults to a lower temperature (0.4) than prompt writing (0.7). |
| `seed` | Reproducible outputs (passed to llama.cpp). |
| `unload_after_run` | Free VRAM immediately after generation. |

`<think>…</think>` reasoning blocks (emitted by some Qwen3 variants) are stripped from the output automatically.

---

## Troubleshooting

**Model fails to load: `missing tensor 'blk.NN....'`, `unknown model architecture`, or a generic `Failed to load model from file` — even though the file is complete**
Your installed llama-cpp-python bundles a llama.cpp that is older than the model. New architectures (Qwen3-VL, hybrid SSM models, NextN/MTP layers, etc.) need a recent build. Fix: install a prebuilt wheel from the JamePeng fork — no compiling required:

1. Go to https://github.com/JamePeng/llama-cpp-python/releases
2. Pick the newest release matching your **OS** (`win`/`linux`) and **CUDA** (`cu124`/`cu126`/`cu128`/...; RTX 50xx needs `cu128`+; check with `nvidia-smi`).
3. Expand **Assets** (collapsed by default!) and download the `.whl` matching your **Python** (`cp312` = 3.12 etc.). Check with `<your-python> --version`; for portable ComfyUI that's `python_embeded\python.exe --version`.
4. Install into ComfyUI's Python: `<your-python> -m pip install --force-reinstall <downloaded-file>.whl`, then restart ComfyUI.

The wheel MUST go into the same Python environment ComfyUI runs on (portable installs: `python_embeded\python.exe -m pip ...`, not your system Python).

**Building from source fails: `Filename too long` / `Unable to checkout ... submodule` / CMake or compiler errors**
Windows source builds need git long-path support (`git config --system core.longpaths true`), Visual Studio Build Tools and the CUDA Toolkit. Skip all of that: use the prebuilt wheel above.

**"No compatible multimodal chat handler found"**
Same root cause: llama-cpp-python too old for the architecture. Same fix: prebuilt wheel above. The backend tries these handlers in order: `Qwen3VLChatHandler`, `Qwen25VLChatHandler`, `MiniCPMv26ChatHandler`, `Llava16ChatHandler`, `Llava15ChatHandler`.

**"No GGUF model found" / dropdowns empty**
Files are not in `ComfyUI/models/LLM/`, or ComfyUI wasn't restarted after copying. Vision needs BOTH files from the same HuggingFace repo: the main model and the `mmproj-*.gguf`.

**Model loads but ignores the image / hallucinates content**
Almost always a mismatched or missing mmproj. Use the mmproj from the *same* HuggingFace repo as the main model.

**DLL errors on startup (e.g. `cudart64_*.dll not found`)**
The wheel's CUDA version doesn't match your system. Install a wheel for a CUDA version your driver supports, or install the matching CUDA runtime from https://developer.nvidia.com/cuda-downloads

**Out of memory**
Lower `n_gpu_layers`, lower `n_ctx`, use a smaller quant (Q4_K_S instead of Q6_K), or use the Unload node before sampling.

**Slow first run**
The first generation includes model load time (tens of seconds for 27B). Subsequent runs reuse the cached instance.

*Note: pip may print "dependency conflict" warnings from unrelated packages after installing the wheel — these are harmless for this node pack.*

---

## Responsible use

These nodes run whatever GGUF you point them at, including uncensored/"abliterated" models that have no built-in safety behavior. You are responsible for the content you generate and for complying with the licenses of the models you use.

## License

MIT — see [LICENSE](LICENSE).
