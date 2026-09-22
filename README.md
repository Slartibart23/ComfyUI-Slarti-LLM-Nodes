# ComfyUI-Slarti-LLM-Nodes

ComfyUI custom nodes for running **local GGUF language / vision-language models** via [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), plus a VRAM-unload utility. Built and tested for [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF) (main model + separate `mmproj` vision projector), but works with any llama.cpp-compatible GGUF.

> ⚠️ Note: the model this pack was tested with is an uncensored/"abliterated" model with no built-in safety behavior. You are responsible for the content you generate. The nodes themselves are model-agnostic.

| Node | Purpose |
| --- | --- |
| **LocalLLM Prompt Generator (GGUF)** | Text → text. Turn a short idea into a detailed SD/SDXL/Flux prompt, rewrite/translate prompts, or chain any text task. |
| **LocalVLM Image Caption (GGUF + mmproj)** | Image → text. Caption single images or whole batches with a multimodal GGUF model. |
| **LocalLLM Unload (passthrough)** | Frees the LLM from VRAM before heavy sampling steps. Passes its text input through unchanged. |
<!-- TODO: add the fourth node of v2.0 here -->

Both main nodes share one model cache: as long as model, mmproj, context size and GPU layers are identical, the model is loaded **once** and reused.

## ✨ What's new

**v2.0.1**
- Fixed false *"CPU-only"* detection with recent llama-cpp-python builds, which load their compute backends lazily. The GPU check now loads the backends first, so the node no longer tries to reinstall llama-cpp-python on every run.
- Windows: CUDA runtime DLLs (`cudart`, `cublas`, `cublasLt`) missing from a llama-cpp-python CUDA wheel are now copied automatically from your PyTorch installation. Previously the CUDA backend failed to load silently and everything ran on the CPU.

**v2.0.0**
- Automatic installation of a GPU-enabled llama-cpp-python on first use (see below).

## 📚 Prompt library

The [`prompts/`](prompts) folder contains ready-to-use system prompts. For **image models** (Krea2 / FLUX2 / FLUX2 Klein Base): image captioning, captioning with person replacement, and idea-to-prompt expansion with freely described persons. For **MiniMax H3** video: text-to-video, image-to-video, and four reference-to-video combinations (person/background images + voice audio references). Paste them into the node's `system_prompt` field.

**Note:** the H3 video prompts hard-code **German** as the spoken dialogue language by convention. [`prompts/README.md`](prompts/README.md) explains the setup and how to switch languages.

---

## Installation

### 1. Install the node pack

**Via ComfyUI-Manager (recommended):** open the Manager, click **Custom Nodes Manager**, search for **Slarti LLM Nodes** and install.

**Via comfy-cli:**

```
comfy node install slarti-llm-nodes
```

**Manually:**

```
cd ComfyUI/custom_nodes
git clone https://github.com/Slartibart23/ComfyUI-Slarti-LLM-Nodes.git
```

### 2. llama-cpp-python with GPU support (automatic)

A plain `pip install llama-cpp-python` gives you a **CPU-only** build, which is far too slow for large models. You don't need to do anything yourself: on the first run of a LocalLLM node, the pack checks whether llama-cpp-python is installed **with GPU offload** and, if not, installs a matching CUDA build into the **same Python that runs ComfyUI** (for portable installs that is `python_embeded`). It tries, in order:

1. A wheel already cached in `ComfyUI/models/LLM/.wheels` (offline, seconds).
2. A prebuilt CUDA wheel from the [JamePeng fork releases](https://github.com/JamePeng/llama-cpp-python/releases), matched to your OS, Python and CUDA version (taken from your PyTorch).
3. abetlen's prebuilt CUDA wheel index.
4. A source build with `CMAKE_ARGS=-DGGML_CUDA=on` (needs the CUDA Toolkit and build tools).

**Afterwards restart ComfyUI once**, so the new build is loaded. The startup log should then show a line like:

```
[LocalLLM] ComfyUI-Slarti-LLM-Nodes v2.0.1 loaded (4 nodes) - llama-cpp-python 0.4.0: GPU
```

> **Windows:** if llama-cpp-python is already loaded when the automatic install runs, Windows locks its DLLs and the install fails with `Access denied`. In that case close ComfyUI and install the downloaded wheel manually:
> ```
> python_embeded\python.exe -m pip install --force-reinstall --no-deps "ComfyUI\models\LLM\.wheels\<wheel-file>.whl"
> ```

To disable the automatic installation (e.g. if you manage your environment yourself), set the environment variable `LOCALLLM_NO_AUTOINSTALL=1`.

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

- **system_prompt** ships with a sensible SD/Flux prompt-engineering default. Edit freely, or use one from the prompt library.
- **user_prompt** is your idea ("a cozy cabin in a snowy forest").
- The optional **text_input** socket lets you pipe text from other nodes (e.g. a caption) into the request, which is great for caption → prompt-rewrite chains.

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

Because both nodes share the cache, this chain loads the model **once**. Note that switching between with/without mmproj or different `n_ctx` values triggers a reload, since a 27B model does not fit in VRAM twice.

### VRAM management

A 27B Q4_K_M (~16.5 GB) plus SDXL/Flux may not fit in VRAM simultaneously. Options:

1. Insert **LocalLLM Unload** between the text output and your sampler. It frees the LLM right before sampling.
2. Or set **unload_after_run** to `true` directly on the LLM/VLM node.
3. Or lower **n_gpu_layers** (e.g. 40) to keep part of the model in RAM.

---

## Node parameters (both main nodes)

| Parameter | Meaning |
| --- | --- |
| `n_ctx` | Context window. Vision tokens are large, so keep ≥ 8192 for captioning. |
| `n_gpu_layers` | `-1` = offload everything to GPU. Lower it if you run out of VRAM. |
| `temperature` / `top_p` / `top_k` / `repeat_penalty` | Standard sampling controls. Captioning defaults to a lower temperature (0.4) than prompt writing (0.7). |
| `seed` | Reproducible outputs (passed to llama.cpp). |
| `unload_after_run` | Free VRAM immediately after generation. |

`<think>…</think>` reasoning blocks (emitted by some Qwen3 variants) are stripped from the output automatically.

---

## Troubleshooting

**Startup log says `llama-cpp-python ...: CPU-only`**
Check the lines directly above it. You should see `ggml_cuda_init: found 1 CUDA devices` and `load_backend: loaded CUDA backend`. If they are missing:
- Your llama-cpp-python is a CPU build. Run a LocalLLM node once; the automatic installer replaces it (restart afterwards).
- Or `ggml-cuda.dll` cannot find its CUDA runtime. Since v2.0.1 the missing DLLs are copied from `torch/lib` automatically. This requires a CUDA build of PyTorch whose CUDA major version matches the llama-cpp-python wheel (e.g. both `cu13x`). You'll see `[LocalLLM] Copied CUDA runtime ...` in the log when this happens.

**Model fails to load: `missing tensor 'blk.NN....'`, `unknown model architecture`, or a generic `Failed to load model from file`, even though the file is complete**
Your installed llama-cpp-python bundles a llama.cpp that is older than the model. New architectures (Qwen3-VL, hybrid SSM models, NextN/MTP layers, etc.) need a recent build. Fix: install a prebuilt wheel from the JamePeng fork, no compiling required:

1. Go to <https://github.com/JamePeng/llama-cpp-python/releases>
2. Pick the newest release matching your **OS** (`win`/`linux`) and **CUDA** (`cu124`/`cu126`/`cu128`/`cu130`/...; RTX 50xx needs `cu128` or newer; check with `nvidia-smi`).
3. Expand **Assets** (collapsed by default!) and download the `.whl` matching your **Python** (`cp312` = 3.12 etc.). Check with `<your-python> --version`; for portable ComfyUI that's `python_embeded\python.exe --version`.
4. Close ComfyUI, then install into ComfyUI's Python: `<your-python> -m pip install --force-reinstall --no-deps <downloaded-file>.whl`, and start ComfyUI again.

The wheel MUST go into the same Python environment ComfyUI runs on (portable installs: `python_embeded\python.exe -m pip ...`, not your system Python).

**Automatic install fails with `[WinError 5] Access denied` on a `.dll`**
llama-cpp-python was already loaded, so Windows locks its files. Close ComfyUI and install the wheel from `ComfyUI/models/LLM/.wheels` manually (see Installation, step 2).

**Building from source fails: `Filename too long` / `Unable to checkout ... submodule` / CMake or compiler errors**
Windows source builds need git long-path support (`git config --system core.longpaths true`), Visual Studio Build Tools and the CUDA Toolkit. Skip all of that and use a prebuilt wheel.

**"No compatible multimodal chat handler found"**
Same root cause: llama-cpp-python is too old for the architecture. Same fix: prebuilt wheel above. The backend tries these handlers in order: `Qwen3VLChatHandler`, `Qwen25VLChatHandler`, `MiniCPMv26ChatHandler`, `Llava16ChatHandler`, `Llava15ChatHandler`.

**"No GGUF model found" / dropdowns empty**
Files are not in `ComfyUI/models/LLM/`, or ComfyUI wasn't restarted after copying. Vision needs BOTH files from the same HuggingFace repo: the main model and the `mmproj-*.gguf`.

**Model loads but ignores the image / hallucinates content**
Almost always a mismatched or missing mmproj. Use the mmproj from the *same* HuggingFace repo as the main model.

**Out of memory**
Lower `n_gpu_layers`, lower `n_ctx`, use a smaller quant (Q4_K_S instead of Q6_K), or use the Unload node before sampling.

**Slow first run**
The first generation includes model load time (tens of seconds for 27B). Subsequent runs reuse the cached instance.

*Note: pip may print "dependency conflict" warnings from unrelated packages after installing a wheel. These are harmless for this node pack.*

---

## Responsible use

These nodes run whatever GGUF you point them at, including uncensored/"abliterated" models that have no built-in safety behavior. You are responsible for the content you generate and for complying with the licenses of the models you use.

## License

MIT, see [LICENSE](LICENSE).
