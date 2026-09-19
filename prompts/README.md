# Prompt Library

Ready-to-use **system prompts** for the LocalLLM Prompt Generator / LocalVLM Image Caption nodes. Each file targets one MiniMax H3 generation mode. Easiest way to use them: add a **LocalLLM System Prompt (Library)** node, pick the template in its dropdown and connect its output to the `system_prompt_input` socket of the Prompt Generator / Image Caption node. (Pasting the code block into the node's `system_prompt` field still works, too.) Any `.md` file you add to this folder appears in the dropdown after pressing `R` in ComfyUI - the first fenced code block is used as the prompt, the first `# heading` as its name.

## ⚠️ Dialogue language: German (video prompts only)

The **MiniMax H3 video prompts** hard-code **German** as the spoken dialogue language (`<d>[German] ...</d>`), regardless of the input language. This is a deliberate convention of this library, not an H3 requirement. To use another language, edit the DIALOGUE rule in the prompt: replace every `[German]` with e.g. `[English]` or `[French]` — always the **English language name, capitalized exactly like that** (never all caps, never the native name like `[Deutsch]`). H3 supports 11 dialogue languages. The image prompts (Krea2/FLUX2) produce English output and are unaffected.

## Files

### Image models (Krea2 / FLUX2 / FLUX2 Klein Base)

| File | Mode | Node |
| --- | --- | --- |
| `image-caption.md` | Describe an image as a recreation prompt | LocalVLM Image Caption |
| `image-caption-person-replace.md` | Same image, swap in a described person | LocalVLM Image Caption |
| `image-prompt-generator-persons.md` | Expand an idea with described persons | LocalLLM Prompt Generator |

### Video model (MiniMax H3)

| File | Mode | References attached in H3 |
| --- | --- | --- |
| `minimax-h3-t2v.md` | Text-to-video | none |
| `minimax-h3-i2v.md` | Image-to-video | 1 image (first frame) |
| `minimax-h3-ref2vid-1img-1audio.md` | Reference-to-video | 1 person image + 1 voice audio |
| `minimax-h3-ref2vid-1img-2audio.md` | Reference-to-video | 1 image with 2 people + 2 voice audios |
| `minimax-h3-ref2vid-2img-1audio.md` | Reference-to-video | 1 person image + 1 background image + 1 voice audio |
| `minimax-h3-ref2vid-2img-2audio.md` | Reference-to-video | 1 image with 2 people + 1 background image + 2 voice audios |

## Recommended node settings

All prompts in this library assume: `n_ctx` **8192**, `max_tokens` **4096**, `suppress_thinking` **true**, `temperature` **0.5–0.6**. Lower context or token budgets can cut the answer off mid-prompt.

## Which node?

- **i2v**: use the **LocalVLM Image Caption** node (it sees the image and anchors `<Picture 1>` to real content). Put your direction into the `prompt` field.
- **t2v and all ref2vid modes**: use the **LocalLLM Prompt Generator** node. The LLM never sees the reference files — it only writes the marker bindings (`<Picture N>`, `<Audio N>`); you attach the actual files in H3.

## H3 reference rules (summary)

- Audio can never be the only reference — it must travel with at least one image or video.
- Limits per generation: up to 9 images, 3 video clips, 3 audio clips, 12 files total.
- A voice audio transfers **timbre only**; the spoken words always come from the prompt text.
- Attach reference files in the same order as the markers: `<Picture 1>` = first image, `<Audio 1>` = first audio, and so on.

## User prompt: how free can it be?

**Free-flowing text in any language works.** The system prompts extract what they need; rigid templates are optional comfort, not a requirement. Only three things must be explicit somewhere in your text:

1. **Duration** — e.g. "8 Sekunden" / "10 seconds". Without it, neither the LLM nor H3 can pace the action.
2. **Spoken words in quotation marks, attributed to a person** — e.g. `Louise says: "The storm is coming sooner."` Quotes are the signal for *verbatim*; unquoted descriptions ("she mumbles something about rain") are treated as paraphrase and rewritten freely.
3. **With two speakers: the characters must be tellable apart** — by name, position, or feature ("Louise", "the man on the right", "the older one"). No S1/S2 labels needed.

### How the S1/S2 voice mapping works (two-audio modes)

You do **not** assign speaker IDs yourself. The LLM assigns them: whoever **speaks first** becomes **(S1)** and is bound to **`<Audio 1>`**; the other becomes **(S2)** with **`<Audio 2>`**. The generated prompt then states the mapping in plain words in its opening sentence — e.g. *"S1 is Charles, the gray-haired man, and speaks with the timbre of `<Audio 1>`; S2 is Louise with `<Audio 2>`"*. Read that sentence and attach your audio files in the matching order. To force a specific mapping instead, just say it anywhere in your text, in any phrasing: `"Audio 1 is Louise's voice"` or `"Audio 2 = Charles"` — an explicit statement always wins over the speaks-first default.

### Structured template (optional)

If you prefer structure, this works well and minimizes model interpretation:

```
Duration: 8 seconds

Scene: <who and where - appearance, clothing, environment>

Action: <what happens, in order - include WHEN each person speaks>

Louise says: "Exact words to be spoken."
Charles replies: "Exact words to be spoken."
```

Labels and text may be written in any language — adapt this template freely. Quoted lines are rendered in the configured dialogue language (German by default).

Each prompt file below ends with a filled example user prompt for its mode.
