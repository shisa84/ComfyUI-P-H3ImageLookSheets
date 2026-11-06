# ComfyUI-P-H3ImageLookSheets

Reference shots of a person wearing a specific look, built on
**P-H3-Image** (`p-h3-image Conditioning`): MiniMax H3 FL2VA with Pruna's
P-H3-Image LoRA.

Reference images go in — the person, and the look (an outfit shown in one
or several photos, shoes, accessories) — and P-H3-Image renders that
person, in that look, once per configured shot: one still image per shot,
all in a single run.

Same pipeline as [ComfyUI-H3LookSheets](https://github.com/shisa84/ComfyUI-H3LookSheets)
(describe the references, configure the shots, write the prompt), with a
prompt written for P-H3-Image's edit mode instead of ref2va's video
prompt.

---

## The pipeline, node by node

As wired in `example_workflows/ComfyUI-P-H3ImageLookSheets.json`.

1. **Load the reference photos** (`LoadImage`): the person, then the look
   (front/back of a garment, shoes, accessories…).
2. **Describe them** with two **Describe Reference** nodes: `target=person`
   with the person photo(s), `target=outfit` with the outfit photo(s) →
   `description` (JSON, one entry per image) + `images`.
3. **Route the images**: both `images` outputs go into **Image Aggregator**
   → `image_1` … `image_9`, in the order the prompt numbers them.
   `image_1` (the person) goes through **Scale Image to Total Pixels**
   (4 MP) before **p-h3-image Conditioning**; the output image takes its
   size.
4. **Write the prompts**: both `description` outputs, plus one **Shot
   Config** per shot, go into **Look Sheet Prompt - Custom Shots** →
   `prompts`, one per shot. **Prompt View** shows them colorized and can
   override them.
5. **Generate**: `prompts` and the 9 images go into **p-h3-image
   Conditioning**, then Basic Guider, Sampler Custom Advanced (8-step
   ManualSigmas, `euler`), VAE Decode and Save Image. ComfyUI runs this
   chain once per prompt, so every shot comes out as its own image.

```
LoadImage (person) ─► Describe Reference (target=person) ─► description ──┐
                               │ images                                   │
LoadImage (outfit) ×N ─► Describe Reference (target=outfit) ─► description┤
                               │ images                Shot Config ×N ────┤
                               ▼                                          ▼
                       Image Aggregator             Look Sheet Prompt - Custom Shots
                               │ image_1…9                   │ prompts (one per shot)
                               │                             ▼
                               │                        Prompt View
                               ▼                             │ prompts
                     p-h3-image Conditioning ◄───────────────┘
                               │ positive, latent   (once per shot)
                               ▼
           Basic Guider ─► Sampler Custom Advanced ─► VAE Decode ─► Save Image
```

---

## Nodes

| Node | Category | What it does |
|---|---|---|
| **Look Sheet Prompt - Custom Shots (P-H3 Image)** | `P-H3ImageLookSheets` | Writes one P-H3-Image edit prompt per connected shot (`PH3LookSheetsPrompt`) |
| **Shot Config (P-H3 Image Look Sheet)** | `P-H3ImageLookSheets` | One shot's framing/expression, packed for the node above (`PH3LookSheetsShotConfig`) |
| **Describe Reference (P-H3 Image Look Sheet)** | `P-H3ImageLookSheets` | One-sentence vision description per reference photo (up to 8), with retry and fallback prompts (`PH3LookSheetsDescribe`) |
| **Image Aggregator (P-H3 Image Look Sheet)** | `P-H3ImageLookSheets` | Routes the person and outfit images onto `p-h3-image Conditioning`'s `image_1` … `image_9` (`PH3LookSheetsImageAggregator`) |
| **Prompt View (P-H3 Image Look Sheet)** | `P-H3ImageLookSheets` | Colorized preview of every shot's prompt, with a copy button and an edit-a-copy override (`PH3LookSheetsPromptView`) |

---

### Look Sheet Prompt - Custom Shots (P-H3 Image)

Writes one edit prompt per shot. Each prompt starts with `Edit <Picture 1>.`,
keeps the person from `<Picture 1>` (or `<Picture 1>`, `<Picture 2>`… when
the person is shown across several photos), dresses them in every outfit
item, each named with its own `<Picture N>`, and states the shot's framing,
pose, expression and backdrop.

Up to 15 `shot_N` sockets (plug one, the next appears), each fed by a
**Shot Config** node; however many are connected becomes the number of
prompts, and of images.

| Input | Type | Notes |
|---|---|---|
| `person_description` | STRING | From **Describe Reference** (`target=person`) or written by hand |
| `outfit_description` | STRING | From **Describe Reference** (`target=outfit`) or written by hand |
| `person_gender` | `auto` / `female` / `male` | Auto-detects from the first person description's wording; override when needed |
| `backdrop` | STRING (multiline) | Default: plain light neutral grey studio backdrop |
| `shot_0` … `shot_14` | STRING (Autogrow) | Each fed by a **Shot Config** node |

| Output | Type | Notes |
|---|---|---|
| `prompts` | STRING list | One prompt per connected shot, in shot order. Wire into **Prompt View** or straight into `p-h3-image Conditioning`'s `prompt` |
| `shot_count` | INT | How many shots (and images) there are |

`person_description` / `outfit_description` take plain text or **Describe
Reference**'s JSON: each JSON entry becomes its own `<Picture N>`, person
photo(s) first, then outfit photo(s) — the same order **Image Aggregator**
wires them. An outfit entry left empty is kept as "the outfit shown" so
the numbering never shifts.

### Shot Config (P-H3 Image Look Sheet)

| Combo | Options |
|---|---|
| `shot_description` | body region and camera orientation together, e.g. `Upper Body - Back` (5 regions × 8 orientations) |
| `expression` | neutral, happy, smiling, sad, angry, surprised, scared, disgusted, shy/embarrassed, confident, serious, laughing, crying, smirking, confused |

Regions: `Full Body`, `Upper Body` (head to hips), `Lower Body` (waist to
feet), `Waist` (chest to upper thighs), `Head`. Orientations: `Front`,
`Front 3/4 Left`, `Front 3/4 Right`, `Left Profile`, `Right Profile`,
`Back 3/4 Left`, `Back 3/4 Right`, `Back`. `Left`/`Right` are sides of the
image: `Left Profile` has the nose pointing to the left edge of the image,
`Front 3/4 Left` and `Back 3/4 Left` have the body angled toward it.

`expression` is only written when the face is in frame: region `Full Body`,
`Upper Body` or `Head`, orientation `Front`, `Front 3/4 Left` or
`Front 3/4 Right`. For `Full Body`, `Upper Body` and `Head`, the prompt
also asks for the whole head to stay inside the frame; for `Lower Body` and
`Waist`, it states that the head is outside the frame and doesn't describe
the face or hair.

Each shot only names the outfit items its region shows, when **Describe
Reference** knows their category (see its per-category combos):

| Region | Items named |
|---|---|
| `Full Body` | top, bottom, shoes, accessories |
| `Upper Body` | top, bottom, accessories |
| `Lower Body` | top, bottom, shoes |
| `Waist` | top, bottom |
| `Head` | top, accessories |

A back view uses an item's back description (`top_image_back`,
`bottom_image_back`) when there is one. A photo described without a
category is named in every shot.

---

### Describe Reference (P-H3 Image Look Sheet)

Wraps core's **Generate Text** with one of two fixed prompts selected by
`target`:

| `target` | Asks the vision model to… |
|---|---|
| `person` | Describe face, hair, eyes, skin, body shape; ignore outfit and clothes |
| `outfit` | Describe the outfit and accessories only; skip hair |

One node per target. `images` is an Autogrow input (`image_0` … `image_7`,
plug one, the next appears). Connect them in the order they should reach
`p-h3-image Conditioning`.

| Output | Notes |
|---|---|
| `description` | JSON, one entry per connected image: `{"person_0": "..."}` or `{"outfit_0": "...", "outfit_1": "...", ...}`; an outfit image assigned categories gets one description per category, `{"outfit_0": {"top": "...", "bottom": "..."}, "outfit_1": {"shoes": "..."}}`. Plug straight into **Look Sheet Prompt - Custom Shots** |
| `debug` | Per image: the exact prompt sent and every raw reply, retries included |
| `images` | The connected images, same order as `description` — wire into **Image Aggregator** |

With `target=outfit`, per-category combos say which connected image is the
source for what:

| Combo | Use |
|---|---|
| `top_image_front` / `top_image_back` | Top, front and back view |
| `bottom_image_front` / `bottom_image_back` | Bottom (skirt, trousers, dress bottom), front and back view |
| `shoes_image` | Shoes |
| `accessory_image_1` … `accessory_image_4` | Up to 4 distinct accessories, each from its own photo |

All default to `auto` (each image describes the whole outfit). As soon as
one is set, each image describes only what it was assigned and skips what
another image covers, one description per assigned category (a photo
assigned both `top` and `bottom` gets two). Set a `..._back` only when the
back actually differs from the front. Assigning categories lets **Look
Sheet Prompt** name only the items each shot shows.

Retries automatically (up to 3 times, with a new seed) on an empty reply, a
refusal, a leaked chat-template role tag, or — for an image assigned a
category — a "there's none here" answer, then once more with a simpler
fallback prompt; an image that still fails gets an empty entry.

Inherited from **Generate Text**: `max_length`, `sampling_mode`,
`thinking`, `use_default_template`, `mtp`.

---

### Image Aggregator (P-H3 Image Look Sheet)

Concatenates `images_person`, `images_outfit`, `images_extra_1`,
`images_extra_2` (each a single image or a list, e.g. **Describe
Reference**'s `images` output), in that order, onto `image_1` … `image_9`.
Wire them into `p-h3-image Conditioning`'s slots of the same name; unused
slots come out empty and are skipped. Past 9 images, the rest are dropped
with a warning. `images` returns the same images as one list.

---

### Prompt View (P-H3 Image Look Sheet)

Shows every shot's prompt in one text, each under a `===== Shot N =====`
header, with `<Picture N>` tags colorized, and a **Copy** button.

Read-only by default, always mirroring `prompts`. **Override prompt**
unlocks it for editing a copy in place — the node then outputs the edited
prompts, split back on the `===== Shot N =====` headers (add or remove a
header to add or remove a shot), until **Revert to original** is clicked.

| | |
|---|---|
| Input | `prompts` (STRING list) — from **Look Sheet Prompt - Custom Shots** |
| Output | `prompts` (STRING list) — the edited prompts while overridden, otherwise the input unchanged |

---

## Requirements

**ComfyUI** v0.44.0 or newer (`p-h3-image Conditioning`).

**Models**, from [PrunaAI/P-H3-Image-Comfy](https://huggingface.co/PrunaAI/P-H3-Image-Comfy)
and [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3):

| File | Folder |
|---|---|
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` (or another FL2VA file) | `models/diffusion_models` |
| `p-h3-image_lora_bf16.safetensors` | `models/loras` |
| `p-h3-image_text_encoder_bf16.safetensors` | `models/text_encoders` |
| `p-h3-image_vae_bf16.safetensors` | `models/vae` |

Plus a vision-language model that **Generate Text** can run (Qwen3-VL, e.g.
`qwen3vl_4b_fp8_scaled.safetensors`) for **Describe Reference**.

**Python**: nothing outside what ComfyUI already ships.

---

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/shisa84/ComfyUI-P-H3ImageLookSheets
```

Restart ComfyUI. The nodes appear under the **P-H3ImageLookSheets**
category. Load `example_workflows/ComfyUI-P-H3ImageLookSheets.json`.

---

## License

MIT
