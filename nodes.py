"""Look sheets for P-H3-Image: one edit per shot of a person dressed in a look.

P-H3-Image (`PH3ImageConditioning`, a MiniMax H3 FL2VA model with Pruna's
P-H3-Image LoRA) edits `image_1` — the person — using the other connected
images as references: `<Picture 1>` is the person photo, the look follows as
`<Picture 2>`, `<Picture 3>`... in connection order. Unlike ref2va there is
no video to cut into shots: every shot is its own still image, so the
prompt node returns one prompt per shot as a list and ComfyUI runs the
conditioning/sampling chain once per entry.

P-H3-Image follows wording much more closely than layout hints: a short
"the outfit from <Picture 2>" barely transfers the garment, while a prompt
that names every item (cut, material, colour, size) does. The descriptions
from `PH3LookSheetsDescribe` are what makes that prompt specific.
"""

from __future__ import annotations

import json
import logging
import re

from comfy_api.latest import io
from comfy_extras.nodes_textgen import TextGenerateV2

try:
    # Only present on builds with the dynamic-VRAM "Comfy model compiler"
    # (needs the comfy_aimdo/comfy_kitchen packages) — absent elsewhere, in
    # which case there's no stale-graph cache to drop in the first place.
    import comfy.model_prefetch as model_prefetch
except ImportError:
    model_prefetch = None

CATEGORY = "P-H3ImageLookSheets"

#: Grammar per gender. "noun" is how the edit names the person ("Keep the
#: woman from <Picture 1>..."), "obj" is the object pronoun ("Dress her...").
_PRONOUNS = {
    "female": {"noun": "woman", "subj": "she", "poss": "her", "obj": "her"},
    "male": {"noun": "man", "subj": "he", "poss": "his", "obj": "him"},
}

#: Only reached when auto-detect ties or finds nothing — never offered as a
#: manual choice, each reference photo holds exactly one person.
_NEUTRAL_PRONOUN = {"noun": "person", "subj": "they", "poss": "their", "obj": "them"}

_AUTO_PRONOUN = "auto"
_PRONOUN_OPTIONS = [_AUTO_PRONOUN, "female", "male"]

_SHE_WORDS = re.compile(r"\b(woman|female|girl|she|her|hers)\b", re.IGNORECASE)
_HE_WORDS = re.compile(r"\b(man|male|boy|he|him|his)\b", re.IGNORECASE)


def _detect_pronoun(description: str) -> str | None:
    """Guess "female"/"male" from a short physical-description sentence.

    Ties or no match return None — a one-sentence VLM description often
    skips gendered words entirely, or a styling word (a "boy-cut" on a
    woman) creates a false tie. The caller falls back to neutral grammar.
    """
    she_hits = len(_SHE_WORDS.findall(description))
    he_hits = len(_HE_WORDS.findall(description))
    if she_hits > he_hits:
        return "female"
    if he_hits > she_hits:
        return "male"
    return None


def _picture_tags(start: int, count: int) -> list[str]:
    """`<Picture start>`, `<Picture start+1>`, ... `count` of them.

    PH3ImageConditioning numbers its images by slot order (image_1 is
    <Picture 1>), so person photo(s) first, then outfit photo(s), in the
    order they were described — the same order Image Aggregator wires them.
    """
    return [f"<Picture {start + i}>" for i in range(count)]


def _join_tags_english(tags: list[str]) -> str:
    if len(tags) == 1:
        return tags[0]
    if len(tags) == 2:
        return f"{tags[0]} and {tags[1]}"
    return ", ".join(tags[:-1]) + f" and {tags[-1]}"


def _parse_multi_description(text: str) -> list[str]:
    """Unwrap PH3LookSheetsDescribe's `{"outfit_0": "...", ...}` JSON into an
    ordered list; anything that isn't a JSON object is one description."""
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            return [str(v) for v in parsed.values()]
    return [text]


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


_BLANK_ENTRY_NAMES = {None: "the outfit shown", "top": "the top shown", "bottom": "the bottom shown",
                      "shoes": "the shoes shown", "accessories": "the accessories shown"}


def _parse_outfit_entries(text: str) -> list[list[tuple[str | None, str]]]:
    """Unwrap PH3LookSheetsDescribe's outfit JSON into one list per picture of
    `(category key, description)` — the key is None for an image described
    as a whole, else "top", "bottom_back", "shoes"... Anything that isn't a
    JSON object is one picture described as a whole.

    A blank description (Describe gave up on it) is kept as "the outfit
    shown" / "the top shown"..., so a picture never disappears and every
    later <Picture N> tag stays in sync with the images wired downstream.
    """
    pictures: list[list[tuple[str | None, str]]] = [[(None, text)]]
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            pictures = [[(k, str(t)) for k, t in v.items()] if isinstance(v, dict) else [(None, str(v))]
                        for v in parsed.values()]
    pictures = [entries for entries in pictures if entries] or [[(None, "")]]
    return [[(key, _lower_first(t.strip().rstrip(".")) or _BLANK_ENTRY_NAMES[_base_category(key)])
             for key, t in entries] for entries in pictures]


def _category_entry_key(cat: str, view: str) -> str:
    """Describe's per-category JSON key: "top", "top_back", "accessories"..."""
    return f"{cat}_back" if view == "back" else cat


def _base_category(key: str | None) -> str | None:
    return key.removesuffix("_back") if key else None


def _debug_block(out_key: str, socket: str, prompt: str, attempts: list[dict]) -> str:
    attempts_text = "\n".join(
        f"  attempt {a['attempt']} (seed {a['seed']}): {a['raw']!r}" for a in attempts
    )
    return f"=== {out_key} ({socket}) ===\n--- prompt ---\n{prompt}\n--- raw output(s) ---\n{attempts_text}"


#: Kept as code, not graph text, so this stays in sync with
#: PH3LookSheetsPrompt's expectations without relying on a workflow to
#: wire the right PrimitiveStringMultiline to the right image.
#: Verbatim wording from the working manual setup — an earlier version added
#: a "not a sensitive or sexual context" reassurance line to cut down on
#: refusals, but that backfired: naming "sexual"/"sensitive" at all, even in
#: a negation, primed the model's safety filter instead of calming it. The
#: refusal retry below is the safety net now, not the wording.
_DESCRIBE_SYSTEM_PROMPTS = {
    "person": (
        "Analyzes the person's apparent gender (man or woman), physical "
        "appearance face, hair, eyes, skin, body shape from the provided "
        "image, ignore outfit and clothes, and ignore the background or "
        "setting entirely. \n"
        "Very short description, only 1 sentence."
    ),
    "outfit": (
        "Analyzes the outfit, only describe the outfit, specifically "
        "detailing the clothing and accessories, skip hair description. "
        "Very short description, only 1 sentence"
    ),
}

#: Outfit conflict resolution: when two outfit photos both happen to show
#: e.g. footwear, the generic "describe the whole outfit" prompt makes each
#: image mention its own pair, and H3's prompt ends up listing two different
#: pairs of shoes for the same subject with no way to tell which to wear.
#: These per-category `..._image_<slot>` combos (shown for target=outfit)
#: let the user assign which connected image is the source for each garment
#: category — for `top`/`bottom`, separately for front and back, since a
#: garment like a dress can look different from behind; for `accessories`,
#: separately for up to two distinct items (glasses in one photo, a bag in
#: another), since unlike a garment's two sides, two accessories can each
#: need their own source photo. Assigning any of them switches every
#: connected image's system prompt to an exclusion-aware one — the assigned
#: image is told to describe *only* its assigned categories (and which
#: slot, if that matters), and every other image is told to skip those same
#: categories — so no two images ever claim the same category. Leaving all
#: of them at "auto" (the default) keeps the original single generic
#: "describe the whole outfit" prompt per image, unchanged. This only feeds
#: the description text itself — which shots need a back view is a
#: separate concern, unrelated to this.
_OUTFIT_CATEGORIES = ["top", "bottom", "shoes", "accessories"]

#: Per category, the source-field slots offered and what they mean for
#: `_outfit_wanted_phrase`'s wording below. "view": slots are the SAME
#: garment shot from different angles (front/back) — only garments whose
#: silhouette can meaningfully differ from behind (a top, a skirt/dress
#: bottom) get this; a back-of-the-shoe detail isn't worth doubling shoes'
#: field count for. "item": slots are DIFFERENT physical items that can
#: each need their own source photo (two accessories, e.g. glasses + a
#: bag) — not a "back view" of the first one. "single": exactly one plain
#: field, no split at all.
_CATEGORY_SLOT_KIND = {
    "top": "view",
    "bottom": "view",
    "shoes": "single",
    "accessories": "item",
}
_CATEGORY_SLOTS = {
    "top": ["front", "back"],
    "bottom": ["front", "back"],
    "shoes": ["front"],  # single dummy slot, never shown in the field name
    "accessories": ["1", "2", "3", "4"],
}
#: Only `accessories` needs a field-name prefix different from its category
#: key (singular "accessory_image_1", not "accessories_image_1") — every
#: other category's field prefix is just its own key.
_CATEGORY_FIELD_PREFIX = {"accessories": "accessory"}


def _outfit_field_id(cat: str, slot: str) -> str:
    """The combo's input id for this category/slot — `{prefix}_image_{slot}`
    for a category with more than one slot, plain `{prefix}_image`
    (slot-less) for a single-slot category."""
    prefix = _CATEGORY_FIELD_PREFIX.get(cat, cat)
    return f"{prefix}_image_{slot}" if len(_CATEGORY_SLOTS[cat]) > 1 else f"{prefix}_image"


_CATEGORY_PHRASES = {
    "top": "top / upper-body garment (shirt, blouse, jacket...)",
    "bottom": "bottom / lower-body garment (pants, skirt...)",
    "shoes": "footwear (shoes, boots)",
    "accessories": "accessories (bag, belt, jewelry, headwear, glasses...)",
}

#: Extra, category-specific instructions appended only when that category is
#: this image's own to describe — a "black boots" caption is ambiguous
#: between an ankle boot and a thigh-high one, and that ambiguity is exactly
#: what leads the final generation to render ordinary shoes with
#: separate hosiery instead of one continuous over-the-knee boot (or vice
#: versa). Not every category needs this; only add an entry where the
#: category can vary in a way plain naming doesn't capture.
_CATEGORY_EXTRA_GUIDANCE = {
    "shoes": (
        "If it extends up the leg, state where it starts and how high it "
        "reaches (ankle, calf, knee, or mid-thigh)."
    ),
    "accessories": (
        "For a hoop, bangle, ring, or chain, describe the thickness of the "
        "material AND its size relative to a visible body part it's worn "
        "against or near (e.g. \"a hoop barely wider than the earlobe, "
        "sitting tight against it\" vs \"a hoop hanging well past the "
        "earlobe, several centimeters wide\"). For a headband, hairpin, "
        "barrette, or other hair accessory, state how far it extends "
        "across the head (e.g. \"a full headband spanning ear to ear\" vs "
        "\"a small clip on one side only, a few centimeters wide\") and "
        "where it sits (right at the hairline vs further back on the "
        "crown). A vague adjective like \"small\" or \"large\" alone, "
        "with nothing to compare it to, is not a usable size and leaves "
        "the actual extent guessed. For glasses/eyewear, describe the "
        "frame shape (round, square, cat-eye...), color/material, and "
        "how thick or thin the frame is. If more than one accessory item "
        "is visible (e.g. glasses worn together with a ring or a pin on "
        "the frame), describe every one of them, not just the most "
        "eye-catching one."
    ),
}


def _outfit_source_options(max_images: int) -> list[str]:
    """Same names as the `images` Autogrow's own sockets (`image_0`,
    `image_1`...) — picking a source this way is a direct match against the
    actual socket you connected, no separate 1-based/0-based numbering to
    keep straight."""
    return ["auto"] + [f"image_{i}" for i in range(max_images)]


def _outfit_wanted_phrase(categories_here: list[tuple[str, str]]) -> str:
    """English phrase for what this image should describe, e.g. "the top /
    upper-body garment (...)" (plain, the common case), "the back of the
    top / upper-body garment (...)" (a "back" view slot specifically), or
    both joined when this image was assigned a mix. Only the "back" slot
    gets special wording — every other slot (a plain single-slot category,
    a "front" view, or one of accessories' "1"/"2" item slots) reads the
    same plain way, since none of those are "the back of" anything.
    Grouped by that plain/back split, not by category, since a single photo
    is realistically all-front-facing or all-back (a camera can't shoot
    both sides of a person at once) even though different categories (or
    different accessory items) can each come from a different photo.
    Deduplicated per bucket: two accessory item slots assigned to the same
    photo both map to category "accessories", which would otherwise repeat
    its phrase twice.
    """
    plain_cats = list(dict.fromkeys(c for c, v in categories_here if v != "back"))
    back_cats = list(dict.fromkeys(c for c, v in categories_here if v == "back"))
    parts = []
    if plain_cats:
        parts.append("the " + _join_tags_english([_CATEGORY_PHRASES[c] for c in plain_cats]))
    if back_cats:
        parts.append("the back of the " + _join_tags_english([_CATEGORY_PHRASES[c] for c in back_cats]))
    return " and ".join(parts)


def _outfit_system_prompt(base_prompt: str, categories_here: list[tuple[str, str]],
                           categories_elsewhere: list[str]) -> str:
    """Build this image's outfit system prompt given the category split.

    `categories_here` is a list of `(category, view)` pairs assigned to
    this specific image; `categories_elsewhere` is just category names
    (view doesn't matter for exclusion — a category assigned to another
    image, front or back, is skipped here either way).

    No split at all (both empty) returns `base_prompt` verbatim — the
    original, single generic outfit description.
    """
    if categories_here:
        wanted = _outfit_wanted_phrase(categories_here)
        cats_only = {c for c, _ in categories_here}
        extra = " ".join(_CATEGORY_EXTRA_GUIDANCE[c] for c in cats_only if c in _CATEGORY_EXTRA_GUIDANCE)
        extra_clause = f" {extra}" if extra else ""
        return (
            f"Describe {wanted} shown in the image. Skip everything else "
            "the person is wearing, and skip hair. If it's ambiguous (e.g. a "
            "thigh-high boot that looks like hosiery), describe it anyway "
            f"rather than saying nothing is visible.{extra_clause} One short "
            "sentence only, no explanation."
        )
    if categories_elsewhere:
        skip = _join_tags_english([_CATEGORY_PHRASES[c] for c in categories_elsewhere])
        base = base_prompt.strip()
        if not base.endswith((".", "!", "?")):
            base += "."
        return (
            f"{base} Skip the {skip} — already described from a different "
            "reference photo; describe only what else this image shows of "
            "the outfit."
        )
    return base_prompt


def _outfit_fallback_prompt(categories_here: list[tuple[str, str]]) -> str:
    """A bare-bones, no-frills version of the scoped prompt — used only when
    the full scoped prompt (with its exclusion/ambiguity clauses) fails
    outright after retries. Deliberately minimal: the more clauses a prompt
    carries, the more this particular captioning model seems to misread one
    of them as a reason to refuse (e.g. "don't mention any other category"
    got read as "don't use category words at all", triggering a refusal on
    the word "skirt"). Still scoped to `categories_here` — never the fully
    generic whole-outfit prompt, which would throw away the category split
    entirely and reintroduce the original conflicting-items problem.
    """
    return f"Describe {_outfit_wanted_phrase(categories_here)} shown in the image, in one short sentence."


#: `person`'s own fallback, mirroring `_outfit_fallback_prompt` above: used
#: only when the full person prompt (`_DESCRIBE_SYSTEM_PROMPTS["person"]`)
#: fails outright after seed retries. Deliberately drops "physical
#: appearance", "skin" and "body shape" — those words (or the emphasis on
#: describing a person's body at all) are the likely refusal trigger, same
#: mechanism already noted above for the outfit prompts' "sexual"/"sensitive"
#: wording. Trades away skin/body-shape detail for actually getting a usable
#: description back.
_PERSON_FALLBACK_PROMPT = (
    "Briefly describe whether this person appears to be a man or a woman, "
    "plus their hair (color and style) and eye color. One short sentence "
    "only, no explanation."
)


#: A genuine one-sentence physical/outfit description never contains any of
#: these — catches refusals ("I can't fulfill this request...", "I apologize,
#: but...", copyright/watermark objections) that empty-string detection alone
#: lets straight through, since refusal text isn't empty. Searched anywhere
#: in the text, not just at the start: a refusal often opens with an apology
#: ("I apologize, but I cannot...") before the actual "I can't" clause.
_REFUSAL_PATTERN = re.compile(
    r"\b(i can'?t|i cannot|i'?m (unable|sorry|not able)|i am (unable|not able)|"
    r"as an ai|i must (decline|respect)|i won'?t|i apologize|unfortunately,? i|"
    r"cannot fulfill|can'?t fulfill|due to copyright|intellectual property|"
    r"copyrighted content)\b",
    re.IGNORECASE,
)


#: A genuine "very short, 1 sentence" description doesn't run this long or
#: span this many paragraphs. A small captioning VLM given a longer,
#: multi-clause system prompt (e.g. the category-scoped outfit prompts,
#: which carry conditional "if X, you MUST..." instructions) can start
#: reasoning out loud through the answer field instead of just answering —
#: narrating the instructions back, working through the image, and only
#: then giving a real sentence buried at the end. That text is neither
#: empty nor a refusal, so it would otherwise sail past `_is_bad_output`
#: and get used verbatim, garbled conclusion and all.
_MAX_REASONABLE_LENGTH = 320


#: A category-scoped prompt ("Describe the accessories... shown in the
#: image.") is sent to a photo the user specifically assigned as *that*
#: category's source — so a flat "there's none of it here" answer (e.g.
#: "The image contains no accessories.") is either a wrong assignment or
#: the model not looking hard enough, not a usable description. Anchored at
#: the start: these models lead with the negative claim when there's
#: genuinely nothing to describe, so matching only the opening avoids
#: flagging a real description that merely mentions an absence in passing
#: (e.g. "A red top and black pants, no accessories worn." — content first,
#: aside second, left alone). Only applied where the caller knows the
#: prompt was scoped (`reject_negative=True` below) — for the generic,
#: whole-outfit prompt, an opening "no accessories" observation could still
#: be one clause of an otherwise valid answer.
_NEGATIVE_ANSWER_PATTERN = re.compile(
    r"^(the image (contains|shows|has) no\b|"
    r"(the person |they('re| are) |subject )?(is|are)n?'?t (visible|shown|present|wearing (any|a)\b)|"
    r"(the person |they |subject )?(is|are) not (visible|shown|present|wearing (any|a)\b)|"
    r"(there (is|are) )?no\b.{0,40}(visible|shown|present)\b|"
    r"(does|do)n?'?t (show|contain|have|feature|wear)|"
    r"(does|do) not (show|contain|have|feature|wear)|"
    r"(none|nothing) (else )?.{0,40}(visible|shown|present)\b|"
    r"(i )?(don'?t|do not) see any)",
    re.IGNORECASE,
)


def _is_negative_answer(text: str) -> bool:
    return bool(_NEGATIVE_ANSWER_PATTERN.match(text.strip()))


def _is_bad_output(text: str, reject_negative: bool = False) -> bool:
    text = text.strip()
    if not text or bool(_REFUSAL_PATTERN.search(text)):
        return True
    if reject_negative and _is_negative_answer(text):
        return True
    return len(text) > _MAX_REASONABLE_LENGTH or text.count("\n") >= 2


#: Small chat-tuned models occasionally echo the chat-template's own role
#: tag ("assistant") as literal generated text instead of stopping right
#: after it — a known quirk, not something specific to this node's prompt.
_LEADING_ROLE_TAG = re.compile(r"^\s*(assistant|system|user)\s*[:\n]+\s*", re.IGNORECASE)


def _strip_role_leak(text: str) -> str:
    return _LEADING_ROLE_TAG.sub("", text, count=1)


#: comfy's dynamic-VRAM text generation (comfy/model_prefetch.py) captures a
#: CUDA graph per transformer layer during a `generate()` call and replays it
#: on the next one as long as the layer's weight-vbar signature still
#: matches — but that check never looks at the KV-cache/embed shapes tied to
#: *this* `generate()` call, only the weights. Describing a second image
#: whose token count differs from the previous one's hits that stale graph:
#: PyTorch replays it against buffers sized for the previous call, which can
#: corrupt memory and crash the whole process ("scatter gather kernel index
#: out of bounds"). We describe several images per node call, so dropping
#: any cached graph before each one forces a fresh capture instead of a
#: mismatched replay.
def _drop_stale_cuda_graphs(clip) -> None:
    if model_prefetch is None:
        return
    for module in clip.cond_stage_model.modules():
        if hasattr(module, "_comfy_graph"):
            model_prefetch._drop_graph(module)
            model_prefetch.GRAPH_WARMED_MODULES.discard(module)


class PH3LookSheetsDescribe(TextGenerateV2):
    """Describe Look Sheet reference photos in one sentence each.

    Wraps core's `TextGenerateV2` (same tokenize/generate/decode call, reused
    via `super().execute()`) with the fixed prompt for whichever reference
    this is — `target=person` for the person photo(s), `target=outfit` for
    the look. Each connected image is described separately, and
    `description` comes back as JSON keyed by target and position,
    `{"outfit_0": "...", "outfit_1": "..."}`, never merged into one
    paragraph. Wire it straight into PH3LookSheetsPrompt, which unwraps it
    into one `<Picture N>` per entry.

    Connect the images in the order they'll reach PH3ImageConditioning —
    person photo(s) first, then outfit photo(s); `images` passes them back
    out in that order for Image Aggregator.

    With target=outfit, per-category combos (`top_image_front`,
    `shoes_image`, `accessory_image_1`...) say which connected image is the
    source of what. All "auto" keeps one generic whole-outfit prompt per
    image; once one is set, each image describes only what it was assigned
    and skips what another image covers, so two photos that both show shoes
    never produce two pairs.
    """

    #: PH3ImageConditioning takes 9 images, one of them always the person —
    #: 8 left for the look.
    MAX_IMAGES = 8

    @classmethod
    def define_schema(cls):
        parent = super().define_schema()
        inputs = []
        for inp in parent.inputs:
            if inp.id == "prompt":
                def _outfit_field_tooltip(cat: str, slot: str) -> str:
                    kind = _CATEGORY_SLOT_KIND[cat]
                    if kind == "view":
                        source = f"shows the {slot} view of the {_CATEGORY_PHRASES[cat]}"
                        note = (" Only set `..._back` when a back-view photo exists and its "
                                "design differs from the front (e.g. a dress with a different "
                                "back).")
                    elif kind == "item":
                        source = f"is source #{slot} for the {_CATEGORY_PHRASES[cat]}"
                        note = (" These are separate items, not views of one — use as many "
                                "of these fields as there are distinct accessories that each "
                                "need their own source photo.")
                    else:
                        source = f"is the source for the {_CATEGORY_PHRASES[cat]}"
                        note = ""
                    return (
                        f"Which connected outfit image {source} — same socket name as in "
                        "`images` above. \"auto\" leaves this unassigned; with several outfit "
                        "images all left at \"auto\", each describes the whole outfit on its "
                        f"own, which can produce two conflicting items.{note}"
                    )

                target_options = [
                    io.DynamicCombo.Option(key="person", inputs=[]),
                    io.DynamicCombo.Option(key="outfit", inputs=[
                        io.Combo.Input(
                            _outfit_field_id(cat, slot),
                            options=_outfit_source_options(cls.MAX_IMAGES),
                            default="auto", optional=True,
                            tooltip=_outfit_field_tooltip(cat, slot),
                        )
                        for cat in _OUTFIT_CATEGORIES
                        for slot in _CATEGORY_SLOTS[cat]
                    ]),
                ]
                inputs.append(io.DynamicCombo.Input(
                    "target", options=target_options, display_name="target",
                    tooltip="Which reference these image(s) are: the person (<Picture 1> "
                    "onward) or the outfit (numbered right after the person's picture(s)). "
                    "Choosing outfit reveals per-category source fields below.",
                ))
            elif inp.id == "images":
                inputs.append(io.Autogrow.Input(
                    "images",
                    tooltip="One photo per socket, described separately. Connect them in the "
                    "order they'll reach p-h3-image Conditioning.",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("image"),
                        prefix="image_", min=1, max=cls.MAX_IMAGES,
                    ),
                ))
            elif inp.id in ("videos", "audios", "system_prompt"):
                # Still photos only, and the prompt is fixed per target.
                continue
            else:
                inputs.append(inp)
        return io.Schema(
            node_id="PH3LookSheetsDescribe",
            display_name="Describe Reference (P-H3 Image Look Sheet)",
            category=CATEGORY,
            inputs=inputs,
            outputs=[
                io.String.Output(display_name="description",
                    tooltip='JSON object, one key per connected image: {"outfit_0": "...", '
                    '"outfit_1": "..."}. Wire straight into Look Sheet Prompt.'),
                io.String.Output(display_name="debug",
                    tooltip="Per connected image: the exact prompt sent to the vision model and "
                    "its raw output for every attempt, retries included."),
                io.Image.Output(display_name="images", is_output_list=True,
                    tooltip="The connected images, in the same order as `description` — wire "
                    "into Image Aggregator."),
            ],
        )

    #: Retried on an empty decode, a refusal, a leaked role tag or (for a
    #: category-scoped prompt) a "there's none here" answer — none of these
    #: is tied to a specific seed, a different one often just works.
    MAX_EMPTY_RETRIES = 3
    _RETRY_SEED_STRIDE = 104729  # an arbitrary large prime, just to jump seeds

    @classmethod
    def _describe_one(cls, clip, prompt, target, max_length, sampling_mode,
                      image, thinking, use_default_template, mtp,
                      reject_negative: bool = False) -> tuple[str, list[dict]]:
        """Returns `(cleaned_text, attempts)`; `attempts` logs the seed and raw
        output of every try for the `debug` output."""
        mode = sampling_mode
        base_seed = mode.get("seed") if isinstance(mode, dict) else None
        attempts: list[dict] = []
        text = ""

        for attempt in range(cls.MAX_EMPTY_RETRIES + 1):
            _drop_stale_cuda_graphs(clip)
            out = super().execute(
                clip, prompt, max_length, mode, thinking=thinking,
                use_default_template=use_default_template, mtp=mtp,
                images={"image1": image},
            )
            raw = out.args[0] if out.args else ""
            attempts.append({"attempt": attempt, "seed": mode.get("seed") if isinstance(mode, dict) else None, "raw": raw})
            text = _strip_role_leak(raw)
            if not _is_bad_output(text, reject_negative):
                return text, attempts
            if base_seed is None or attempt == cls.MAX_EMPTY_RETRIES:
                break
            mode = dict(mode)
            mode["seed"] = (base_seed + (attempt + 1) * cls._RETRY_SEED_STRIDE) % 0xffffffffffffffff
            if not text.strip():
                reason = "empty output"
            elif reject_negative and _is_negative_answer(text):
                reason = "negative (\"nothing here\") answer"
            else:
                reason = "refusal/too verbose"
            logging.warning(
                "[PH3LookSheetsDescribe] %s for target=%s (seed %s), retrying with seed %s "
                "(attempt %d/%d)", reason, target, base_seed, mode["seed"], attempt + 1,
                cls.MAX_EMPTY_RETRIES,
            )

        logging.warning("[PH3LookSheetsDescribe] still bad output for target=%s after "
                        "retries — skipping this image.", target)
        return "", attempts

    @classmethod
    def _describe_scoped(cls, clip, base_prompt, categories_here, categories_elsewhere, target,
                         max_length, sampling_mode, image, thinking, use_default_template,
                         mtp) -> tuple[str, str, list[dict]]:
        """One description with its fallback; returns `(text, prompt, attempts)`."""
        prompt = _outfit_system_prompt(base_prompt, categories_here, categories_elsewhere) \
            if target == "outfit" else base_prompt
        text, attempts = cls._describe_one(
            clip, prompt, target, max_length, sampling_mode, image, thinking,
            use_default_template, mtp, reject_negative=bool(categories_here),
        )
        # The scoped prompt's own clauses can make the vision model refuse
        # even after seed retries; one more try with a bare version of the
        # same scoped prompt — never the generic whole-outfit one, which
        # would bring back the conflicting-items problem.
        if not text and categories_here:
            text, fallback_attempts = cls._describe_one(
                clip, _outfit_fallback_prompt(categories_here), target, max_length,
                sampling_mode, image, thinking, use_default_template, mtp,
                reject_negative=True,
            )
            attempts += fallback_attempts
        # Same for `person`: its "physical appearance... body shape" wording
        # is itself the likely refusal trigger, so retry with softer wording.
        elif not text and target == "person":
            text, fallback_attempts = cls._describe_one(
                clip, _PERSON_FALLBACK_PROMPT, target, max_length,
                sampling_mode, image, thinking, use_default_template, mtp,
            )
            attempts += fallback_attempts
        return text, prompt, attempts

    @classmethod
    def execute(cls, clip, target, max_length, sampling_mode, images,
                thinking=False, use_default_template=True, mtp="auto") -> io.NodeOutput:
        target_key = target.get("target") if isinstance(target, dict) else target
        base_prompt = _DESCRIBE_SYSTEM_PROMPTS[target_key]
        ordered = [(key, img) for key, img in (images or {}).items() if img is not None]
        if not ordered:
            raise ValueError("PH3LookSheetsDescribe needs at least one image")

        # cat -> {slot: socket}, keyed by the exact `images` socket name the
        # combos list, so a choice there matches a connected socket directly.
        category_assignment: dict[str, dict[str, str]] = {}
        if target_key == "outfit" and isinstance(target, dict):
            for cat in _OUTFIT_CATEGORIES:
                for slot in _CATEGORY_SLOTS[cat]:
                    choice = target.get(_outfit_field_id(cat, slot), "auto")
                    if choice and choice != "auto":
                        category_assignment.setdefault(cat, {})[slot] = choice
        # A combo pointing at a socket that's no longer connected is a
        # routine thing to hit while iterating on a workflow — ignore that
        # one assignment rather than failing the node.
        connected_keys = {key for key, _ in ordered}
        for cat in list(category_assignment.keys()):
            views = category_assignment[cat]
            for view in list(views.keys()):
                if views[view] not in connected_keys:
                    logging.warning(
                        "[PH3LookSheetsDescribe] %s is set to \"%s\", but that socket isn't "
                        "connected (connected: %s) — ignoring this assignment.",
                        _outfit_field_id(cat, view), views[view], sorted(connected_keys),
                    )
                    del views[view]
            if not views:
                del category_assignment[cat]

        # Keyed by position among connected images, not the socket id, so a
        # gap in the sockets doesn't leave a gap in the <Picture N> numbering.
        # An image with no category assigned gets one string; an image with
        # categories gets one description per category ({"top": ..., "bottom":
        # ...}), each from its own call, so the prompt node can leave out what
        # a shot's framing doesn't show. A failed call keeps its key with an
        # empty string.
        descriptions = {}
        debug_blocks = []
        for position, (key, image) in enumerate(ordered):
            categories_here = [(cat, view) for cat, views in category_assignment.items()
                               for view, socket in views.items() if socket == key]
            categories_elsewhere = sorted({
                cat for cat, views in category_assignment.items()
                if views and not any(socket == key for socket in views.values())
            })
            out_key = f"{target_key}_{position}"
            if target_key == "outfit" and categories_here:
                by_entry: dict[str, list[tuple[str, str]]] = {}
                for cat, view in categories_here:
                    by_entry.setdefault(_category_entry_key(cat, view), []).append((cat, view))
                entries = {}
                for entry_key, scoped in by_entry.items():
                    text, prompt, attempts = cls._describe_scoped(
                        clip, base_prompt, scoped, [], target_key, max_length, sampling_mode,
                        image, thinking, use_default_template, mtp)
                    entries[entry_key] = text
                    debug_blocks.append(_debug_block(f"{out_key}.{entry_key}", key, prompt, attempts))
                descriptions[out_key] = entries
            else:
                text, prompt, attempts = cls._describe_scoped(
                    clip, base_prompt, [], categories_elsewhere, target_key, max_length, sampling_mode,
                    image, thinking, use_default_template, mtp)
                descriptions[out_key] = text
                debug_blocks.append(_debug_block(out_key, key, prompt, attempts))

        if not any(descriptions.values()):
            logging.warning("[PH3LookSheetsDescribe] every image failed for target=%s — "
                            "returning empty descriptions.", target_key)
        image_list = [image for _, image in ordered]
        return io.NodeOutput(json.dumps(descriptions, ensure_ascii=False), "\n\n".join(debug_blocks), image_list)


def _flatten_images(value):
    """One aggregator input as a flat list of images: a single image, a list
    (Describe's `images` output), an unconnected socket (`None`), or nested."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_flatten_images(item))
        return out
    return [value]


#: PH3ImageConditioning's own image slots: image_1 ... image_9.
_MAX_PH3_IMAGES = 9


class PH3LookSheetsImageAggregator(io.ComfyNode):
    """Route the person and outfit images onto PH3ImageConditioning's
    `image_1` ... `image_9` slots, in order: person first, then outfit, then
    the two extras.

    `image_1` is the image P-H3-Image edits — the first person photo. Unused
    slots come out as `None`, which PH3ImageConditioning skips like a
    disconnected socket, so all 9 can always be wired.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PH3LookSheetsImageAggregator",
            display_name="Image Aggregator (P-H3 Image Look Sheet)",
            category=CATEGORY,
            is_input_list=True,
            inputs=[
                io.Image.Input("images_person", optional=True,
                    tooltip="Describe Reference's `images` output with target=person."),
                io.Image.Input("images_outfit", optional=True,
                    tooltip="Describe Reference's `images` output with target=outfit."),
                io.Image.Input("images_extra_1", optional=True,
                    tooltip="Any additional image or image list, placed after images_outfit."),
                io.Image.Input("images_extra_2", optional=True,
                    tooltip="Any additional image or image list, placed last."),
            ],
            outputs=[
                io.Image.Output(id=f"image_{i}", display_name=f"image_{i}",
                    tooltip=f"Image {i} across every connected source, in order — wire into "
                    f"p-h3-image Conditioning's image_{i}.")
                for i in range(1, _MAX_PH3_IMAGES + 1)
            ] + [
                io.Image.Output(id="images", display_name="images", is_output_list=True,
                    tooltip="The same images as one list, no trailing empty slots."),
            ],
        )

    @classmethod
    def execute(cls, images_person=None, images_outfit=None,
                images_extra_1=None, images_extra_2=None) -> io.NodeOutput:
        # This fixed source order is what lines image_N up with <Picture N>
        # in the prompt.
        all_images = []
        for group in (images_person, images_outfit, images_extra_1, images_extra_2):
            all_images.extend(_flatten_images(group))

        if len(all_images) > _MAX_PH3_IMAGES:
            logging.warning("[PH3LookSheetsImageAggregator] %d images connected, only the first "
                            "%d are kept.", len(all_images), _MAX_PH3_IMAGES)
            all_images = all_images[:_MAX_PH3_IMAGES]

        slots = all_images + [None] * (_MAX_PH3_IMAGES - len(all_images))
        return io.NodeOutput(*slots, all_images)


# --------------------------------------------------------------------------
# Shot Config -> Look Sheet Prompt
#
# A ComfyUI Autogrow socket can't carry two combos per shot, so Shot Config
# keeps shot_description/expression as ordinary widgets and packs them into
# one string — one Autogrow socket per shot on the prompt node.
# --------------------------------------------------------------------------

#: Each shot is one literal, physical crop boundary rather than photography
#: jargon like "medium shot", which composed unreliably in testing. Regions
#: without the head state both frame edges ("top of frame is the waist,
#: bottom of frame is just below the feet"): a looser "framed from the
#: stomach to the feet" still came out full-body.
#:
#: H3LookSheets' Chest, Upper Legs and Lower Legs regions are left out: on a
#: 3:4 frame P-H3-Image renders Chest like Upper Body without the head, and
#: Upper/Lower Legs like Lower Body, whatever the wording.
_SHOT_DESCRIPTION_PHRASES = {
    "Full Body - Front": "a full-body shot, front view, facing the camera, from head to toe",
    "Full Body - Front 3/4 Left": "a full-body shot, turned three-quarters to the left, from head to toe",
    "Full Body - Front 3/4 Right": "a full-body shot, turned three-quarters to the right, from head to toe",
    "Full Body - Left Profile": "a full-body shot, left profile, from head to toe",
    "Full Body - Right Profile": "a full-body shot, right profile, from head to toe",
    "Full Body - Back 3/4 Left": "a full-body shot, mostly turned away, seen three-quarters from the back on the left side, from head to toe",
    "Full Body - Back 3/4 Right": "a full-body shot, mostly turned away, seen three-quarters from the back on the right side, from head to toe",
    "Full Body - Back": "a full-body shot, back view, from head to toe",
    "Upper Body - Front": "front view, facing the camera, framed from waist to top of head",
    "Upper Body - Front 3/4 Left": "turned three-quarters to the left, framed from waist to top of head",
    "Upper Body - Front 3/4 Right": "turned three-quarters to the right, framed from waist to top of head",
    "Upper Body - Left Profile": "left profile, framed from waist to top of head",
    "Upper Body - Right Profile": "right profile, framed from waist to top of head",
    "Upper Body - Back 3/4 Left": "mostly turned away, seen three-quarters from the back on the left side, framed from waist to top of head",
    "Upper Body - Back 3/4 Right": "mostly turned away, seen three-quarters from the back on the right side, framed from waist to top of head",
    "Upper Body - Back": "back view, framed from waist to top of head",
    "Lower Body - Front": "front view, facing the camera, a close-up crop, top of frame is the waist, bottom of frame is just below the feet",
    "Lower Body - Front 3/4 Left": "turned three-quarters to the left, a close-up crop, top of frame is the waist, bottom of frame is just below the feet",
    "Lower Body - Front 3/4 Right": "turned three-quarters to the right, a close-up crop, top of frame is the waist, bottom of frame is just below the feet",
    "Lower Body - Left Profile": "left profile, a close-up crop, top of frame is the waist, bottom of frame is just below the feet",
    "Lower Body - Right Profile": "right profile, a close-up crop, top of frame is the waist, bottom of frame is just below the feet",
    "Lower Body - Back 3/4 Left": "mostly turned away, seen three-quarters from the back on the left side, a close-up crop, top of frame is the waist, bottom of frame is just below the feet",
    "Lower Body - Back 3/4 Right": "mostly turned away, seen three-quarters from the back on the right side, a close-up crop, top of frame is the waist, bottom of frame is just below the feet",
    "Lower Body - Back": "back view, a close-up crop, top of frame is the waist, bottom of frame is just below the feet",
    "Waist - Front": "front view, facing the camera, a close-up crop, top of frame is the chest, bottom of frame is the upper thighs, centered on the waist",
    "Waist - Front 3/4 Left": "turned three-quarters to the left, a close-up crop, top of frame is the chest, bottom of frame is the upper thighs, centered on the waist",
    "Waist - Front 3/4 Right": "turned three-quarters to the right, a close-up crop, top of frame is the chest, bottom of frame is the upper thighs, centered on the waist",
    "Waist - Left Profile": "left profile, a close-up crop, top of frame is the chest, bottom of frame is the upper thighs, centered on the waist",
    "Waist - Right Profile": "right profile, a close-up crop, top of frame is the chest, bottom of frame is the upper thighs, centered on the waist",
    "Waist - Back 3/4 Left": "mostly turned away, seen three-quarters from the back on the left side, a close-up crop, top of frame is the chest, bottom of frame is the upper thighs, centered on the waist",
    "Waist - Back 3/4 Right": "mostly turned away, seen three-quarters from the back on the right side, a close-up crop, top of frame is the chest, bottom of frame is the upper thighs, centered on the waist",
    "Waist - Back": "back view, a close-up crop, top of frame is the chest, bottom of frame is the upper thighs, centered on the waist",
    "Head - Front": "front view, facing the camera, close framing on the head",
    "Head - Front 3/4 Left": "turned three-quarters to the left, close framing on the head",
    "Head - Front 3/4 Right": "turned three-quarters to the right, close framing on the head",
    "Head - Left Profile": "left profile, close framing on the head",
    "Head - Right Profile": "right profile, close framing on the head",
    "Head - Back 3/4 Left": "mostly turned away, seen three-quarters from the back on the left side, close framing on the head",
    "Head - Back 3/4 Right": "mostly turned away, seen three-quarters from the back on the right side, close framing on the head",
    "Head - Back": "back view, close framing on the head",
}
_SHOT_DESCRIPTION_OPTIONS = list(_SHOT_DESCRIPTION_PHRASES.keys())

#: Regions whose frame includes the whole head. P-H3-Image keeps the
#: framing of <Picture 1> and crops the top of the head when a shot shows
#: more of the body than the person photo does, unless the prompt states
#: that the whole head stays inside the frame. Every other region gets a
#: prompt that never mentions the face or hair: any wording about them
#: pulls the head back into frame and the crop widens to a full-body shot.
_HEAD_IN_FRAME_REGIONS = {"Full Body", "Upper Body", "Head"}

#: `expression` is only written when the face is actually in frame.
_FACE_VISIBLE_ORIENTATIONS = {"Front", "Front 3/4 Left", "Front 3/4 Right"}

_EXPRESSION_PHRASES = {
    "neutral": "neutral",
    "happy": "happy",
    "smiling": "warmly smiling",
    "sad": "sad",
    "angry": "angry",
    "surprised": "surprised, eyes wide",
    "scared": "frightened, eyes wide and brows raised",
    "disgusted": "disgusted",
    "shy/embarrassed": "shy, embarrassed",
    "confident": "confident",
    "serious": "serious",
    "laughing": "laughing, genuinely amused",
    "crying": "tearful",
    "smirking": "playfully smirking",
    "confused": "confused",
}
_EXPRESSION_OPTIONS = list(_EXPRESSION_PHRASES.keys())

#: None of the option strings above contain it, so splitting is unambiguous.
_SHOT_CONFIG_SEP = "||"

_MAX_CUSTOM_SHOTS = 15

#: Which outfit categories each region's prompt names. P-H3-Image pulls into
#: frame whatever body part a named item is worn on — shoes named in a
#: close crop of the torso bring the legs back and the crop widens — but it
#: also leaves off whatever isn't named: a torso crop naming only the top
#: came out with no skirt at all. So only items worn far from the frame are
#: left out (shoes above the hips, head accessories below the shoulders);
#: top and bottom meet at the waist and are named together. A picture
#: described as a whole (no category assigned in Describe Reference) is
#: always named.
_REGION_CATEGORIES = {
    "Full Body": {"top", "bottom", "shoes", "accessories"},
    "Upper Body": {"top", "bottom", "accessories"},
    "Lower Body": {"top", "bottom", "shoes"},
    "Waist": {"top", "bottom"},
    "Head": {"top", "accessories"},
}


def _shot_outfit_items(pictures, tags, region: str, back_view: bool) -> list[tuple[str, str]]:
    """`(description, <Picture N>)` pairs a shot names, in picture order.

    A back view uses a category's `..._back` description when one exists
    (from any picture) instead of its front one; a front view never uses a
    back description. Falls back to every front description when the
    region's filter leaves nothing to name.
    """
    visible = _REGION_CATEGORIES.get(region, _REGION_CATEGORIES["Full Body"])
    with_back = {_base_category(key) for entries in pictures for key, _ in entries if key and key.endswith("_back")}
    items, front_items = [], []
    for entries, tag in zip(pictures, tags):
        for key, text in entries:
            is_back = bool(key) and key.endswith("_back")
            if not is_back:
                front_items.append((text, tag))
            if key is None:
                items.append((text, tag))
            elif _base_category(key) in visible:
                use_back = back_view and _base_category(key) in with_back
                if is_back == use_back:
                    items.append((text, tag))
    return items or front_items

_DEFAULT_BACKDROP = "plain light neutral grey studio backdrop"

#: The person description is spliced in as "Keep the woman from <Picture 1>
#: ...: <description>", so the vision model's usual "The person appears to
#: be a woman with..." opening is cut down to "a woman with...".
_PERSON_HEDGE = re.compile(r"^(the |this )?(person|subject|individual)( in the image)? (appears|seems) to be\s+", re.IGNORECASE)


class PH3LookSheetsShotConfig(io.ComfyNode):
    """One shot's framing and expression, packed into one string for a
    `shot_N` socket on Look Sheet Prompt."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PH3LookSheetsShotConfig",
            display_name="Shot Config (P-H3 Image Look Sheet)",
            category=CATEGORY,
            inputs=[
                io.Combo.Input("shot_description", options=_SHOT_DESCRIPTION_OPTIONS),
                io.Combo.Input("expression", options=_EXPRESSION_OPTIONS),
            ],
            outputs=[io.String.Output(display_name="shot_config")],
        )

    @classmethod
    def execute(cls, shot_description, expression) -> io.NodeOutput:
        return io.NodeOutput(_SHOT_CONFIG_SEP.join((shot_description, expression)))


class PH3LookSheetsPrompt(io.ComfyNode):
    """Write one P-H3-Image edit prompt per shot.

    `person_description`/`outfit_description` take plain text or Describe
    Reference's JSON — each JSON entry becomes its own `<Picture N>`,
    person photo(s) first, then outfit photo(s), matching Image
    Aggregator's slot order. Every connected `shot_N` (fed by Shot Config)
    is one prompt in the `prompts` list; wired into PH3ImageConditioning's
    `prompt`, ComfyUI runs the conditioning and sampling once per entry.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PH3LookSheetsPrompt",
            display_name="Look Sheet Prompt - Custom Shots (P-H3 Image)",
            category=CATEGORY,
            inputs=[
                io.String.Input("person_description", multiline=True, default=""),
                io.String.Input("outfit_description", multiline=True, default=""),
                io.Combo.Input("person_gender", options=_PRONOUN_OPTIONS, optional=True),
                io.String.Input("backdrop", default=_DEFAULT_BACKDROP, multiline=True, optional=True),
                io.Autogrow.Input(
                    "shots",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.String.Input("shot"),
                        prefix="shot_", min=1, max=_MAX_CUSTOM_SHOTS,
                    ),
                ),
            ],
            outputs=[
                io.String.Output(display_name="prompts", is_output_list=True,
                    tooltip="One edit prompt per connected shot, in shot order."),
                io.Int.Output(display_name="shot_count"),
            ],
        )

    @classmethod
    def execute(cls, person_description, outfit_description, shots,
                person_gender=_AUTO_PRONOUN, backdrop=_DEFAULT_BACKDROP) -> io.NodeOutput:
        # Lowercased: each description is spliced mid-sentence after a colon.
        person_items = [_lower_first(_PERSON_HEDGE.sub("", t.strip()).rstrip("."))
                        for t in _parse_multi_description(person_description) if t.strip()]
        outfit_pictures = _parse_outfit_entries(outfit_description)

        resolved = _detect_pronoun(person_items[0] if person_items else "") \
            if person_gender == _AUTO_PRONOUN else person_gender
        p = _PRONOUNS.get(resolved, _NEUTRAL_PRONOUN)
        set_dressing = backdrop.strip().rstrip(".") or _DEFAULT_BACKDROP

        person_count = max(1, len(person_items))
        person_ref = _join_tags_english(_picture_tags(1, person_count))
        outfit_tags = _picture_tags(1 + person_count, len(outfit_pictures))
        person_clause = f": {person_items[0]}" if person_items else ""

        parsed: list[tuple[str, str]] = []
        for value in (shots or {}).values():
            parts = value.split(_SHOT_CONFIG_SEP) if value else []
            if len(parts) == 2:
                parsed.append((parts[0], parts[1]))
        if not parsed:
            raise ValueError("PH3LookSheetsPrompt needs at least one Shot Config connected")

        pose = f"{p['subj'].capitalize()} stands upright, still, arms relaxed at {p['poss']} sides"
        ending = (f"Background: {set_dressing}, bright even studio lighting. "
                  "No text, watermark, logo or writing anywhere in the image.")

        # Every prompt must start with "Edit <Picture 1>": PH3ImageConditioning
        # then encodes it as is instead of prefixing its own "Edit <Picture 1>. ".
        prompts = []
        for shot_description, expression in parsed:
            region, _, orientation = shot_description.rpartition(" - ")
            framing = _SHOT_DESCRIPTION_PHRASES.get(shot_description, shot_description)
            items = _shot_outfit_items(outfit_pictures, outfit_tags, region, orientation.startswith("Back"))
            outfit_ref = _join_tags_english(list(dict.fromkeys(tag for _, tag in items)))
            # Each item named with its own picture: P-H3-Image copies what the
            # words describe, a bare "the outfit from <Picture 2>" barely
            # transfers anything.
            outfit_fragment = "; ".join(f"{text} ({tag})" for text, tag in items)
            outfit_only = (f"Only the clothing and accessories come from {outfit_ref}; the hair, body, "
                           "pose and camera angle of whoever or whatever wears them there are not kept.")
            if region in _HEAD_IN_FRAME_REGIONS:
                shot_pose = pose
                if orientation in _FACE_VISIBLE_ORIENTATIONS:
                    shot_pose += f", with a {_EXPRESSION_PHRASES.get(expression, expression)} expression"
                prompts.append(
                    f"Edit <Picture 1>. Keep the {p['noun']} from {person_ref} with exactly the same "
                    f"face, hair, hairstyle, skin and body proportions{person_clause}. "
                    f"Dress {p['obj']} in the outfit and accessories from {outfit_ref}: {outfit_fragment}. "
                    f"{outfit_only} "
                    f"Camera: {framing}, {p['poss']} whole head and the top of {p['poss']} hair fully "
                    f"inside the frame. {shot_pose}. {ending}"
                )
            else:
                # Framing first, so it isn't outweighed by the outfit text.
                prompts.append(
                    f"Edit <Picture 1>. Camera: {framing}. The head and face are outside the frame, "
                    "cut off above the top edge of the image. "
                    f"Show the {p['noun']} from {person_ref} with the same skin tone and body "
                    f"proportions, dressed in the outfit and accessories from {outfit_ref}: "
                    f"{outfit_fragment}. {outfit_only} {pose}. {ending}"
                )
        return io.NodeOutput(prompts, len(prompts))


#: Rendered by web/ph3looksheets_prompt_view.js; the value is JSON:
#: {"override": bool, "text": str}. `text` is only read while `override` is
#: true — the editor seeds it from the latest prompts the first time it's on.
PromptViewState = io.Custom("PH3LOOKSHEETS_PROMPT_VIEW")

#: One header line per shot in the joined view text; the override splits on it.
_SHOT_HEADER = "===== Shot {} ====="
_SHOT_HEADER_RE = re.compile(r"^===== Shot \d+ =====[ \t]*$", re.MULTILINE)


def _join_prompts(prompts: list[str]) -> str:
    return "\n\n".join(f"{_SHOT_HEADER.format(i + 1)}\n{text}" for i, text in enumerate(prompts))


def _split_prompts(text: str) -> list[str]:
    parts = [part.strip() for part in _SHOT_HEADER_RE.split(text)]
    return [part for part in parts if part]


class PH3LookSheetsPromptView(io.ComfyNode):
    """Colorized preview of the per-shot prompts, with a copy button and an
    edit-a-copy override.

    Takes the whole `prompts` list at once and shows it as one text, one
    `===== Shot N =====` header per prompt. While overridden, the edited text
    is split back on those headers, so each shot can be edited in place;
    adding or removing a header adds or removes a shot.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PH3LookSheetsPromptView",
            display_name="Prompt View (P-H3 Image Look Sheet)",
            category=CATEGORY,
            is_input_list=True,
            inputs=[
                io.String.Input("prompts", force_input=True),
                PromptViewState.Input("view_state"),
            ],
            outputs=[io.String.Output(display_name="prompts", is_output_list=True)],
        )

    @classmethod
    def execute(cls, prompts, view_state=None) -> io.NodeOutput:
        state_text = view_state[0] if view_state else ""
        state = json.loads(state_text) if state_text else {}
        joined = _join_prompts(prompts)
        out = _split_prompts(state.get("text", "")) if state.get("override") else list(prompts)
        # Echoed back so the widget can refresh its read-only mirror.
        return io.NodeOutput(out or list(prompts), ui={"ph3looksheets_prompt": [joined]})


NODE_CLASS_MAPPINGS = {
    "PH3LookSheetsDescribe": PH3LookSheetsDescribe,
    "PH3LookSheetsImageAggregator": PH3LookSheetsImageAggregator,
    "PH3LookSheetsShotConfig": PH3LookSheetsShotConfig,
    "PH3LookSheetsPrompt": PH3LookSheetsPrompt,
    "PH3LookSheetsPromptView": PH3LookSheetsPromptView,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PH3LookSheetsDescribe": "Describe Reference (P-H3 Image Look Sheet)",
    "PH3LookSheetsImageAggregator": "Image Aggregator (P-H3 Image Look Sheet)",
    "PH3LookSheetsShotConfig": "Shot Config (P-H3 Image Look Sheet)",
    "PH3LookSheetsPrompt": "Look Sheet Prompt - Custom Shots (P-H3 Image)",
    "PH3LookSheetsPromptView": "Prompt View (P-H3 Image Look Sheet)",
}
