# ══════════════════════════════════════════════════
# Sainyx image tags: vocabulary, encoding, prompt parsing
# ══════════════════════════════════════════════════
#
# The image model is conditioned on tags (the same style of tags the training
# images carry, e.g. "gogeta", "blue_hair", "super_saiyan"). This module turns
# a free-text prompt into the tags the model actually knows. It has no torch
# dependency, so it is cheap to import and test.

import re
from collections import Counter

# Tags that describe the file or the post rather than what is drawn. They would
# only waste vocabulary slots.
META_TAGS = {
    "highres", "absurdres", "incredibly_absurdres", "lowres", "tagme", "commentary",
    "commentary_request", "english_commentary", "translated", "translation_request",
    "artist_name", "signature", "watermark", "copyright_name", "twitter_username",
    "web_address", "dated", "border", "letterboxed", "official_art", "scan", "jpeg_artifacts",
    "bad_id", "bad_pixiv_id", "pixiv_id", "text_focus", "english_text",
}

# Posts with these tags make poor single-image training samples.
EXCLUDE_TAGS = {
    "comic", "4koma", "monochrome", "greyscale", "sketch", "multiple_views", "doujinshi",
    "manga", "page_number", "character_sheet", "turnaround", "text_focus", "chibi",
}

# Words people type that are not tags themselves. Values are candidate tags;
# only candidates present in the model's vocabulary are used.
ALIASES = {
    "goku": ["son_goku"], "kakarot": ["son_goku"], "kakarotto": ["son_goku"],
    "gohan": ["son_gohan"], "goten": ["son_goten"], "trunks": ["trunks_(dragon_ball)"],
    "buu": ["majin_buu"], "cell": ["cell_(dragon_ball)"], "broly": ["broly_(dragon_ball_super)"],
    "freeza": ["frieza"], "ssj": ["super_saiyan"], "ssj2": ["super_saiyan_2"],
    "ssj3": ["super_saiyan_3"], "ssb": ["super_saiyan_blue", "blue_hair"],
    "ssgss": ["super_saiyan_blue", "blue_hair"], "super saiyan blue": ["super_saiyan_blue", "blue_hair"],
    "blue": ["super_saiyan_blue", "blue_hair"], "gold": ["blonde_hair"], "golden": ["blonde_hair"],
    "yellow hair": ["blonde_hair"], "ultra instinct": ["ultra_instinct"],
    "vegito": ["vegetto", "vegito"], "tien": ["tien_shinhan"], "roshi": ["master_roshi"],
    "saiyan": ["saiyan"], "fusion": ["fusion"], "man": ["1boy"], "boy": ["1boy"],
    "girl": ["1girl"], "woman": ["1girl"], "alone": ["solo"],
}

STOPWORDS = {
    "a", "an", "the", "of", "with", "in", "on", "at", "and", "me", "my", "please", "image",
    "picture", "photo", "drawing", "draw", "generate", "make", "create", "show", "give", "art",
    "anime", "style", "is", "for", "to", "like",
}


def tag_phrase(tag):
    """'trunks_(dragon_ball)' -> 'trunks', 'blue_hair' -> 'blue hair'."""
    base = re.sub(r"_\([^)]*\)", "", tag)
    return base.replace("_", " ").strip().lower()


def build_vocab(tag_lists, character_tags=(), min_count=25, max_tags=512):
    """Most frequent useful tags, ordered by frequency. Character tags that
    appear at least a few times are always kept."""
    counts = Counter(t for tags in tag_lists for t in set(tags) if t not in META_TAGS)
    vocab = [t for t, c in counts.most_common() if c >= min_count][:max_tags]
    for tag in character_tags:
        if counts.get(tag, 0) >= 10 and tag not in vocab:
            vocab.append(tag)
    return vocab


def encode_indices(tags, vocab):
    index = {t: i for i, t in enumerate(vocab)}
    return sorted({index[t] for t in tags if t in index})


def _words(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).split()


def parse_prompt(prompt, vocab):
    """Return (matched_tags, unmatched_words) for a free-text prompt.

    Longest phrase wins, so "super saiyan blue" is read as one thing, not as
    "super saiyan" plus "blue". Words the model has never seen are returned so
    the caller can say so instead of silently ignoring them.
    """
    vocab_set = set(vocab)
    phrases = {}
    for tag in vocab:
        phrases.setdefault(tag_phrase(tag), []).append(tag)
    for phrase, candidates in ALIASES.items():
        present = [c for c in candidates if c in vocab_set]
        if present:
            phrases.setdefault(phrase, [])
            phrases[phrase] = list(dict.fromkeys(phrases[phrase] + present))

    words = _words(prompt)
    matched, unmatched = [], []
    i = 0
    while i < len(words):
        hit = False
        for n in range(min(4, len(words) - i), 0, -1):
            phrase = " ".join(words[i:i + n])
            if phrase in phrases:
                for tag in phrases[phrase]:
                    if tag not in matched:
                        matched.append(tag)
                i += n
                hit = True
                break
        if not hit:
            if words[i] not in STOPWORDS:
                unmatched.append(words[i])
            i += 1
    return matched, unmatched


def resolve_prompt(model, prompt):
    """Prompt -> conditioning info for a loaded image model.

    Returns a dict:
      conditioned   whether the model understands prompts at all
      tags          tags to condition on (may be empty)
      unmatched     words the model has no tag for
      error         a user-facing message when the prompt can't be honoured, else None
    """
    vocab = getattr(model, "tag_vocab", None)
    if not vocab:
        return {"conditioned": False, "tags": [], "unmatched": [], "error": None}
    tags, unmatched = parse_prompt(prompt, vocab)
    if not tags:
        known = list(getattr(model, "character_tags", [])) or vocab[:12]
        names = ", ".join(tag_phrase(t) for t in known[:12])
        return {
            "conditioned": True, "tags": [], "unmatched": unmatched,
            "error": f"Sainyx doesn't know any of those words yet. Try a character such as: {names}.",
        }
    return {"conditioned": True, "tags": tags, "unmatched": unmatched, "error": None}