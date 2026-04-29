"""ASR noise injection engine — simulate speech-to-text transcription errors.

This module provides a configurable noise injection pipeline that simulates
common ASR (Automatic Speech Recognition) transcription errors at multiple
levels: character, word, and utterance. Entity annotation offsets are
automatically updated to remain valid after noise injection.

Each noise function operates independently with its own offset tracker,
and entity spans are updated between operations to maintain correct
protection boundaries.
"""

import random
import re
import string
from copy import deepcopy
from typing import Any

from src.utils.config import load_config, get_seed
from src.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Noise lookup tables — based on common ASR error patterns
# ---------------------------------------------------------------------------

ADJACENT_KEYS: dict[str, str] = {
    "a": "sqwz", "b": "vghn", "c": "xdfv", "d": "sfcer", "e": "wrsdf",
    "f": "dgrtcv", "g": "fhtybn", "h": "gjyunm", "i": "uojkl", "j": "hkunmi",
    "k": "jlomi", "l": "kop", "m": "njk", "n": "bhjm", "o": "iplk",
    "p": "ol", "q": "wa", "r": "edft", "s": "adwxze", "t": "rfgy",
    "u": "yhjki", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu",
    "z": "xas",
}

HOMOPHONES: dict[str, list[str]] = {
    "their": ["there", "they're"], "there": ["their", "they're"],
    "they're": ["their", "there"], "your": ["you're"], "you're": ["your"],
    "to": ["too", "two"], "too": ["to", "two"], "two": ["to", "too"],
    "for": ["four"], "four": ["for"], "hear": ["here"], "here": ["hear"],
    "write": ["right"], "right": ["write"], "know": ["no"], "no": ["know"],
    "by": ["buy", "bye"], "buy": ["by", "bye"], "bye": ["by", "buy"],
    "wear": ["where"], "where": ["wear"], "won": ["one"], "one": ["won"],
    "would": ["wood"], "its": ["it's"], "it's": ["its"],
    "than": ["then"], "then": ["than"],
}

DIGIT_LETTER_CONFUSIONS: dict[str, list[str]] = {
    "0": ["o", "O"], "o": ["0"], "O": ["0"],
    "1": ["l", "i", "I"], "l": ["1"], "i": ["1"], "I": ["1"],
    "5": ["s", "S"], "s": ["5"], "S": ["5"],
    "8": ["b", "B"], "b": ["8"], "B": ["8"],
    "2": ["z", "Z"], "z": ["2"], "Z": ["2"],
}

FILLERS = [
    "uh", "um", "like", "you know", "I mean", "so", "well",
    "right", "mhm", "huh", "hmm", "kind of", "sort of",
]

ASR_WORD_ERRORS: dict[str, list[str]] = {
    "the": ["da", "duh"], "going": ["gonna"], "want to": ["wanna"],
    "got to": ["gotta"], "because": ["cause", "cuz"], "about": ["bout"],
    "probably": ["prolly"], "them": ["em"], "okay": ["ok", "k"],
    "yes": ["yeah", "yep", "yah"],
}


# ---------------------------------------------------------------------------
# Offset tracking
# ---------------------------------------------------------------------------

class OffsetTracker:
    """Track cumulative character offset changes for entity annotation updates."""

    def __init__(self) -> None:
        self._shifts: list[tuple[int, int]] = []

    def record(self, position: int, delta: int) -> None:
        """Record a text mutation at position with delta characters."""
        self._shifts.append((position, delta))

    def apply(self, original_offset: int) -> int:
        """Compute new offset after all recorded mutations."""
        adjusted = original_offset
        for pos, delta in self._shifts:
            if pos <= adjusted:
                adjusted += delta
        return adjusted

    def has_shifts(self) -> bool:
        """Return True if any shifts were recorded."""
        return len(self._shifts) > 0


def _update_spans(
    original_spans: list[tuple[int, int]],
    tracker: OffsetTracker,
) -> list[tuple[int, int]]:
    """Translate entity spans through a tracker's shifts.

    Args:
        original_spans: Current entity (start, end) positions.
        tracker: Tracker from the most recent noise operation.

    Returns:
        Updated spans in the new coordinate space.
    """
    if not tracker.has_shifts():
        return original_spans
    return [(tracker.apply(s), tracker.apply(e)) for s, e in original_spans]


# ---------------------------------------------------------------------------
# Individual noise functions — each creates and returns its own tracker
# ---------------------------------------------------------------------------

def _apply_char_noise(
    text: str,
    entity_spans: list[tuple[int, int]],
    error_rate: float,
    rng: random.Random,
) -> tuple[str, OffsetTracker]:
    """Apply character-level noise. Returns (new_text, tracker)."""
    tracker = OffsetTracker()
    if error_rate <= 0:
        return text, tracker

    protected = set()
    for start, end in entity_spans:
        protected.update(range(start, end))

    result = []
    for i, char in enumerate(text):
        if i in protected or rng.random() > error_rate:
            result.append(char)
            continue

        op = rng.choices(["substitute", "delete", "insert"], weights=[0.5, 0.25, 0.25])[0]

        if op == "substitute":
            if char.lower() in ADJACENT_KEYS:
                replacement = rng.choice(ADJACENT_KEYS[char.lower()])
                if char.isupper():
                    replacement = replacement.upper()
                result.append(replacement)
            elif char in DIGIT_LETTER_CONFUSIONS:
                result.append(rng.choice(DIGIT_LETTER_CONFUSIONS[char]))
            else:
                result.append(char)
        elif op == "delete":
            tracker.record(i, -1)
        elif op == "insert":
            extra = rng.choice(string.ascii_lowercase)
            result.append(extra)
            result.append(char)
            tracker.record(i, +1)

    return "".join(result), tracker


def _apply_word_noise(
    text: str,
    entity_spans: list[tuple[int, int]],
    error_rate: float,
    rng: random.Random,
) -> tuple[str, OffsetTracker]:
    """Apply word-level noise. Returns (new_text, tracker)."""
    tracker = OffsetTracker()
    if error_rate <= 0:
        return text, tracker

    protected = set()
    for start, end in entity_spans:
        protected.update(range(start, end))

    words = []
    for match in re.finditer(r"\S+|\s+", text):
        words.append((match.group(), match.start(), match.end()))

    result = []
    for word_text, w_start, w_end in words:
        if word_text.isspace():
            result.append(word_text)
            continue

        word_positions = set(range(w_start, w_end))
        if word_positions & protected:
            result.append(word_text)
            continue

        if rng.random() > error_rate:
            result.append(word_text)
            continue

        word_lower = word_text.lower().strip(".,!?;:'\"")

        if word_lower in HOMOPHONES:
            replacement = rng.choice(HOMOPHONES[word_lower])
            delta = len(replacement) - len(word_text)
            if delta != 0:
                tracker.record(w_start, delta)
            result.append(replacement)
        elif word_lower in ASR_WORD_ERRORS:
            replacement = rng.choice(ASR_WORD_ERRORS[word_lower])
            delta = len(replacement) - len(word_text)
            if delta != 0:
                tracker.record(w_start, delta)
            result.append(replacement)
        elif len(result) > 0 and rng.random() < 0.3:
            if result and result[-1].isspace():
                removed_space = result.pop()
                tracker.record(w_start - len(removed_space), -len(removed_space))
            result.append(word_text)
        elif len(word_text) > 4 and rng.random() < 0.3:
            split_pos = rng.randint(2, len(word_text) - 2)
            replacement = word_text[:split_pos] + " " + word_text[split_pos:]
            tracker.record(w_start + split_pos, +1)
            result.append(replacement)
        else:
            result.append(word_text)

    return "".join(result), tracker


def _apply_filler_insertion(
    text: str,
    entity_spans: list[tuple[int, int]],
    filler_prob: float,
    rng: random.Random,
) -> tuple[str, OffsetTracker]:
    """Insert filler words. Returns (new_text, tracker)."""
    tracker = OffsetTracker()
    if filler_prob <= 0:
        return text, tracker

    protected = set()
    for start, end in entity_spans:
        protected.update(range(start, end))

    words = text.split(" ")
    result = []
    current_pos = 0

    for word in words:
        word_start = text.find(word, current_pos)
        word_end = word_start + len(word)
        word_positions = set(range(word_start, word_end))

        if not (word_positions & protected) and rng.random() < filler_prob and len(result) > 0:
            filler = rng.choice(FILLERS)
            insertion = filler + " "
            tracker.record(word_start, len(insertion))
            result.append(insertion + word)
        else:
            result.append(word)

        current_pos = word_end

    return " ".join(result), tracker


def _apply_case_corruption(
    text: str,
    entity_spans: list[tuple[int, int]],
    prob: float,
    rng: random.Random,
) -> str:
    """Randomly corrupt case. No length change, no tracker needed."""
    if rng.random() > prob:
        return text

    protected = set()
    for start, end in entity_spans:
        protected.update(range(start, end))

    mode = rng.choices(["lower", "random"], weights=[0.7, 0.3])[0]

    result = list(text)
    for i, char in enumerate(result):
        if i in protected:
            continue
        if mode == "lower":
            result[i] = char.lower()
        elif mode == "random" and char.isalpha():
            if rng.random() < 0.3:
                result[i] = char.swapcase()

    return "".join(result)


def _apply_punctuation_removal(
    text: str,
    entity_spans: list[tuple[int, int]],
    prob: float,
    rng: random.Random,
) -> tuple[str, OffsetTracker]:
    """Remove punctuation. Returns (new_text, tracker)."""
    tracker = OffsetTracker()
    if rng.random() > prob:
        return text, tracker

    protected = set()
    for start, end in entity_spans:
        protected.update(range(start, end))

    punctuation = set(".,!?;:'\"-()[]")
    result = []
    removed = 0

    for i, char in enumerate(text):
        if char in punctuation and i not in protected and rng.random() < 0.7:
            tracker.record(i - removed, -1)
            removed += 1
        else:
            result.append(char)

    return "".join(result), tracker


def _apply_repetition(
    text: str,
    entity_spans: list[tuple[int, int]],
    rng: random.Random,
    prob: float = 0.05,
) -> tuple[str, OffsetTracker]:
    """Simulate word repetition. Returns (new_text, tracker)."""
    tracker = OffsetTracker()
    if prob <= 0:
        return text, tracker

    protected = set()
    for start, end in entity_spans:
        protected.update(range(start, end))

    words = text.split(" ")
    result = []
    current_pos = 0

    for word in words:
        word_start = text.find(word, current_pos)
        word_end = word_start + len(word)
        word_positions = set(range(word_start, word_end))

        result.append(word)

        if not (word_positions & protected) and rng.random() < prob and len(word) > 2:
            repeated = " " + word
            tracker.record(word_end, len(repeated))
            result.append(repeated)

        current_pos = word_end

    return " ".join(result), tracker


# ---------------------------------------------------------------------------
# Main noise injection interface
# ---------------------------------------------------------------------------

def inject_noise(
    sample: dict[str, Any],
    noise_level: str = "moderate",
    config: dict[str, Any] | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Apply ASR noise to a single NER sample with correct offset tracking.

    Each noise operation uses its own tracker, and entity spans are updated
    between operations to maintain correct protection boundaries.

    Args:
        sample: Dict with "text" and "entities" keys.
        noise_level: One of "clean", "mild", "moderate", "severe".
        config: Optional config override.
        rng: Optional random generator for reproducibility.

    Returns:
        New sample dict with noised text and updated entity offsets.
    """
    if config is None:
        config = load_config()
    if rng is None:
        rng = random.Random(get_seed())

    noise_config = config["noise"]["levels"][noise_level]
    sample = deepcopy(sample)

    if noise_level == "clean":
        sample["noise_level"] = "clean"
        return sample

    text = sample["text"]
    entities = sample["entities"]

    # Current entity spans — updated after each operation
    entity_spans = [(e["start"], e["end"]) for e in entities]

    # 1. Case corruption (no length change, no tracker needed)
    text = _apply_case_corruption(
        text, entity_spans, noise_config["case_corruption_prob"], rng
    )

    # 2. Punctuation removal
    text, tracker = _apply_punctuation_removal(
        text, entity_spans, noise_config["punctuation_removal_prob"], rng
    )
    entity_spans = _update_spans(entity_spans, tracker)

    # 3. Word-level noise
    text, tracker = _apply_word_noise(
        text, entity_spans, noise_config["word_error_rate"], rng
    )
    entity_spans = _update_spans(entity_spans, tracker)

    # 4. Character-level noise
    text, tracker = _apply_char_noise(
        text, entity_spans, noise_config["char_error_rate"], rng
    )
    entity_spans = _update_spans(entity_spans, tracker)

    # 5. Filler insertion
    text, tracker = _apply_filler_insertion(
        text, entity_spans, noise_config["filler_prob"], rng
    )
    entity_spans = _update_spans(entity_spans, tracker)

    # 6. Repetition
    text, tracker = _apply_repetition(text, entity_spans, rng)
    entity_spans = _update_spans(entity_spans, tracker)

    # Update entity annotations using final spans
    updated_entities = []
    for ent, (new_start, new_end) in zip(entities, entity_spans):
        # Bounds check
        new_start = max(0, new_start)
        new_end = min(len(text), new_end)

        if new_start >= new_end:
            ent_copy = {**ent, "start": new_start, "end": new_end, "partial": True}
            updated_entities.append(ent_copy)
            continue

        extracted = text[new_start:new_end]

        ent_copy = {
            **ent,
            "start": new_start,
            "end": new_end,
        }

        # Check if entity text is still recoverable
        if extracted.lower().strip() == ent["text"].lower().strip():
            ent_copy["text"] = extracted
        else:
            ent_copy["text"] = extracted
            ent_copy["partial"] = True
            log.debug(
                "Entity partially corrupted",
                original=ent["text"],
                extracted=extracted,
                noise_level=noise_level,
            )

        updated_entities.append(ent_copy)

    sample["text"] = text
    sample["entities"] = updated_entities
    sample["noise_level"] = noise_level
    return sample


def inject_noise_batch(
    samples: list[dict[str, Any]],
    noise_level: str = "moderate",
    config: dict[str, Any] | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Apply noise to a batch of samples."""
    if seed is None:
        seed = get_seed()
    rng = random.Random(seed)

    results = []
    for sample in samples:
        noised = inject_noise(sample, noise_level, config, rng)
        results.append(noised)

    log.info(
        "Batch noise injection complete",
        n_samples=len(results),
        noise_level=noise_level,
    )
    return results


def create_noisy_dataset(
    samples: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Create a multi-noise-level dataset from clean samples."""
    if config is None:
        config = load_config()
    if seed is None:
        seed = get_seed()

    all_samples = []
    noise_levels = list(config["noise"]["levels"].keys())

    for level in noise_levels:
        rng = random.Random(seed)
        for sample in samples:
            noised = inject_noise(sample, level, config, rng)
            all_samples.append(noised)

    log.info(
        "Multi-level noisy dataset created",
        original_samples=len(samples),
        total_samples=len(all_samples),
        levels=noise_levels,
    )
    return all_samples


def apply_oral_format_transforms(
    samples: list[dict[str, Any]],
    transform_ratio: float = 0.3,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Transform a subset of samples to have oral/spoken entity formats."""
    if seed is None:
        seed = get_seed()
    rng = random.Random(seed)

    transforms = {
        "@": " at ",
        ".com": " dot com",
        ".org": " dot org",
        ".net": " dot net",
        ".co": " dot co",
    }

    transformed_count = 0
    result = deepcopy(samples)

    for sample in result:
        email_entities = [
            (i, e) for i, e in enumerate(sample["entities"])
            if e["label"] == "EMAIL" and "@" in e["text"]
        ]

        if not email_entities or rng.random() > transform_ratio:
            continue

        for ent_idx, ent in email_entities:
            old_text = ent["text"]
            new_text = old_text

            for pattern, replacement in transforms.items():
                if pattern in new_text:
                    new_text = new_text.replace(pattern, replacement, 1)

            if new_text == old_text:
                continue

            delta = len(new_text) - len(old_text)
            full_text = sample["text"]
            start = ent["start"]
            end = ent["end"]
            sample["text"] = full_text[:start] + new_text + full_text[end:]

            sample["entities"][ent_idx]["text"] = new_text
            sample["entities"][ent_idx]["end"] = start + len(new_text)

            for other_idx, other_ent in enumerate(sample["entities"]):
                if other_idx == ent_idx:
                    continue
                if other_ent["start"] >= end:
                    other_ent["start"] += delta
                    other_ent["end"] += delta

            transformed_count += 1

    log.info(
        "Email ASR transforms applied",
        total_samples=len(result),
        transformed=transformed_count,
    )

    return result


def apply_email_asr_transforms(
    samples: list[dict[str, Any]],
    transform_ratio: float = 0.3,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Alias for apply_oral_format_transforms for backward compatibility."""
    return apply_oral_format_transforms(samples, transform_ratio, seed)