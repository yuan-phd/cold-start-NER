"""Generate diverse training samples using Google Gemini to address
training data diversity gaps identified by run_diversity.py.

Targets specific weaknesses in the GPT-4o-mini generated data:
- 18% of samples start with "um, hi, this" → diverse openings required
- 96% of NAME entities follow "is" → varied entity positions required
- Formulaic entity contexts → diverse trigger patterns required
- Low sentence structure diversity → varied conversational patterns required

Uses Gemini 2.5 Flash to produce samples from a different LLM family,
reducing same-model distribution overlap with GPT-4o-mini training data.

Depends on: GEMINI_API_KEY environment variable
Output: data/synthetic/train_gemini.json, results/gemini_generation_stats.json

Usage:
    GEMINI_API_KEY=your_key python scripts/run_generate_gemini.py
"""

import json
import os
import re
import time
from pathlib import Path

from src.data.validate import validate_sample
from src.utils.logger import setup_logging, get_logger

log = get_logger(__name__)

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

ENTITY_TYPES = {"NAME", "EMAIL", "CONTRACT_ID", "PRODUCT", "ISSUE_DATE"}

# 10 diverse opening styles — explicitly avoiding "um hi this is" and "thank you for"
OPENING_STYLES = [
    "Start mid-conversation, as if the caller has already been talking",
    "Start with the caller expressing frustration or urgency",
    "Start with the caller asking a question directly, no greeting",
    "Start with the caller giving information immediately without introduction",
    "Start with the caller responding to an agent's question",
    "Start with a very casual/informal opening",
    "Start with the caller explaining a situation before mentioning any entities",
    "Start with the caller referring to a previous call or interaction",
    "Start with the caller mentioning a product or issue before introducing themselves",
    "Start with the caller spelling out or clarifying something",
]

# 10 entity position patterns — explicitly avoiding "name is [NAME]"
ENTITY_POSITIONS = [
    "Place the NAME entity at the very end of the text, not after 'my name is'",
    "Place the NAME entity in the middle of a sentence, not as self-introduction",
    "Mention the name in third person ('my wife Sarah ordered...', 'this is for John...')",
    "Place CONTRACT_ID at the very start of the text",
    "Place EMAIL in the middle of the text, surrounded by other information",
    "Place ISSUE_DATE at the start, before any other entities",
    "Have the caller mention the product name before their own name",
    "Have entities appear in non-standard order (date first, then product, then name last)",
    "Embed the entity inside a subordinate clause, not as a direct statement",
    "Have the entity mentioned as a correction or afterthought at the end",
]

# 10 conversation contexts
CONTEXTS = [
    "The caller is checking on an order they placed for someone else (a gift)",
    "The caller is following up on a previous complaint",
    "The caller wants to modify an existing order, not return it",
    "The caller is comparing two products and asking about one specifically",
    "The caller received the wrong item",
    "The caller is asking about warranty coverage for a specific product",
    "The caller is disputing a charge and referencing specific dates",
    "The caller's account was accessed by someone else",
    "The caller is a business customer ordering multiple items",
    "The caller is elderly or speaks slowly with many pauses",
]

SYSTEM_PROMPT = """You are generating training data for a Named Entity Recognition (NER) system
that processes customer service phone call transcripts. The text should sound like
real spoken language transcribed by a speech-to-text system.

Entity types to include (use 2-4 per sample):
- NAME: Customer names (first, last, or full)
- EMAIL: Email addresses (can be in oral format like "john at gmail dot com")
- CONTRACT_ID: Order/contract/reference numbers (e.g., ORD-2024-5591, SUB-33018)
- PRODUCT: Product names (e.g., "iPhone 15 Pro", "Dyson V12")
- ISSUE_DATE: Dates (absolute like "March 15th" or relative like "about two weeks ago")

CRITICAL RULES:
1. Return ONLY a valid JSON array. No markdown, no backticks, no explanation text.
2. Entity "text" must be an EXACT substring of the transcript text.
3. Entity "start" and "end" must be correct character offsets where text[start:end] == entity_text.
4. Each sample text should be 1-3 sentences, 40-120 words.
5. Make text sound natural and conversational with hesitations, filler words, informal grammar.
6. Vary sentence structure significantly between samples.
7. Do NOT start samples with "um, hi, this is" or "thank you for calling".
8. Do NOT always place NAME after the word "is" — use varied contexts."""


def _make_prompt(batch_idx: int) -> str:
    """Create a generation prompt with diversity constraints."""
    opening = OPENING_STYLES[batch_idx % len(OPENING_STYLES)]
    position = ENTITY_POSITIONS[batch_idx % len(ENTITY_POSITIONS)]
    context = CONTEXTS[batch_idx % len(CONTEXTS)]

    return f"""Generate 3 customer service transcript samples for NER training.

STYLE REQUIREMENTS:
- Opening: {opening}
- Entity placement: {position}
- Context: {context}

Return ONLY a JSON array of 3 objects. No markdown. No code blocks.
Each object: {{"text": "...", "entities": [{{"text": "...", "label": "...", "start": 0, "end": 5}}]}}"""


def call_gemini(prompt: str, api_key: str) -> str | None:
    """Call Gemini API. Returns response text or None on failure."""
    import urllib.request
    import urllib.error

    url = f"{GEMINI_API_URL}?key={api_key}"

    payload = json.dumps({
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.9,
            "maxOutputTokens": 4096,
        },
    })

    req = urllib.request.Request(
        url,
        data=payload.encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
            # Gemini 2.5 may return thinking + response parts
            parts = data["candidates"][0]["content"]["parts"]
            for part in reversed(parts):
                if "text" in part and not part.get("thought", False):
                    return part["text"]
            return parts[-1].get("text", "")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")[:200] if e.fp else ""
        log.error("Gemini API error", status=e.code, body=body)
        return None
    except Exception as e:
        log.error("Gemini API error", error=str(e))
        return None


def parse_response(response_text: str) -> list[dict]:
    """Parse Gemini response into list of sample dicts.

    Only attempts clean parsing. Truncated responses are rejected entirely
    rather than recovered with bracket-appending hacks.
    """
    text = response_text.strip()

    # Strip markdown code blocks
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Attempt 1: Direct parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [s for s in data if isinstance(s, dict) and "text" in s]
        if isinstance(data, dict) and "text" in data:
            return [data]
    except json.JSONDecodeError:
        pass

    # Attempt 2: Find JSON array in text (handles preamble text)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return [s for s in data if isinstance(s, dict) and "text" in s]
        except json.JSONDecodeError:
            pass

    return []


def fix_offsets(sample: dict) -> dict | None:
    """Fix entity offsets by string search. Returns fixed sample or None.

    Handles:
    - Wrong offsets from LLM (fixes via text.find())
    - Case-insensitive fallback for ASR-style text
    - Sequential search to prevent duplicate-entity collision
    - Label validation
    - Overlap detection between fixed entities
    """
    text = sample["text"]
    entities = sample.get("entities", [])

    if not entities:
        return None

    fixed_entities = []
    # Track used spans to prevent overlaps and duplicate-position assignments
    used_spans: list[tuple[int, int]] = []

    for entity in entities:
        entity_text = entity.get("text", "")
        label = entity.get("label", "")

        # Skip invalid labels
        if label not in ENTITY_TYPES:
            continue

        # Skip empty entity text
        if not entity_text or len(entity_text.strip()) == 0:
            continue

        # Check if current offsets are already correct
        start = entity.get("start", -1)
        end = entity.get("end", -1)
        if (0 <= start < len(text)
                and end <= len(text)
                and text[start:end] == entity_text):
            # Offsets are correct — check for overlap with previously fixed entities
            if not _overlaps_any(start, end, used_spans):
                fixed_entities.append({
                    "text": entity_text,
                    "label": label,
                    "start": start,
                    "end": end,
                })
                used_spans.append((start, end))
            continue

        # Find entity text in transcript, searching AFTER all previously used spans
        # to handle duplicate entity text correctly
        search_start = 0
        found = False

        while search_start < len(text):
            idx = text.find(entity_text, search_start)
            if idx == -1:
                break
            candidate_end = idx + len(entity_text)

            if not _overlaps_any(idx, candidate_end, used_spans):
                fixed_entities.append({
                    "text": entity_text,
                    "label": label,
                    "start": idx,
                    "end": candidate_end,
                })
                used_spans.append((idx, candidate_end))
                found = True
                break

            # This position overlaps — try the next occurrence
            search_start = idx + 1

        if found:
            continue

        # Case-insensitive fallback
        text_lower = text.lower()
        entity_lower = entity_text.lower()
        search_start = 0

        while search_start < len(text_lower):
            idx = text_lower.find(entity_lower, search_start)
            if idx == -1:
                break
            candidate_end = idx + len(entity_text)

            if not _overlaps_any(idx, candidate_end, used_spans):
                actual_text = text[idx:candidate_end]
                log.debug(
                    "Case-insensitive fix",
                    original=entity_text,
                    actual=actual_text,
                )
                fixed_entities.append({
                    "text": actual_text,
                    "label": label,
                    "start": idx,
                    "end": candidate_end,
                })
                used_spans.append((idx, candidate_end))
                found = True
                break

            search_start = idx + 1

        # If still not found, skip this entity (don't reject entire sample)

    if not fixed_entities:
        return None

    return {"text": text, "entities": fixed_entities}


def _overlaps_any(
    start: int, end: int, spans: list[tuple[int, int]],
) -> bool:
    """Check if [start, end) overlaps with any existing span."""
    for s, e in spans:
        if start < e and end > s:
            return True
    return False


def main() -> None:
    setup_logging()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        log.error("GEMINI_API_KEY not set.")
        print("Usage: GEMINI_API_KEY=your_key python scripts/run_generate_gemini.py")
        return

    target_samples = 150
    batch_size = 3
    num_batches = target_samples // batch_size

    all_samples: list[dict] = []
    stats = {
        "total_batches": num_batches,
        "successful_batches": 0,
        "failed_batches_parse": 0,
        "failed_batches_api": 0,
        "total_generated": 0,
        "accepted": 0,
        "rejected_no_entities": 0,
        "rejected_validation": 0,
        "rejected_offset_fail": 0,
    }

    log.info("Starting Gemini generation", target=target_samples, batches=num_batches)

    for batch_idx in range(num_batches):
        prompt = _make_prompt(batch_idx)
        log.info(f"Batch {batch_idx + 1}/{num_batches}")

        response = call_gemini(prompt, api_key)
        if response is None:
            stats["failed_batches_api"] += 1
            time.sleep(10)
            continue

        samples = parse_response(response)
        if not samples:
            stats["failed_batches_parse"] += 1
            log.warning(
                f"Batch {batch_idx + 1} parse failed",
                response_preview=response[:80],
            )
            continue

        stats["successful_batches"] += 1

        for sample in samples:
            stats["total_generated"] += 1

            # Fix offsets
            fixed = fix_offsets(sample)
            if fixed is None:
                stats["rejected_no_entities"] += 1
                continue

            # Verify all offsets are correct after fix
            offsets_ok = True
            for e in fixed["entities"]:
                if fixed["text"][e["start"]:e["end"]] != e["text"]:
                    offsets_ok = False
                    break
            if not offsets_ok:
                stats["rejected_offset_fail"] += 1
                continue

            # Validate with project's validator
            is_valid, issues = validate_sample(fixed)
            if not is_valid:
                stats["rejected_validation"] += 1
                continue

            stats["accepted"] += 1
            all_samples.append(fixed)

        # Rate limit for free tier
        time.sleep(6)

    # Save samples
    output_path = Path("data/synthetic/train_gemini.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_samples, f, indent=2)

    # Save stats
    stats["acceptance_rate"] = round(
        stats["accepted"] / max(stats["total_generated"], 1), 4,
    )
    stats_path = Path("results/gemini_generation_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("GEMINI GENERATION RESULTS")
    print("=" * 60)
    print(f"Batches: {stats['successful_batches']}/{num_batches} successful "
          f"({stats['failed_batches_api']} API errors, "
          f"{stats['failed_batches_parse']} parse errors)")
    print(f"Generated: {stats['total_generated']} samples")
    print(f"Accepted: {stats['accepted']} ({stats['acceptance_rate']*100:.1f}%)")
    print(f"Rejected: {stats['rejected_no_entities']} no entities, "
          f"{stats['rejected_offset_fail']} offset fail, "
          f"{stats['rejected_validation']} validation fail")
    print(f"Saved to: {output_path}")

    if all_samples:
        # Diversity check
        openings = set()
        for s in all_samples:
            tokens = s["text"].lower().split()[:3]
            if len(tokens) >= 3:
                openings.add(" ".join(tokens))

        entity_counts: dict[str, int] = {}
        for s in all_samples:
            for e in s.get("entities", []):
                entity_counts[e["label"]] = entity_counts.get(e["label"], 0) + 1

        print("\nDiversity check:")
        print(f"  Unique openings: {len(openings)} / {len(all_samples)} "
              f"({len(openings)/len(all_samples)*100:.1f}%)")
        print(f"  Entity distribution: {entity_counts}")
    print("=" * 60)


if __name__ == "__main__":
    main()
