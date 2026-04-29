"""Hybrid NER — route by entity type: rules for structured, model for semantic.

Design: Rules handle CONTRACT_ID only (rigid alphanumeric patterns where regex
achieves perfect F1). The transformer model handles NAME, PRODUCT, ISSUE_DATE,
and EMAIL (all require semantic understanding or handle format variants that
regex captures poorly).

This design was refined through evaluation: the initial assumption that rules
should own all structured entities (EMAIL + CONTRACT_ID) was disproven by data.
Rules achieved F1=1.000 on CONTRACT_ID but only 0.636 on EMAIL, while the model
achieved 0.870 on EMAIL. The difference is structural: CONTRACT_IDs have rigid
patterns regex captures perfectly, while EMAILs have oral format variants
("at", "dot com") the model learned from training data.

Similarly, ISSUE_DATE was initially in RULES_FALLBACK_TYPES but the rules date
fallback added false positives that hurt ensemble performance (ensemble F1=0.611
vs model-alone F1=0.909). The fallback was removed based on this evidence.

NAME and PRODUCT have no fallback — the model is the sole source.
NAME is the hardest semantic entity, especially in noisy ASR text.
"""

from typing import Any

from src.models import rules
from src.utils.logger import get_logger

log = get_logger(__name__)

# Explicit ownership assignment
RULES_OWNED_TYPES = {"CONTRACT_ID"}
MODEL_OWNED_TYPES = {"NAME", "PRODUCT", "ISSUE_DATE", "EMAIL"}

# No reliable rules fallback for any model-owned type:
# - ISSUE_DATE: removed because date regex fallback added false positives (ensemble F1=0.611 vs model F1=0.909)
# - NAME/PRODUCT: never had rules coverage
# - EMAIL: moved to model ownership (model F1=0.870 vs rules F1=0.636)
RULES_FALLBACK_TYPES: set[str] = set()


def _spans_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Check if two entity spans overlap."""
    return a["start"] < b["end"] and b["start"] < a["end"]


def predict(
    text: str,
    model: Any,
    confidence_threshold: float = 0.3,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Extract entities using type-routed hybrid approach.

    Args:
        text: Input text string.
        model: A model object with a .predict(text) method.
        confidence_threshold: Minimum confidence for model predictions.

    Returns:
        Tuple of (entity list, conflict stats dict).
    """
    rule_entities = rules.predict(text)
    model_entities = model.predict(text)
    model_entities = [e for e in model_entities if e["confidence"] >= confidence_threshold]
    # Filter single-character entities from model output only — no valid entity
    # type is 1 char. Rules are excluded because they produce structurally
    # validated spans that never contain single-character artifacts.
    model_entities = [e for e in model_entities if len(e.get("text", "")) >= 2]

    # --- Primary routing by ownership ---
    from_rules = [e for e in rule_entities if e["label"] in RULES_OWNED_TYPES]
    from_model = [e for e in model_entities if e["label"] in MODEL_OWNED_TYPES]

    # --- Detect conflicts ---
    conflicts = 0
    for r_ent in rule_entities:
        for m_ent in model_entities:
            if _spans_overlap(r_ent, m_ent):
                conflicts += 1
                log.debug(
                    "Ensemble conflict",
                    rules=f"{r_ent['label']}:'{r_ent['text']}'",
                    model=f"{m_ent['label']}:'{m_ent['text']}'",
                )

    # --- Fallback 1: Model catches EMAIL/CONTRACT_ID that rules missed ---
    rules_spans = [(e["start"], e["end"]) for e in from_rules]
    model_fallback_count = 0
    for m_ent in model_entities:
        if m_ent["label"] in RULES_OWNED_TYPES:
            if not any(_spans_overlap(m_ent, {"start": s, "end": e}) for s, e in rules_spans):
                from_model.append(m_ent)
                model_fallback_count += 1

    # --- Fallback 2: Rules catch ISSUE_DATE that model missed ---
    # Only ISSUE_DATE — NAME and PRODUCT have no reliable rules fallback
    model_spans = [(e["start"], e["end"]) for e in from_model]
    rules_fallback_count = 0
    for r_ent in rule_entities:
        if r_ent["label"] in RULES_FALLBACK_TYPES and r_ent["label"] in MODEL_OWNED_TYPES:
            if not any(_spans_overlap(r_ent, {"start": s, "end": e}) for s, e in model_spans):
                from_model.append(r_ent)
                rules_fallback_count += 1

    # --- Combine and deduplicate ---
    combined = from_rules + from_model
    combined.sort(key=lambda e: e["start"])

    resolved = []
    for ent in combined:
        if resolved and _spans_overlap(resolved[-1], ent):
            continue
        resolved.append(ent)

    # --- Self-correction filter (post-NER) ---
    # Detect correction markers and remove entities preceding the correction.
    # If the same entity type appears before and after a marker, the pre-marker
    # entity is the corrected-away value and should be removed.
    # "i mean" excluded — too common as discourse filler, causes false removals
    CORRECTION_MARKERS = [
        "no wait", "no actually", "no sorry",
        "actually no", "wait no", "sorry i meant", "no no",
    ]
    correction_removals = 0
    text_lower = text.lower()

    marker_positions = []
    for marker in CORRECTION_MARKERS:
        idx = 0
        while idx < len(text_lower):
            pos = text_lower.find(marker, idx)
            if pos == -1:
                break
            marker_positions.append((pos, pos + len(marker)))
            idx = pos + 1
    marker_positions.sort()

    # Only remove entities that are (a) within 30 chars of the marker,
    # (b) before the marker, and (c) have a same-type entity after the marker.
    # Distance constraint prevents removing distant legitimate entities.
    MAX_CORRECTION_DISTANCE = 30

    if marker_positions:
        filtered = []
        for ent in resolved:
            should_remove = False
            for m_start, m_end in marker_positions:
                # Entity must end before marker and be within distance limit
                if ent["end"] <= m_start and (m_start - ent["end"]) <= MAX_CORRECTION_DISTANCE:
                    # Check if a same-type entity exists after this marker
                    has_post_correction = any(
                        other["label"] == ent["label"]
                        and other["start"] >= m_end
                        for other in resolved
                        if other is not ent
                    )
                    if has_post_correction:
                        should_remove = True
                        correction_removals += 1
                        log.debug(
                            "Self-correction filter",
                            removed=f"{ent['label']}:'{ent['text']}'",
                            marker=text[m_start:m_end],
                        )
                        break
            if not should_remove:
                filtered.append(ent)
        resolved = filtered

    # --- Partial entity detection (structural entities only) ---
    # Check EMAIL and CONTRACT_ID for format completeness.
    # Semantic entities (NAME, PRODUCT, ISSUE_DATE) are not checked —
    # reliable partial detection for these requires speaker intent inference.
    import re

    for ent in resolved:
        ent["partial"] = False  # default

        if ent["label"] == "EMAIL":
            t = ent["text"].lower()
            has_at = "@" in t or " at " in t or t.startswith("at ") or t.endswith(" at")
            has_domain = any(d in t for d in [".com", ".org", ".net", ".co", "dot com", "dot org", "dot net", "dot co"])
            if not has_at or not has_domain:
                ent["partial"] = True

        elif ent["label"] == "CONTRACT_ID":
            t = ent["text"]
            # Standard format: 2-4 letters + separator + digits
            standard = bool(re.match(r"^[A-Za-z]{2,4}[-\s]", t))
            # Oral format: contains "dash" (may use word-numbers instead of digits)
            oral = bool(re.search(r"dash", t, re.IGNORECASE))
            # Has digits or word-numbers (for spelled-out oral format)
            has_digits = bool(re.search(r"\d", t))
            has_word_numbers = bool(re.search(
                r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|oh)\b",
                t, re.IGNORECASE,
            ))
            # Partial if: no numeric content at all, or no recognizable format
            if (not has_digits and not has_word_numbers) or (not standard and not oral):
                ent["partial"] = True

    stats = {
        "conflicts": conflicts,
        "model_fallback": model_fallback_count,
        "rules_fallback": rules_fallback_count,
        "correction_removals": correction_removals,
        "total_entities": len(resolved),
    }

    if conflicts > 0 or model_fallback_count > 0 or rules_fallback_count > 0 or correction_removals > 0:
        log.info("Ensemble stats", **stats)

    return resolved, stats