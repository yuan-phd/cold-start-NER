"""Data quality validation — verify span alignment, schema, and distributions.

Provides validation for both clean and noisy NER samples, with detailed
statistics on entity type distribution, text length, and data quality.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

log = get_logger(__name__)

VALID_LABELS = {"NAME", "EMAIL", "CONTRACT_ID", "PRODUCT", "ISSUE_DATE"}


def validate_sample(sample: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a single NER sample.

    Args:
        sample: Dict with "text" and "entities" keys.

    Returns:
        (is_valid, list_of_issues) tuple.
    """
    issues = []

    # Check required keys
    if "text" not in sample:
        issues.append("Missing 'text' key")
        return False, issues
    if "entities" not in sample:
        issues.append("Missing 'entities' key")
        return False, issues

    text = sample["text"]
    if not isinstance(text, str) or len(text.strip()) < 5:
        issues.append(f"Text too short or invalid: '{text[:50]}'")

    for i, ent in enumerate(sample["entities"]):
        prefix = f"Entity {i}"

        # Required fields
        for field in ("text", "label", "start", "end"):
            if field not in ent:
                issues.append(f"{prefix}: missing '{field}'")
                continue

        if "start" not in ent or "end" not in ent:
            continue

        # Label check
        if ent.get("label") not in VALID_LABELS:
            issues.append(f"{prefix}: invalid label '{ent.get('label')}'")

        # Offset types
        if not isinstance(ent["start"], int) or not isinstance(ent["end"], int):
            issues.append(f"{prefix}: non-integer offsets")
            continue

        # Offset bounds
        if ent["start"] < 0 or ent["end"] > len(text):
            issues.append(f"{prefix}: offset out of bounds [{ent['start']}:{ent['end']}] for text length {len(text)}")
            continue

        if ent["start"] >= ent["end"]:
            issues.append(f"{prefix}: start >= end [{ent['start']}:{ent['end']}]")
            continue

        # Span alignment
        extracted = text[ent["start"]:ent["end"]]
        if extracted != ent["text"]:
            # Check if it's a known partial/noisy entity
            if not ent.get("partial", False):
                issues.append(
                    f"{prefix}: span mismatch text[{ent['start']}:{ent['end']}] = "
                    f"'{extracted}' != '{ent['text']}'"
                )

    return len(issues) == 0, issues


def validate_dataset(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate an entire dataset and compute statistics.

    Args:
        samples: List of NER sample dicts.

    Returns:
        Validation report dict with stats and issues.
    """
    report = {
        "total_samples": len(samples),
        "valid_samples": 0,
        "invalid_samples": 0,
        "total_entities": 0,
        "entity_type_counts": Counter(),
        "samples_by_entity_count": Counter(),
        "text_lengths": [],
        "issues": [],
        "samples_with_issues": [],
    }

    for i, sample in enumerate(samples):
        is_valid, issues = validate_sample(sample)

        if is_valid:
            report["valid_samples"] += 1
        else:
            report["invalid_samples"] += 1
            report["samples_with_issues"].append({"index": i, "issues": issues})

        report["issues"].extend(issues)

        entities = sample.get("entities", [])
        report["total_entities"] += len(entities)
        report["samples_by_entity_count"][len(entities)] += 1
        report["text_lengths"].append(len(sample.get("text", "")))

        for ent in entities:
            label = ent.get("label", "UNKNOWN")
            report["entity_type_counts"][label] += 1

    # Compute summary stats
    text_lengths = report["text_lengths"]
    if text_lengths:
        report["text_length_stats"] = {
            "min": min(text_lengths),
            "max": max(text_lengths),
            "mean": sum(text_lengths) / len(text_lengths),
        }

    # Convert Counters to dicts for JSON serialization
    report["entity_type_counts"] = dict(report["entity_type_counts"])
    report["samples_by_entity_count"] = dict(report["samples_by_entity_count"])

    log.info(
        "Dataset validation complete",
        total=report["total_samples"],
        valid=report["valid_samples"],
        invalid=report["invalid_samples"],
        entities=report["total_entities"],
        entity_types=report["entity_type_counts"],
    )

    return report


def validate_file(path: Path) -> dict[str, Any]:
    """Load and validate a JSON dataset file.

    Args:
        path: Path to JSON file containing a list of NER samples.

    Returns:
        Validation report dict.
    """
    with open(path) as f:
        samples = json.load(f)

    log.info("Validating dataset file", path=str(path), samples=len(samples))
    report = validate_dataset(samples)
    report["file"] = str(path)
    return report


def print_report(report: dict[str, Any]) -> None:
    """Print a human-readable validation report.

    Args:
        report: Report dict from validate_dataset or validate_file.
    """
    print("\n" + "=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)

    if "file" in report:
        print(f"File: {report['file']}")

    print(f"\nSamples: {report['total_samples']} total, "
          f"{report['valid_samples']} valid, "
          f"{report['invalid_samples']} invalid "
          f"({report['valid_samples']/max(report['total_samples'],1):.1%} acceptance rate)")

    print(f"\nTotal entities: {report['total_entities']}")
    print(f"Avg entities per sample: {report['total_entities']/max(report['total_samples'],1):.1f}")

    print("\nEntity type distribution:")
    for etype, count in sorted(report["entity_type_counts"].items()):
        pct = count / max(report["total_entities"], 1) * 100
        bar = "█" * int(pct / 2)
        print(f"  {etype:<15} {count:>4} ({pct:5.1f}%) {bar}")

    print("\nSamples by entity count:")
    for n_ents, count in sorted(report["samples_by_entity_count"].items()):
        print(f"  {n_ents} entities: {count} samples")

    if "text_length_stats" in report:
        stats = report["text_length_stats"]
        print(f"\nText length: min={stats['min']}, max={stats['max']}, mean={stats['mean']:.0f}")

    if report["invalid_samples"] > 0:
        print("\nFirst 5 issues:")
        for item in report["samples_with_issues"][:5]:
            print(f"  Sample {item['index']}:")
            for issue in item["issues"]:
                print(f"    - {issue}")

    print("=" * 60 + "\n")