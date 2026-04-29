"""NER evaluation metrics — strict match, partial match, per-entity, per-noise-level.

Implements entity-level evaluation (not token-level) with support for both
strict matching (exact span + type) and partial matching (IoU-based overlap).
"""

from collections import defaultdict
from typing import Any

from src.utils.logger import get_logger

log = get_logger(__name__)


def _compute_iou(pred: dict[str, Any], gold: dict[str, Any]) -> float:
    """Compute Intersection over Union between two spans.

    Args:
        pred: Predicted entity with start/end.
        gold: Gold entity with start/end.

    Returns:
        IoU score between 0 and 1.
    """
    inter_start = max(pred["start"], gold["start"])
    inter_end = min(pred["end"], gold["end"])
    intersection = max(0, inter_end - inter_start)

    union = (pred["end"] - pred["start"]) + (gold["end"] - gold["start"]) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _match_entities(
    predictions: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    mode: str = "strict",
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    """Match predicted entities against gold entities.

    Args:
        predictions: List of predicted entity dicts.
        gold: List of gold entity dicts.
        mode: "strict" (exact match) or "partial" (IoU-based).
        iou_threshold: IoU threshold for partial matching.

    Returns:
        (true_positives, false_positives, false_negatives) counts.
    """
    matched_gold = set()
    tp = 0

    for pred in predictions:
        best_match = None
        best_iou = 0.0

        for i, g in enumerate(gold):
            if i in matched_gold:
                continue
            if pred["label"] != g["label"]:
                continue

            if mode == "strict":
                if pred["start"] == g["start"] and pred["end"] == g["end"]:
                    best_match = i
                    break
            elif mode == "partial":
                iou = _compute_iou(pred, g)
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_match = i

        if best_match is not None:
            tp += 1
            matched_gold.add(best_match)

    fp = len(predictions) - tp
    fn = len(gold) - len(matched_gold)

    return tp, fp, fn


def _precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Compute precision, recall, and F1 from counts.

    Args:
        tp: True positive count.
        fp: False positive count.
        fn: False negative count.

    Returns:
        Dict with precision, recall, f1 values.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": tp + fn,
    }


def evaluate(
    all_predictions: list[list[dict[str, Any]]],
    all_gold: list[list[dict[str, Any]]],
    mode: str = "strict",
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate NER predictions against gold annotations.

    Args:
        all_predictions: List of predicted entity lists (one per sample).
        all_gold: List of gold entity lists (one per sample).
        mode: "strict" or "partial" matching.
        iou_threshold: IoU threshold for partial mode.

    Returns:
        Evaluation report with overall and per-entity-type metrics.
    """
    assert len(all_predictions) == len(all_gold), (
        f"Prediction/gold count mismatch: {len(all_predictions)} vs {len(all_gold)}"
    )

    # Overall counts
    total_tp, total_fp, total_fn = 0, 0, 0

    # Per-entity-type counts
    type_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for preds, golds in zip(all_predictions, all_gold):
        # Overall
        tp, fp, fn = _match_entities(preds, golds, mode, iou_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Per entity type
        entity_types = set(e["label"] for e in preds + golds)
        for etype in entity_types:
            type_preds = [e for e in preds if e["label"] == etype]
            type_golds = [e for e in golds if e["label"] == etype]
            t_tp, t_fp, t_fn = _match_entities(type_preds, type_golds, mode, iou_threshold)
            type_counts[etype]["tp"] += t_tp
            type_counts[etype]["fp"] += t_fp
            type_counts[etype]["fn"] += t_fn

    # Compute metrics
    report = {
        "mode": mode,
        "iou_threshold": iou_threshold if mode == "partial" else None,
        "overall": _precision_recall_f1(total_tp, total_fp, total_fn),
        "per_entity_type": {},
    }

    for etype, counts in sorted(type_counts.items()):
        report["per_entity_type"][etype] = _precision_recall_f1(
            counts["tp"], counts["fp"], counts["fn"]
        )

    return report


def evaluate_by_noise_level(
    samples: list[dict[str, Any]],
    predict_fn: Any,
    mode: str = "strict",
    iou_threshold: float = 0.5,
) -> dict[str, dict[str, Any]]:
    """Evaluate predictions grouped by noise level.

    Args:
        samples: List of NER samples with "noise_level" metadata.
        predict_fn: Callable that takes text and returns entity list.
        mode: "strict" or "partial" matching.
        iou_threshold: IoU threshold for partial mode.

    Returns:
        Dict mapping noise_level to evaluation report.
    """
    # Group samples by noise level
    by_level: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        level = sample.get("noise_level", "unknown")
        by_level[level].append(sample)

    results = {}
    for level, level_samples in sorted(by_level.items()):
        all_preds = [predict_fn(s["text"]) for s in level_samples]
        all_gold = [s["entities"] for s in level_samples]
        results[level] = evaluate(all_preds, all_gold, mode, iou_threshold)
        log.info(
            "Noise level evaluation",
            level=level,
            f1=results[level]["overall"]["f1"],
            samples=len(level_samples),
        )

    return results


def format_report(report: dict[str, Any]) -> str:
    """Format an evaluation report as a readable string.

    Args:
        report: Report dict from evaluate().

    Returns:
        Formatted string.
    """
    lines = []
    lines.append(f"\n{'='*65}")
    lines.append(f"NER Evaluation Report ({report['mode']} match)")
    lines.append(f"{'='*65}")

    overall = report["overall"]
    lines.append(
        f"\nOverall:  P={overall['precision']:.3f}  R={overall['recall']:.3f}  "
        f"F1={overall['f1']:.3f}  (TP={overall['tp']} FP={overall['fp']} FN={overall['fn']})"
    )

    lines.append(f"\n{'Entity Type':<15} {'Precision':>9} {'Recall':>8} {'F1':>8} {'Support':>9}")
    lines.append("-" * 55)

    for etype, metrics in report["per_entity_type"].items():
        lines.append(
            f"{etype:<15} {metrics['precision']:>9.3f} {metrics['recall']:>8.3f} "
            f"{metrics['f1']:>8.3f} {metrics['support']:>9}"
        )

    lines.append(f"{'='*65}\n")
    return "\n".join(lines)