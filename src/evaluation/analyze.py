"""Error analysis — classify errors, generate visualizations, identify failure patterns.

Provides detailed analysis of NER model errors including error type classification,
noise-level vs entity-type heatmaps, and learning curve generation.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.evaluation.metrics import evaluate, _compute_iou
from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def classify_errors(
    predictions: list[dict[str, Any]],
    gold: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Classify each prediction error into a category.

    Error types:
    - BOUNDARY: Correct type, overlapping span but not exact match
    - TYPE_CONFUSION: Overlapping span but wrong entity type
    - COMPLETE_MISS: Gold entity with no overlapping prediction (false negative)
    - FALSE_ALARM: Prediction with no overlapping gold entity (false positive)

    Args:
        predictions: Predicted entity list.
        gold: Gold entity list.

    Returns:
        List of error dicts with category, details, and involved entities.
    """
    errors = []
    matched_gold = set()
    matched_pred = set()

    # Check each prediction against gold
    for p_idx, pred in enumerate(predictions):
        best_overlap = None
        best_iou = 0.0
        best_g_idx = None

        for g_idx, g in enumerate(gold):
            iou = _compute_iou(pred, g)
            if iou > best_iou:
                best_iou = iou
                best_overlap = g
                best_g_idx = g_idx

        if best_overlap is not None and best_iou > 0.1:
            matched_pred.add(p_idx)

            if pred["label"] != best_overlap["label"]:
                errors.append({
                    "type": "TYPE_CONFUSION",
                    "predicted": pred,
                    "gold": best_overlap,
                    "iou": round(best_iou, 3),
                    "detail": f"Predicted {pred['label']}, actual {best_overlap['label']}",
                })
                matched_gold.add(best_g_idx)

            elif pred["start"] != best_overlap["start"] or pred["end"] != best_overlap["end"]:
                errors.append({
                    "type": "BOUNDARY",
                    "predicted": pred,
                    "gold": best_overlap,
                    "iou": round(best_iou, 3),
                    "detail": f"Pred [{pred['start']}:{pred['end']}] vs Gold [{best_overlap['start']}:{best_overlap['end']}]",
                })
                matched_gold.add(best_g_idx)

            else:
                # Correct match — not an error
                matched_gold.add(best_g_idx)

    # False alarms: predictions with no gold overlap
    for p_idx, pred in enumerate(predictions):
        if p_idx not in matched_pred:
            errors.append({
                "type": "FALSE_ALARM",
                "predicted": pred,
                "gold": None,
                "detail": f"No matching gold entity for '{pred['text']}' ({pred['label']})",
            })

    # Complete misses: gold entities with no prediction overlap
    for g_idx, g in enumerate(gold):
        if g_idx not in matched_gold:
            errors.append({
                "type": "COMPLETE_MISS",
                "predicted": None,
                "gold": g,
                "detail": f"Missed '{g['text']}' ({g['label']})",
            })

    return errors


def analyze_errors(
    samples: list[dict[str, Any]],
    predict_fn: Any,
) -> dict[str, Any]:
    """Run full error analysis on a dataset.

    Args:
        samples: NER samples with gold annotations.
        predict_fn: Callable that takes text and returns entity list.

    Returns:
        Error analysis report.
    """
    all_errors = []
    error_type_counts = defaultdict(int)
    error_by_entity_type = defaultdict(lambda: defaultdict(int))
    error_by_noise_level = defaultdict(lambda: defaultdict(int))

    for sample in samples:
        preds = predict_fn(sample["text"])
        gold = sample["entities"]
        noise_level = sample.get("noise_level", "unknown")

        errors = classify_errors(preds, gold)
        all_errors.extend(errors)

        for error in errors:
            error_type_counts[error["type"]] += 1

            # Track by entity type
            entity = error.get("gold") or error.get("predicted")
            if entity:
                error_by_entity_type[entity["label"]][error["type"]] += 1

            # Track by noise level
            error_by_noise_level[noise_level][error["type"]] += 1

    report = {
        "total_errors": len(all_errors),
        "error_type_counts": dict(error_type_counts),
        "error_by_entity_type": {k: dict(v) for k, v in error_by_entity_type.items()},
        "error_by_noise_level": {k: dict(v) for k, v in error_by_noise_level.items()},
        "sample_errors": all_errors[:50],  # Keep first 50 for inspection
    }

    log.info(
        "Error analysis complete",
        total_errors=len(all_errors),
        types=dict(error_type_counts),
    )

    return report


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_noise_entity_heatmap(
    samples: list[dict[str, Any]],
    predict_fn: Any,
    output_path: Path,
    mode: str = "strict",
) -> None:
    """Generate a heatmap of F1 scores: noise level × entity type.

    Args:
        samples: NER samples with noise_level metadata.
        predict_fn: Callable that takes text and returns entity list.
        output_path: Path to save the heatmap PNG.
    """
    # Group by noise level
    by_level = defaultdict(list)
    for s in samples:
        by_level[s.get("noise_level", "unknown")].append(s)

    level_order = ["clean", "mild", "moderate", "severe"]
    entity_types = ["NAME", "EMAIL", "CONTRACT_ID", "PRODUCT", "ISSUE_DATE"]

    # Compute F1 for each (noise_level, entity_type) pair
    matrix = np.zeros((len(level_order), len(entity_types)))

    for i, level in enumerate(level_order):
        if level not in by_level:
            continue
        level_samples = by_level[level]
        all_preds = [predict_fn(s["text"]) for s in level_samples]
        all_gold = [s["entities"] for s in level_samples]
        report = evaluate(all_preds, all_gold, mode=mode)

        for j, etype in enumerate(entity_types):
            if etype in report["per_entity_type"]:
                matrix[i, j] = report["per_entity_type"][etype]["f1"]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        xticklabels=entity_types,
        yticklabels=level_order,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_title("F1 Score by Noise Level × Entity Type")
    ax.set_xlabel("Entity Type")
    ax.set_ylabel("Noise Level")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    log.info("Heatmap saved", path=str(output_path))


def plot_error_distribution(
    error_report: dict[str, Any],
    output_path: Path,
) -> None:
    """Plot error type distribution as a bar chart.

    Args:
        error_report: Report from analyze_errors().
        output_path: Path to save the chart PNG.
    """
    counts = error_report["error_type_counts"]
    if not counts:
        return

    labels = list(counts.keys())
    values = list(counts.values())
    colors = {
        "BOUNDARY": "#f39c12",
        "TYPE_CONFUSION": "#e74c3c",
        "COMPLETE_MISS": "#3498db",
        "FALSE_ALARM": "#9b59b6",
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        labels, values,
        color=[colors.get(lbl, "#95a5a6") for lbl in labels],
    )
    ax.set_title("Error Type Distribution")
    ax.set_ylabel("Count")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            str(val), ha="center", va="bottom", fontweight="bold",
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    log.info("Error distribution chart saved", path=str(output_path))


def plot_learning_curve(
    training_history: dict[str, Any],
    output_path: Path,
) -> None:
    """Plot training and validation loss curves.

    Args:
        training_history: History dict from training.
        output_path: Path to save the chart PNG.
    """
    train_loss = training_history["train_loss"]
    val_loss = training_history["val_loss"]
    epochs = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, "b-o", label="Train Loss", markersize=4)
    ax.plot(epochs, val_loss, "r-o", label="Val Loss", markersize=4)
    ax.set_title(f"Training Curve — {training_history.get('model_key', 'model')}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    log.info("Learning curve saved", path=str(output_path))


def plot_inference_speed(
    speed_results: dict[str, float],
    output_path: Path,
) -> None:
    """Plot inference speed comparison across models.

    Args:
        speed_results: Dict mapping model name to avg ms per sample.
        output_path: Path to save the chart PNG.
    """
    models = list(speed_results.keys())
    times = list(speed_results.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(models, times, color="#3498db")
    ax.set_title("Inference Speed (CPU)")
    ax.set_xlabel("Average ms per sample")

    for bar, val in zip(bars, times):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}ms", ha="left", va="center",
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    log.info("Inference speed chart saved", path=str(output_path))


def save_error_report(report: dict[str, Any], output_path: Path) -> None:
    """Save error analysis report as JSON.

    Args:
        report: Error analysis report dict.
        output_path: Path to save JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Error report saved", path=str(output_path))