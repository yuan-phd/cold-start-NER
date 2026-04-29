"""Confidence calibration analysis — assess whether model confidence scores
are well-calibrated against actual prediction accuracy.

Runs BERT-Small inference on the gold clean set, buckets predictions by
confidence score, and computes actual strict/partial precision per bucket.
Generates a reliability diagram comparing predicted confidence to observed
accuracy.

Output:
    results/calibration_results.json — per-bucket statistics
    results/figures/calibration.png — reliability diagram

Usage:
    python scripts/run_calibration.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.models.transformer_ner import TransformerNER
from src.utils.config import load_config
from src.utils.logger import setup_logging, get_logger


# Confidence buckets: finer resolution where predictions actually land
BUCKETS = [
    (0.0, 0.5),
    (0.5, 0.7),
    (0.7, 0.8),
    (0.8, 0.9),
    (0.9, 1.01),  # 1.01 to include confidence=1.0
]

MIN_BUCKET_SIZE = 5  # Below this, bucket is flagged as unreliable


def _bucket_label(lo: float, hi: float) -> str:
    """Human-readable bucket label."""
    hi_display = min(hi, 1.0)
    return f"{lo:.1f}-{hi_display:.1f}"


def _is_strict_match(pred: dict, gold_entities: list[dict]) -> bool:
    """Check if prediction strictly matches any gold entity."""
    return any(
        pred["label"] == g["label"]
        and pred["start"] == g["start"]
        and pred["end"] == g["end"]
        for g in gold_entities
    )


def _is_partial_match(pred: dict, gold_entities: list[dict], iou_threshold: float = 0.5) -> bool:
    """Check if prediction partially matches any gold entity (IoU >= threshold)."""
    for g in gold_entities:
        if pred["label"] != g["label"]:
            continue
        overlap_start = max(pred["start"], g["start"])
        overlap_end = min(pred["end"], g["end"])
        overlap = max(0, overlap_end - overlap_start)
        union = max(pred["end"], g["end"]) - min(pred["start"], g["start"])
        if union > 0 and overlap / union >= iou_threshold:
            return True
    return False


def run_calibration(
    model_dir: Path,
    gold_path: Path,
) -> dict:
    """Run confidence calibration analysis.

    Args:
        model_dir: Path to trained BERT-Small.
        gold_path: Path to gold test set JSON.

    Returns:
        Calibration results dict with per-bucket statistics.
    """
    log = get_logger(__name__)

    model = TransformerNER(model_dir)

    with open(gold_path) as f:
        gold = json.load(f)
    gold_clean = [s for s in gold if not s.get("noisy", False)]

    log.info("Running calibration", model=str(model_dir), samples=len(gold_clean))

    # Collect all predictions with their correctness
    bucket_data = defaultdict(lambda: {
        "total": 0,
        "strict_correct": 0,
        "partial_correct": 0,
        "confidences": [],
    })

    total_predictions = 0

    for sample in gold_clean:
        preds = model.predict(sample["text"])
        gold_ents = sample["entities"]

        for pred in preds:
            conf = pred.get("confidence", 0)
            total_predictions += 1

            # Find bucket
            bucket_key = None
            for lo, hi in BUCKETS:
                if lo <= conf < hi:
                    bucket_key = _bucket_label(lo, hi)
                    break
            if bucket_key is None:
                continue

            bucket_data[bucket_key]["total"] += 1
            bucket_data[bucket_key]["confidences"].append(conf)

            if _is_strict_match(pred, gold_ents):
                bucket_data[bucket_key]["strict_correct"] += 1
            if _is_partial_match(pred, gold_ents):
                bucket_data[bucket_key]["partial_correct"] += 1

    # Compute per-bucket statistics
    results = {
        "total_predictions": total_predictions,
        "buckets": {},
    }

    for lo, hi in BUCKETS:
        key = _bucket_label(lo, hi)
        b = bucket_data.get(key, {"total": 0, "strict_correct": 0, "partial_correct": 0, "confidences": []})
        n = b["total"]

        bucket_result = {
            "count": n,
            "reliable": n >= MIN_BUCKET_SIZE,
        }

        if n > 0:
            bucket_result["mean_confidence"] = round(np.mean(b["confidences"]), 4)
            bucket_result["strict_precision"] = round(b["strict_correct"] / n, 4)
            bucket_result["partial_precision"] = round(b["partial_correct"] / n, 4)
        else:
            bucket_result["mean_confidence"] = None
            bucket_result["strict_precision"] = None
            bucket_result["partial_precision"] = None

        if not bucket_result["reliable"]:
            bucket_result["note"] = f"n={n}, below threshold of {MIN_BUCKET_SIZE} — interpret with caution"

        results["buckets"][key] = bucket_result

    # Compute overall calibration error (ECE)
    # Expected Calibration Error: weighted average of |confidence - accuracy| per bucket
    ece = 0.0
    for key, b in results["buckets"].items():
        if b["count"] > 0 and b["reliable"] and b["mean_confidence"] is not None:
            weight = b["count"] / total_predictions
            gap = abs(b["mean_confidence"] - b["strict_precision"])
            ece += weight * gap
    results["expected_calibration_error"] = round(ece, 4)

    return results


def plot_reliability_diagram(
    results: dict,
    output_path: Path,
) -> None:
    """Generate reliability diagram (calibration plot)."""
    buckets = results["buckets"]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

    # Plot reliable buckets
    confs = []
    strict_precs = []
    partial_precs = []
    sizes = []
    labels_for_bars = []

    for key, b in buckets.items():
        if b["count"] > 0 and b["mean_confidence"] is not None:
            confs.append(b["mean_confidence"])
            strict_precs.append(b["strict_precision"])
            partial_precs.append(b["partial_precision"])
            sizes.append(b["count"])
            labels_for_bars.append(key)

    if confs:
        bar_width = 0.03

        # Strict precision bars
        ax.bar(
            [c - bar_width / 2 for c in confs],
            strict_precs,
            width=bar_width,
            alpha=0.7,
            color="#E91E63",
            label="Strict precision",
            edgecolor="black",
            linewidth=0.5,
        )

        # Partial precision bars
        ax.bar(
            [c + bar_width / 2 for c in confs],
            partial_precs,
            width=bar_width,
            alpha=0.7,
            color="#2196F3",
            label="Partial precision",
            edgecolor="black",
            linewidth=0.5,
        )

        # Annotate counts
        for c, sp, n, reliable in zip(confs, strict_precs, sizes, [buckets[lbl]["reliable"] for lbl in labels_for_bars]):
            suffix = "" if reliable else "*"
            ax.annotate(
                f"n={n}{suffix}",
                (c, max(sp, 0.02)),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                color="gray",
            )

    ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax.set_ylabel("Observed Precision", fontsize=12)
    ax.set_title("Confidence Calibration — BERT-Small", fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Add ECE annotation
    ece = results.get("expected_calibration_error", 0)
    ax.text(
        0.95, 0.05, f"ECE = {ece:.3f}",
        transform=ax.transAxes, fontsize=11,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )

    # Footnote for unreliable buckets
    ax.text(
        0.95, 0.01, f"* n < {MIN_BUCKET_SIZE}, unreliable",
        transform=ax.transAxes, fontsize=8,
        ha="right", va="bottom", color="gray",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_logging()
    log = get_logger(__name__)
    config = load_config()

    model_dir = Path(config["training"]["bert_small"]["save_dir"])
    gold_path = Path(config["evaluation"]["eval_data_dir"]) / "gold_test.json"

    # Run calibration
    results = run_calibration(model_dir, gold_path)

    # Save results
    results_path = Path("results/calibration_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Calibration results saved", path=str(results_path))

    # Generate plot
    fig_path = Path("results/figures/calibration.png")
    plot_reliability_diagram(results, fig_path)
    log.info("Calibration plot saved", path=str(fig_path))

    # Print summary
    print("\n" + "=" * 60)
    print("CONFIDENCE CALIBRATION — BERT-Small")
    print("=" * 60)
    print(f"Total predictions: {results['total_predictions']}")
    print(f"Expected Calibration Error (ECE): {results['expected_calibration_error']:.4f}")
    print()
    print(f"{'Bucket':<12} {'Count':>6} {'Mean Conf':>10} {'Strict Prec':>12} {'Partial Prec':>13} {'Reliable':>9}")
    print("-" * 64)

    for key, b in results["buckets"].items():
        n = b["count"]
        if n == 0:
            print(f"{key:<12} {n:>6} {'—':>10} {'—':>12} {'—':>13} {'—':>9}")
        else:
            rel = "yes" if b["reliable"] else "NO"
            print(
                f"{key:<12} {n:>6} {b['mean_confidence']:>10.4f} "
                f"{b['strict_precision']:>12.4f} {b['partial_precision']:>13.4f} {rel:>9}"
            )

    print("=" * 60)


if __name__ == "__main__":
    main()
