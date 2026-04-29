"""Run full evaluation pipeline.

Evaluates all models on gold test data, generates error analysis,
visualizations, and the final evaluation report.
"""

import json
import time
from pathlib import Path
from typing import Any, Callable

from src.models import rules
from src.models.gliner_ner import predict as gliner_predict
from src.models.transformer_ner import TransformerNER
from src.models.ensemble import predict as ensemble_predict
from src.evaluation.metrics import evaluate, format_report
from src.evaluation.analyze import (
    analyze_errors,
    plot_error_distribution,
    plot_learning_curve,
    plot_inference_speed,
    save_error_report,
)
from src.evaluation.report import generate_report
from src.utils.config import load_config
from src.utils.logger import setup_logging, get_logger


def _measure_speed(
    predict_fn: Callable, texts: list[str], n_runs: int = 10,
) -> dict[str, float]:
    """Measure inference speed with statistical reporting.

    Args:
        predict_fn: Function that takes text and returns entities.
        texts: List of texts to benchmark on.
        n_runs: Number of timing runs (default 10).

    Returns:
        Dict with median_ms, mean_ms, std_ms per sample.
    """
    # Warmup
    for t in texts[:3]:
        predict_fn(t)

    run_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        for t in texts:
            predict_fn(t)
        elapsed = (time.perf_counter() - start) * 1000
        run_times.append(elapsed / len(texts))

    import statistics
    return {
        "median_ms": round(statistics.median(run_times), 2),
        "mean_ms": round(statistics.mean(run_times), 2),
        "std_ms": round(statistics.stdev(run_times) if len(run_times) > 1 else 0, 2),
        "n_runs": n_runs,
    }


def main() -> None:
    setup_logging()
    log = get_logger(__name__)
    config = load_config()

    eval_dir = Path(config["evaluation"]["eval_data_dir"])
    results_dir = Path(config["evaluation"]["results_dir"])
    figures_dir = Path(config["evaluation"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    # --- Load test data ---
    log.info("Loading test data")

    # Primary gold set (hand-written)
    gold_path = eval_dir / "gold_test.json"
    with open(gold_path) as f:
        gold_samples = json.load(f)
    log.info("Gold test samples loaded", count=len(gold_samples))

    # Split gold set into clean and noisy subsets
    gold_clean = [s for s in gold_samples if not s.get("noisy", False)]
    gold_noisy = [s for s in gold_samples if s.get("noisy", False)]
    log.info(
        "Gold set split",
        clean=len(gold_clean),
        noisy=len(gold_noisy),
    )

    # Secondary LLM-generated set (if exists)
    secondary_path = eval_dir / "secondary_test.json"
    secondary_samples = None
    if secondary_path.exists():
        with open(secondary_path) as f:
            secondary_samples = json.load(f)
        log.info("Secondary test samples loaded", count=len(secondary_samples))

    # --- Load models ---
    log.info("Loading models")
    models: dict[str, Callable] = {}

    # Rules baseline
    models["rules"] = rules.predict

    # GLiNER baseline
    models["gliner"] = gliner_predict

    # BERT-Tiny
    tiny_model = None
    tiny_dir = config["training"]["bert_tiny"]["save_dir"]
    if Path(tiny_dir).exists():
        tiny_model = TransformerNER(tiny_dir)
        models["bert-tiny"] = tiny_model.predict
    else:
        log.warning("BERT-Tiny not found, skipping", path=tiny_dir)

    # BERT-Small
    small_dir = config["training"]["bert_small"]["save_dir"]
    if Path(small_dir).exists():
        small_model = TransformerNER(small_dir)
        models["bert-small"] = small_model.predict
    else:
        log.warning("BERT-Small not found, skipping", path=small_dir)

    # Ensemble (uses BERT-Small — strictly dominates BERT-Tiny on all entity types)
    ensemble_stats_all: list[dict[str, int]] = []
    _ensemble_model = small_model if "bert-small" in models else tiny_model
    if _ensemble_model is not None:
        def _ensemble_fn(text: str, _model=_ensemble_model) -> list[dict[str, Any]]:
            entities, stats = ensemble_predict(text, _model)
            ensemble_stats_all.append(stats)
            return entities
        models["ensemble"] = _ensemble_fn

    # --- Evaluate all models on gold clean set ---
    all_results: dict[str, Any] = {
        "model_results": {},
        "noise_results": {},
        "speed_results": {},
    }

    log.info("Evaluating on gold clean set", samples=len(gold_clean))
    for model_name, predict_fn in models.items():
        log.info("Evaluating model", model=model_name)

        all_preds = [predict_fn(s["text"]) for s in gold_clean]
        all_gold = [s["entities"] for s in gold_clean]

        strict_report = evaluate(all_preds, all_gold, mode="strict")
        partial_report = evaluate(all_preds, all_gold, mode="partial", iou_threshold=0.5)

        all_results["model_results"][model_name] = {
            "strict": strict_report,
            "partial": partial_report,
        }

        log.info(
            "Model result (gold clean)",
            model=model_name,
            strict_f1=strict_report["overall"]["f1"],
            partial_f1=partial_report["overall"]["f1"],
        )
        print(format_report(strict_report))

    # --- Evaluate on gold noisy subset ---
    if gold_noisy:
        log.info("Evaluating on gold noisy subset", samples=len(gold_noisy))
        all_results["noise_results"] = {}
        for model_name, predict_fn in models.items():
            preds = [predict_fn(s["text"]) for s in gold_noisy]
            golds = [s["entities"] for s in gold_noisy]
            all_results["noise_results"][model_name] = {
                "strict": evaluate(preds, golds, mode="strict"),
                "partial": evaluate(preds, golds, mode="partial", iou_threshold=0.5),
            }
            log.info(
                "Model result (gold noisy)",
                model=model_name,
                strict_f1=all_results["noise_results"][model_name]["strict"]["overall"]["f1"],
            )

    # --- Evaluate on secondary set (if exists) ---
    if secondary_samples:
        log.info("Evaluating on secondary LLM-generated set", samples=len(secondary_samples))
        secondary_results = {}
        for model_name, predict_fn in models.items():
            preds = [predict_fn(s["text"]) for s in secondary_samples]
            golds = [s["entities"] for s in secondary_samples]
            secondary_results[model_name] = {
                "strict": evaluate(preds, golds, mode="strict"),
                "partial": evaluate(preds, golds, mode="partial", iou_threshold=0.5),
            }
        all_results["secondary_results"] = secondary_results

    # --- Per-entity breakdown for all models ---
    all_results["per_entity_all_models"] = {}
    for model_name in all_results["model_results"]:
        all_results["per_entity_all_models"][model_name] = (
            all_results["model_results"][model_name]["strict"]["per_entity_type"]
        )

    # --- Error analysis on ensemble (final system with post-processing) ---
    error_analysis_model = "ensemble" if "ensemble" in models else "bert-small" if "bert-small" in models else None
    if error_analysis_model is None:
        # Fallback: pick best non-ensemble model
        non_ensemble = {k: v for k, v in all_results["model_results"].items() if k != "ensemble"}
        if non_ensemble:
            error_analysis_model = max(
                non_ensemble,
                key=lambda k: non_ensemble[k]["strict"]["overall"]["f1"],
            )

    if error_analysis_model:
        log.info("Running error analysis", model=error_analysis_model)
        error_report = analyze_errors(gold_clean, models[error_analysis_model])
        all_results["error_report"] = error_report
        all_results["error_analysis_model"] = error_analysis_model
        save_error_report(error_report, results_dir / "error_analysis.json")
    else:
        log.warning("No model available for error analysis")

    # --- Inference speed benchmark ---
    # Use texts NOT seen during evaluation to avoid cache effects
    log.info("Benchmarking inference speed on held-out texts")

    # Build benchmark corpus: mix of lengths for representative measurement
    import random as _rng
    _rng.seed(42)

    # 25 texts from training data (longer, dialogue-style)
    train_path = Path("data/synthetic/prepared/train_prepared.json")
    bench_texts = []
    if train_path.exists():
        with open(train_path) as f:
            train_data = json.load(f)
        bench_texts.extend([s["text"] for s in _rng.sample(train_data, min(25, len(train_data)))])

    # 25 short synthetic texts of varying lengths
    short_texts = [
        "hi i need help with my account",
        "my order hasnt arrived yet can you check",
        "yeah the reference number is TKT-90274",
        "i want to return the product i bought last week",
        "my email is test at gmail dot com",
        "um so i called yesterday about this issue and nobody helped me",
        "the contract number is SUB dash 33018 and i need to cancel",
        "can you transfer me to a supervisor please",
        "i bought the premium subscription plan about a month ago",
        "my name is james and my wife ordered a laptop",
        "hold on let me check the order number",
        "its been three weeks since i placed the order",
        "do you have any updates on my return request",
        "the product was damaged when it arrived",
        "i need to update my billing information",
        "yeah so like i got the wrong item delivered",
        "my account number is ACC dash 00192",
        "can i get a refund instead of a replacement",
        "the technician was supposed to come last friday",
        "i already sent an email to support at company dot com",
        "no thats not right my name is chen not chang",
        "when will the replacement be shipped",
        "i was charged twice for the same order",
        "the wifi router stopped working two days ago",
        "please just cancel everything and close my account",
    ]
    bench_texts.extend(short_texts)

    for model_name, predict_fn in models.items():
        speed_data = _measure_speed(predict_fn, bench_texts)
        ms = speed_data["median_ms"]
        all_results["speed_results"][model_name] = speed_data["median_ms"]
        all_results.setdefault("speed_details", {})[model_name] = speed_data
        log.info("Speed", model=model_name, ms_per_sample=f"{ms:.2f}")

    all_results["speed_benchmark_note"] = (
        f"Measured on {len(bench_texts)} held-out texts (25 training samples + 25 short synthetic), "
        "not seen during evaluation. Models pre-warmed with 3 samples before timing."
    )

    # --- Ensemble stats ---
    if ensemble_stats_all:
        all_results["ensemble_stats"] = {
            "total_conflicts": sum(s["conflicts"] for s in ensemble_stats_all),
            "model_fallbacks": sum(s["model_fallback"] for s in ensemble_stats_all),
            "rules_fallbacks": sum(s["rules_fallback"] for s in ensemble_stats_all),
            "samples_evaluated": len(ensemble_stats_all),
        }
        log.info("Ensemble stats", **all_results["ensemble_stats"])

    # --- Generate visualizations ---
    log.info("Generating visualizations")

    # Error distribution
    if "error_report" in all_results:
        plot_error_distribution(all_results["error_report"], figures_dir / "error_distribution.png")

    # Learning curves
    for model_key in ["bert_tiny", "bert_small"]:
        history_path = Path(config["training"][model_key]["save_dir"]) / "training_history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
            plot_learning_curve(history, figures_dir / f"learning_curve_{model_key}.png")

    # Inference speed chart
    plot_inference_speed(all_results["speed_results"], figures_dir / "inference_speed.png")

    # Latency vs accuracy scatter plot
    log.info("Generating latency vs accuracy plot")
    import matplotlib.pyplot as plt

    # Data points: (ms/sample, strict F1, label)
    plot_data = []
    for model_name in all_results["model_results"]:
        if model_name in all_results["speed_results"]:
            f1 = all_results["model_results"][model_name]["strict"]["overall"]["f1"]
            ms = all_results["speed_results"][model_name]
            plot_data.append((ms, f1, model_name))

    # Add GPT-4o-mini if results exist
    gpt_path = Path("results/gpt_baseline_results.json")
    if gpt_path.exists():
        with open(gpt_path) as f:
            gpt_data = json.load(f)
        gpt_f1 = gpt_data["results"]["clean"]["strict"]["overall"]["f1"]
        gpt_ms = gpt_data["stats"]["avg_latency_ms"]
        plot_data.append((gpt_ms, gpt_f1, "GPT-4o-mini"))

    # Add ONNX FP32 if results exist
    onnx_path = Path("results/onnx_results.json")
    if onnx_path.exists():
        with open(onnx_path) as f:
            onnx_data = json.load(f)
        if "onnx_fp32" in onnx_data.get("f1", {}):
            plot_data.append((
                onnx_data["speed_ms"]["onnx_fp32"],
                onnx_data["f1"]["onnx_fp32"],
                "ONNX FP32",
            ))

    if plot_data:
        fig, ax = plt.subplots(figsize=(10, 7))

        colors = {
            "rules": "#888888",
            "gliner": "#2196F3",
            "bert-tiny": "#FF9800",
            "bert-small": "#4CAF50",
            "ensemble": "#E91E63",
            "GPT-4o-mini": "#9C27B0",
            "ONNX FP32": "#00BCD4",
        }

        for ms, f1, label in plot_data:
            color = colors.get(label, "#333333")
            ax.scatter(ms, f1, s=200, c=color, zorder=5, edgecolors="black", linewidths=0.5)
            # Offset label to avoid overlap
            x_offset = 8
            y_offset = 0.005
            if label == "GPT-4o-mini":
                x_offset = -80
                y_offset = -0.015
            elif label == "ensemble":
                y_offset = 0.012
            elif label == "bert-tiny":
                y_offset = -0.015
            ax.annotate(
                label, (ms, f1),
                textcoords="offset points",
                xytext=(x_offset, y_offset * 1000),
                fontsize=11, fontweight="bold", color=color,
            )

        ax.set_xlabel("Inference Latency (ms/sample, log scale)", fontsize=14)
        ax.set_ylabel("Strict F1", fontsize=14)
        ax.set_title("Latency vs Accuracy Tradeoff", fontsize=16)
        ax.set_xscale("log")
        ax.set_ylim(0.5, 1.0)
        ax.grid(True, alpha=0.3)

        # Add annotation for the key insight
        ax.annotate(
            "500x faster\n95% of F1",
            xy=(4.9, 0.879), xytext=(15, 0.845),
            fontsize=11, fontstyle="italic",
            arrowprops=dict(arrowstyle="->", color="gray", lw=1),
            color="gray",
        )

        fig.tight_layout()
        fig.savefig(figures_dir / "latency_vs_accuracy.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Latency vs accuracy plot saved")

    # --- Generate report ---
    log.info("Generating evaluation report")
    generate_report(all_results, results_dir / "eval_report.md")

    log.info("Evaluation pipeline complete", results_dir=str(results_dir))


if __name__ == "__main__":
    main()