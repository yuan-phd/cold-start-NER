"""Bootstrap confidence intervals for NER evaluation metrics.

Computes 95% confidence intervals via bootstrap resampling on the gold
test set. Reports CIs for all models on both clean and noisy subsets.

Output:
    results/bootstrap_results.json — per-model CIs
    Console summary

Usage:
    python scripts/run_bootstrap.py
"""

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.metrics import evaluate
from src.models.transformer_ner import TransformerNER
from src.models import rules as rules_module
from src.models import ensemble as ensemble_module
from src.utils.config import load_config
from src.utils.logger import setup_logging, get_logger


def bootstrap_f1(
    preds: list[list[dict]],
    golds: list[list[dict]],
    n_bootstrap: int = 1000,
    seed: int = 42,
    mode: str = "strict",
) -> dict[str, float]:
    """Compute bootstrap 95% CI for F1 score.

    Args:
        preds: Per-sample predictions (list of entity lists).
        golds: Per-sample gold annotations (list of entity lists).
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed for reproducibility.
        mode: "strict" or "partial".

    Returns:
        Dict with mean, std, ci_lower, ci_upper, median.
    """
    rng = random.Random(seed)
    n = len(preds)
    f1_scores: list[float] = []

    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        boot_preds = [preds[i] for i in indices]
        boot_golds = [golds[i] for i in indices]
        result = evaluate(boot_preds, boot_golds, mode=mode)
        f1_scores.append(result["overall"]["f1"])

    arr = np.array(f1_scores)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "ci_lower": float(np.percentile(arr, 2.5)),
        "ci_upper": float(np.percentile(arr, 97.5)),
        "median": float(np.median(arr)),
        "n_bootstrap": n_bootstrap,
    }


def main() -> None:
    setup_logging()
    log = get_logger(__name__)
    config = load_config()

    # Load gold test set
    eval_dir = Path(config["evaluation"]["eval_data_dir"])
    with open(eval_dir / "gold_test.json") as f:
        gold = json.load(f)
    gold_clean = [s for s in gold if not s.get("noisy", False)]
    gold_noisy = [s for s in gold if s.get("noisy", False)]

    # Pre-load BERT model once (reused across subsets)
    bert_small = TransformerNER(config["training"]["bert_small"]["save_dir"])

    # Define predict functions for each model
    def predict_rules(text: str) -> list[dict]:
        return rules_module.predict(text)

    def predict_bert(text: str) -> list[dict]:
        return bert_small.predict(text)

    def predict_ensemble(text: str) -> list[dict]:
        entities, _stats = ensemble_module.predict(text, model=bert_small)
        return entities

    models = {
        "rules": predict_rules,
        "bert-small": predict_bert,
        "ensemble": predict_ensemble,
    }

    results: dict[str, Any] = {}

    for subset_name, subset in [("clean", gold_clean), ("noisy", gold_noisy)]:
        log.info(f"Bootstrap on {subset_name} set", n=len(subset))
        results[subset_name] = {}

        for model_name, predict_fn in models.items():
            log.info(f"  {model_name}")
            preds = [predict_fn(s["text"]) for s in subset]
            golds = [s["entities"] for s in subset]

            strict_ci = bootstrap_f1(preds, golds, n_bootstrap=1000, mode="strict")
            partial_ci = bootstrap_f1(preds, golds, n_bootstrap=1000, mode="partial")

            results[subset_name][model_name] = {
                "strict": strict_ci,
                "partial": partial_ci,
            }

    # Print summary
    print("\n" + "=" * 70)
    print("BOOTSTRAP 95% CONFIDENCE INTERVALS (1000 resamples)")
    print("=" * 70)

    for subset_name in ["clean", "noisy"]:
        n = len(gold_clean) if subset_name == "clean" else len(gold_noisy)
        print(f"\n--- {subset_name.capitalize()} set (n={n}) ---")
        print(f"  {'Model':<15} {'Strict F1':>10} {'95% CI':>20} {'Partial F1':>12} {'95% CI':>20}")
        print(f"  {'-'*79}")

        for model_name in ["rules", "bert-small", "ensemble"]:
            s = results[subset_name][model_name]["strict"]
            p = results[subset_name][model_name]["partial"]
            print(
                f"  {model_name:<15} {s['mean']:>10.3f} "
                f"[{s['ci_lower']:.3f}, {s['ci_upper']:.3f}]"
                f" {p['mean']:>12.3f} "
                f"[{p['ci_lower']:.3f}, {p['ci_upper']:.3f}]"
            )

    print("=" * 70)

    # Save
    output_path = Path("results/bootstrap_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Bootstrap results saved", path=str(output_path))


if __name__ == "__main__":
    main()
