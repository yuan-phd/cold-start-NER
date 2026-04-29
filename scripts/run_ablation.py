"""Ablation study — measure the contribution of each data decision (V3).

Three variants, each removing one data decision from the V3 base (898 samples):
  Variant A: No noise augmentation (clean data only) — 898 samples
  Variant B: No oral format transforms — ~3592 samples (898 x 4 noise levels)
  Variant C: Partial entities filtered — ~3592 samples (898 x 4 noise levels)

All variants use BERT-Small with BIOES tagging (21 labels) for consistency
with V3 main results. Evaluated on V3 gold set (41 clean + 74 noisy).

V2 ablation used 660 base samples with BIO tagging on 54-sample gold set.
V3 re-run updates: 898 base samples (GPT + Gemini), BIOES tagging,
TalkBank-calibrated noise, 115-sample gold set.

Note: Variant A has an inherent sample count confound (898 vs ~3592)
that cannot be eliminated — acknowledged in README.
"""

import json
import time
from pathlib import Path
from typing import Any

from src.data.noise import create_noisy_dataset
from src.models.transformer_ner import train, TransformerNER
from src.evaluation.metrics import evaluate, format_report
from src.utils.config import load_config
from src.utils.logger import setup_logging, get_logger


def _load_raw_898() -> tuple[list[dict], list[dict]]:
    """Load all 898 raw base samples WITHOUT oral format transforms.

    Merges: train.json (600) + train_negative.json (60) +
            train_oral.json (100) + train_gemini.json (138) = 898
    Val: val.json (120)

    Oral format transforms are NOT applied — Variant B needs this.
    """
    data_dir = Path("data/synthetic")
    train_samples: list[dict] = []

    for fname in ["train.json", "train_negative.json", "train_oral.json", "train_gemini.json"]:
        fpath = data_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                train_samples.extend(json.load(f))

    with open(data_dir / "val.json") as f:
        val_samples = json.load(f)

    return train_samples, val_samples


def _load_prepared_898() -> tuple[list[dict], list[dict]]:
    """Load prepared data WITH oral format transforms already applied.

    Uses train_prepared.json (898 samples, output of run_prepare.py).
    Val: val.json (120) — oral transforms on val are less critical.
    """
    with open("data/synthetic/prepared/train_prepared.json") as f:
        train_samples = json.load(f)

    with open("data/synthetic/val.json") as f:
        val_samples = json.load(f)

    return train_samples, val_samples


def _load_gold(config: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """Load gold test set, split into clean and noisy."""
    eval_dir = Path(config["evaluation"]["eval_data_dir"])
    with open(eval_dir / "gold_test.json") as f:
        gold = json.load(f)
    clean = [s for s in gold if not s.get("noisy", False)]
    noisy = [s for s in gold if s.get("noisy", False)]
    return clean, noisy


def _model_exists(model_dir: Path) -> bool:
    """Check if a trained model already exists in the directory."""
    return (model_dir / "config.json").exists()


def _evaluate_model(
    model_dir: Path, gold_samples: list[dict], label: str, log: Any,
) -> dict:
    """Evaluate a trained model on a gold subset."""
    model = TransformerNER(model_dir)
    preds = [model.predict(s["text"]) for s in gold_samples]
    golds = [s["entities"] for s in gold_samples]
    strict = evaluate(preds, golds, mode="strict")
    partial = evaluate(preds, golds, mode="partial", iou_threshold=0.5)
    log.info(
        f"Ablation: {label}",
        strict_f1=strict["overall"]["f1"],
        partial_f1=partial["overall"]["f1"],
    )
    print(f"\n--- {label} ---")
    print(format_report(strict))
    return {"strict": strict, "partial": partial}


def main() -> None:
    setup_logging()
    log = get_logger(__name__)
    config = load_config()

    gold_clean, gold_noisy = _load_gold(config)
    log.info("Gold sets loaded", clean=len(gold_clean), noisy=len(gold_noisy))

    ablation_dir = Path("results/models/ablation")
    ablation_dir.mkdir(parents=True, exist_ok=True)

    bert_small_config = config["training"]["bert_small"]
    results: dict[str, Any] = {}

    # Track sample counts for summary
    sample_counts: dict[str, int] = {}

    # =================================================================
    # Variant A: No noise augmentation
    # 898 base WITH oral transforms, NO noise — 898 training samples
    # Tests: does noise augmentation help?
    # =================================================================
    log.info("=" * 60)
    log.info("Variant A: No noise augmentation (898 clean samples)")

    train_a, val_a = _load_prepared_898()
    sample_counts["A"] = len(train_a)

    variant_a_dir = ablation_dir / "variant_a_no_noise"
    config_a = {
        **config,
        "training": {
            **config["training"],
            "bert_small": {**bert_small_config, "save_dir": str(variant_a_dir)},
        },
    }

    if _model_exists(variant_a_dir):
        log.info("Variant A already trained, skipping", path=str(variant_a_dir))
    else:
        log.info("Training Variant A", samples=len(train_a))
        start = time.perf_counter()
        train(
            train_samples=train_a,
            val_samples=val_a,
            model_key="bert_small",
            config=config_a,
        )
        log.info("Variant A trained", time_s=f"{time.perf_counter() - start:.0f}")

    results["A_no_noise"] = _evaluate_model(
        variant_a_dir, gold_clean, "Variant A: No noise (CLEAN)", log,
    )
    results["A_no_noise_noisy"] = _evaluate_model(
        variant_a_dir, gold_noisy, "Variant A: No noise (NOISY)", log,
    )

    # =================================================================
    # Variant B: No oral format transforms
    # 898 base WITHOUT oral transforms, WITH noise
    # Tests: do oral format transforms help?
    # =================================================================
    log.info("=" * 60)
    log.info("Variant B: No oral format transforms")

    train_b_raw, val_b_raw = _load_raw_898()
    # Apply noise WITHOUT oral transforms
    noisy_train_b = create_noisy_dataset(train_b_raw, config=config)
    noisy_val_b = create_noisy_dataset(val_b_raw, config=config)
    sample_counts["B"] = len(noisy_train_b)

    variant_b_dir = ablation_dir / "variant_b_no_oral"
    config_b = {
        **config,
        "training": {
            **config["training"],
            "bert_small": {**bert_small_config, "save_dir": str(variant_b_dir)},
        },
    }

    if _model_exists(variant_b_dir):
        log.info("Variant B already trained, skipping", path=str(variant_b_dir))
    else:
        log.info("Training Variant B", samples=len(noisy_train_b))
        start = time.perf_counter()
        train(
            train_samples=noisy_train_b,
            val_samples=noisy_val_b,
            model_key="bert_small",
            config=config_b,
        )
        log.info("Variant B trained", time_s=f"{time.perf_counter() - start:.0f}")

    results["B_no_oral"] = _evaluate_model(
        variant_b_dir, gold_clean, "Variant B: No oral (CLEAN)", log,
    )
    results["B_no_oral_noisy"] = _evaluate_model(
        variant_b_dir, gold_noisy, "Variant B: No oral (NOISY)", log,
    )

    # =================================================================
    # Variant C: Partial entities filtered
    # 898 base WITH oral transforms + noise, partial entities REMOVED
    # Tests: does preserving partial entities help?
    # =================================================================
    log.info("=" * 60)
    log.info("Variant C: Partial entities filtered")

    train_c_base, val_c_base = _load_prepared_898()
    noisy_train_c = create_noisy_dataset(train_c_base, config=config)
    noisy_val_c = create_noisy_dataset(val_c_base, config=config)

    # Remove partial entities from training data
    for s in noisy_train_c:
        s["entities"] = [e for e in s["entities"] if not e.get("partial", False)]
    for s in noisy_val_c:
        s["entities"] = [e for e in s["entities"] if not e.get("partial", False)]
    sample_counts["C"] = len(noisy_train_c)

    variant_c_dir = ablation_dir / "variant_c_no_partial"
    config_c = {
        **config,
        "training": {
            **config["training"],
            "bert_small": {**bert_small_config, "save_dir": str(variant_c_dir)},
        },
    }

    if _model_exists(variant_c_dir):
        log.info("Variant C already trained, skipping", path=str(variant_c_dir))
    else:
        log.info("Training Variant C", samples=len(noisy_train_c))
        start = time.perf_counter()
        train(
            train_samples=noisy_train_c,
            val_samples=noisy_val_c,
            model_key="bert_small",
            config=config_c,
        )
        log.info("Variant C trained", time_s=f"{time.perf_counter() - start:.0f}")

    results["C_no_partial"] = _evaluate_model(
        variant_c_dir, gold_clean, "Variant C: No partial (CLEAN)", log,
    )
    results["C_no_partial_noisy"] = _evaluate_model(
        variant_c_dir, gold_noisy, "Variant C: No partial (NOISY)", log,
    )

    # =================================================================
    # Main model baseline for comparison
    # =================================================================
    log.info("=" * 60)
    log.info("Main model (full V3 pipeline) for comparison")

    main_dir = Path(config["training"]["bert_small"]["save_dir"])
    results["main"] = _evaluate_model(
        main_dir, gold_clean, "Main (all decisions, CLEAN)", log,
    )
    results["main_noisy"] = _evaluate_model(
        main_dir, gold_noisy, "Main (all decisions, NOISY)", log,
    )

    # =================================================================
    # Summary
    # =================================================================
    # Count main training samples from noisy dir
    with open("data/noisy/train_noisy.json") as f:
        main_train_count = len(json.load(f))

    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS (V3)")
    print("=" * 70)

    print(f"\n--- Clean test set (n={len(gold_clean)}) ---")
    print(f"  {'Variant':<50} {'Strict F1':>10} {'Partial F1':>11} {'Samples':>8}")
    print(f"  {'-'*81}")

    clean_rows = [
        ("Main (898 base + oral + noise + partial)", "main", main_train_count),
        ("A: No noise augmentation", "A_no_noise", sample_counts["A"]),
        ("B: No oral format transforms", "B_no_oral", sample_counts["B"]),
        ("C: Partial entities filtered", "C_no_partial", sample_counts["C"]),
    ]

    for label, key, n in clean_rows:
        sf1 = results[key]["strict"]["overall"]["f1"]
        pf1 = results[key]["partial"]["overall"]["f1"]
        print(f"  {label:<50} {sf1:>10.3f} {pf1:>11.3f} {n:>8}")

    print(f"\n--- Noisy test set (n={len(gold_noisy)}) ---")
    print(f"  {'Variant':<50} {'Strict F1':>10} {'Partial F1':>11}")
    print(f"  {'-'*73}")

    noisy_rows = [
        ("Main (all decisions)", "main_noisy"),
        ("A: No noise augmentation", "A_no_noise_noisy"),
        ("B: No oral format transforms", "B_no_oral_noisy"),
        ("C: Partial entities filtered", "C_no_partial_noisy"),
    ]

    for label, key in noisy_rows:
        sf1 = results[key]["strict"]["overall"]["f1"]
        pf1 = results[key]["partial"]["overall"]["f1"]
        print(f"  {label:<50} {sf1:>10.3f} {pf1:>11.3f}")

    print("\n" + "=" * 70)
    print("Notes:")
    print("- Variant A sample count confound (898 vs ~3592) cannot be eliminated.")
    print("- Val sets differ across variants due to different preprocessing.")
    print("- Results are approximate contribution estimates, not controlled experiments.")
    print("=" * 70)

    # Save
    output_path = Path("results/ablation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Ablation results saved", path=str(output_path))


if __name__ == "__main__":
    main()
