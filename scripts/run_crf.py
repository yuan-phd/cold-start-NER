"""Train and evaluate CRF baseline for comparison with BERT models.

Trains a CRF with hand-crafted features on the same noisy training data
used for BERT, then evaluates on the gold test set using the same metrics.

Output:
    results/crf_results.json — evaluation results and training stats

Usage:
    python scripts/run_crf.py
"""

import json
import time
from pathlib import Path

from src.models.crf_ner import CRFModel
from src.evaluation.metrics import evaluate, format_report
from src.utils.config import load_config
from src.utils.logger import setup_logging, get_logger


def main() -> None:
    setup_logging()
    log = get_logger(__name__)
    config = load_config()

    # Load training data (same noisy data as BERT)
    noisy_dir = Path(config["noise"]["output_dir"])
    with open(noisy_dir / "train_noisy.json") as f:
        train_data = json.load(f)
    with open(noisy_dir / "val_noisy.json") as f:
        val_data = json.load(f)

    log.info("Data loaded", train=len(train_data), val=len(val_data))

    # Train CRF
    crf = CRFModel()
    log.info("Training CRF baseline")
    start_time = time.perf_counter()
    train_stats = crf.train(train_data, val_data)
    train_time = time.perf_counter() - start_time
    train_stats["train_time_seconds"] = round(train_time, 1)
    log.info("CRF trained", time_s=f"{train_time:.1f}")

    # Load gold test set
    eval_dir = Path(config["evaluation"]["eval_data_dir"])
    with open(eval_dir / "gold_test.json") as f:
        gold = json.load(f)
    gold_clean = [s for s in gold if not s.get("noisy", False)]
    gold_noisy = [s for s in gold if s.get("noisy", False)]

    # Evaluate on gold clean
    log.info("Evaluating CRF on gold clean set", samples=len(gold_clean))
    preds_clean = [crf.predict(s["text"]) for s in gold_clean]
    golds_clean = [s["entities"] for s in gold_clean]

    strict_clean = evaluate(preds_clean, golds_clean, mode="strict")
    partial_clean = evaluate(preds_clean, golds_clean, mode="partial", iou_threshold=0.5)

    # Evaluate on gold noisy
    noisy_results = None
    if gold_noisy:
        log.info("Evaluating CRF on gold noisy subset", samples=len(gold_noisy))
        preds_noisy = [crf.predict(s["text"]) for s in gold_noisy]
        golds_noisy = [s["entities"] for s in gold_noisy]
        noisy_results = {
            "strict": evaluate(preds_noisy, golds_noisy, mode="strict"),
            "partial": evaluate(preds_noisy, golds_noisy, mode="partial", iou_threshold=0.5),
        }

    # Speed benchmark
    log.info("Benchmarking CRF speed")
    import random
    rng = random.Random(42)

    bench_texts = []
    train_path = Path("data/synthetic/prepared/train_prepared.json")
    if train_path.exists():
        with open(train_path) as f:
            prep_data = json.load(f)
        bench_texts.extend([s["text"] for s in rng.sample(prep_data, min(25, len(prep_data)))])

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

    # Warmup
    for t in bench_texts[:3]:
        crf.predict(t)

    n_runs = 3
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        for t in bench_texts:
            crf.predict(t)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed / len(bench_texts))
    avg_ms = sum(times) / len(times)

    # Compile results
    results = {
        "training": train_stats,
        "clean": {
            "strict": strict_clean,
            "partial": partial_clean,
        },
        "speed_ms_per_sample": round(avg_ms, 2),
    }
    if noisy_results:
        results["noisy"] = noisy_results

    # Save
    output_path = Path("results/crf_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 70)
    print("CRF BASELINE RESULTS")
    print("=" * 70)
    print(f"Training time: {train_stats['train_time_seconds']}s on {train_stats['train_sequences']} sequences")
    print(f"Speed: {avg_ms:.2f} ms/sample")
    print()

    print("--- Gold Clean Set ---")
    print(format_report(strict_clean))

    print("\n--- Per-Entity (Strict F1) ---")
    for etype, data in sorted(strict_clean["per_entity_type"].items()):
        print(f"  {etype:<15} F1={data['f1']:.3f} P={data['precision']:.3f} R={data['recall']:.3f} (n={data['support']})")

    if noisy_results:
        print("\n--- Gold Noisy Subset ---")
        print(f"  Strict F1:  {noisy_results['strict']['overall']['f1']:.3f}")
        print(f"  Partial F1: {noisy_results['partial']['overall']['f1']:.3f}")
        print("\n--- Per-Entity Noisy (Strict F1) ---")
        for etype, data in sorted(noisy_results["strict"]["per_entity_type"].items()):
            print(f"  {etype:<15} F1={data['f1']:.3f} P={data['precision']:.3f} R={data['recall']:.3f} (n={data['support']})")

    # Comparison with other models (loaded from eval results)
    print("\n--- Comparison ---")
    print(f"  {'Model':<15} {'Clean Strict F1':>16} {'Noisy Strict F1':>16} {'ms/sample':>10}")
    print(f"  {'-'*59}")
    print(f"  {'CRF':<15} {strict_clean['overall']['f1']:>16.3f}", end="")
    if noisy_results:
        print(f" {noisy_results['strict']['overall']['f1']:>16.3f}", end="")
    else:
        print(f" {'—':>16}", end="")
    print(f" {avg_ms:>10.2f}")

    eval_report_path = Path("results/eval_report.json")
    if eval_report_path.exists():
        with open(eval_report_path) as f:
            eval_data = json.load(f)
        for model_name in ["bert-small", "ensemble"]:
            if model_name in eval_data.get("model_results", {}):
                clean_f1 = eval_data["model_results"][model_name]["strict"]["overall"]["f1"]
                noisy_f1 = eval_data.get("noise_results", {}).get(model_name, {}).get("strict", {}).get("overall", {}).get("f1", "—")
                speed = eval_data.get("speed_results", {}).get(model_name, "—")
                noisy_str = f"{noisy_f1:>16.3f}" if isinstance(noisy_f1, float) else f"{'—':>16}"
                speed_str = f"{speed:>10.2f}" if isinstance(speed, float) else f"{'—':>10}"
                print(f"  {model_name:<15} {clean_f1:>16.3f} {noisy_str} {speed_str}")

    print("=" * 70)
    log.info("CRF results saved", path=str(output_path))


if __name__ == "__main__":
    main()
