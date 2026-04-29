"""Diagnostic script — systematic validation of data, training, and inference.

Run this after each pipeline stage to catch issues early.
Saves all diagnostic results to results/diagnostics/.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from src.data.validate import validate_dataset
from src.models.transformer_ner import (
    NERDataset, TransformerNER, ID2LABEL,
)
from src.models import rules
from src.utils.config import load_config
from src.utils.logger import setup_logging, get_logger


def diagnose_data(config: dict[str, Any], log: Any) -> dict[str, Any]:
    """Diagnose data quality across all pipeline stages."""
    results = {}

    # Check each data file
    data_files = {
        "synthetic_train": "data/synthetic/train.json",
        "synthetic_val": "data/synthetic/val.json",
        "oral_train": "data/synthetic/train_oral.json",
        "prepared_train": "data/synthetic/prepared/train_prepared.json",
        "prepared_val": "data/synthetic/prepared/val_prepared.json",
        "noisy_train": "data/noisy/train_noisy.json",
        "noisy_val": "data/noisy/val_noisy.json",
        "gold_test": "data/eval/gold_test.json",
    }

    for name, path in data_files.items():
        p = Path(path)
        if not p.exists():
            results[name] = {"status": "MISSING"}
            continue

        with open(p) as f:
            data = json.load(f)

        report = validate_dataset(data)
        results[name] = {
            "status": "OK" if report["invalid_samples"] == 0 else "HAS_ERRORS",
            "total": report["total_samples"],
            "valid": report["valid_samples"],
            "invalid": report["invalid_samples"],
            "entities": report["total_entities"],
            "entity_types": report["entity_type_counts"],
        }

    log.info("Data diagnosis complete")
    return results


def diagnose_tokenizer(config: dict[str, Any], log: Any) -> dict[str, Any]:
    """Check tokenizer behavior on representative samples."""
    results = {}

    for model_key in ["bert_tiny", "bert_small"]:
        model_name = config["training"][model_key]["model_name"]

        test_texts = [
            "my name is Jennifer Taylor",
            "email is john.smith@gmail.com",
            "order ORD-2024-5591",
            "john at gmail dot com",
            "CT dash 55123",
            "last tuesday about a week ago",
            "the Sony WH-1000XM5 headphones",
            "Priya Patel thats p r i y a",
        ]

        log.info("Loading tokenizer WITH do_lower_case=True (training config)")
        tokenizer_lower = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)

        unk_count_lower = 0
        unk_examples_lower = []
        for text in test_texts:
            tokens = tokenizer_lower.tokenize(text)
            for tok in tokens:
                if tok == "[UNK]":
                    unk_count_lower += 1
                    unk_examples_lower.append({"text": text, "tokens": tokens})
                    break

        log.info("Loading tokenizer WITHOUT do_lower_case (diagnostic baseline for comparison)")
        tokenizer_base = AutoTokenizer.from_pretrained(model_name)

        unk_count_base = 0
        unk_examples_base = []
        for text in test_texts:
            tokens = tokenizer_base.tokenize(text)
            for tok in tokens:
                if tok == "[UNK]":
                    unk_count_base += 1
                    unk_examples_base.append({"text": text, "tokens": tokens})
                    break

        results[model_key] = {
            "with_lower": {
                "vocab_size": tokenizer_lower.vocab_size,
                "unk_rate": f"{unk_count_lower}/{len(test_texts)} texts have [UNK]",
                "unk_examples": unk_examples_lower[:5],
                "note": "This is the training configuration",
            },
            "without_lower": {
                "vocab_size": tokenizer_base.vocab_size,
                "unk_rate": f"{unk_count_base}/{len(test_texts)} texts have [UNK]",
                "unk_examples": unk_examples_base[:5],
                "note": "Diagnostic baseline — shows UNK rate without the do_lower_case fix",
            },
        }

    log.info("Tokenizer diagnosis complete")
    return results


def diagnose_label_distribution(config: dict[str, Any], log: Any) -> dict[str, Any]:
    """Analyze BIO label distribution in encoded training data."""
    results = {}

    noisy_path = Path("data/noisy/train_noisy.json")
    if not noisy_path.exists():
        return {"status": "MISSING noisy training data"}

    with open(noisy_path) as f:
        data = json.load(f)

    for model_key in ["bert_tiny", "bert_small"]:
        model_name = config["training"][model_key]["model_name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Encode first 200 samples
        dataset = NERDataset(data[:200], tokenizer, max_length=128)

        label_counts = Counter()
        for i in range(len(dataset)):
            labels = dataset[i]["labels"]
            for label_id in labels.tolist():
                if label_id == -100:
                    continue
                label_counts[ID2LABEL[label_id]] += 1

        total = sum(label_counts.values())
        distribution = {
            label: {"count": count, "pct": f"{count/total*100:.1f}%"}
            for label, count in label_counts.most_common()
        }

        # Check B vs I ratio
        b_total = sum(c for tag, c in label_counts.items() if tag.startswith("B-"))
        i_total = sum(c for tag, c in label_counts.items() if tag.startswith("I-"))

        results[model_key] = {
            "distribution": distribution,
            "b_total": b_total,
            "i_total": i_total,
            "b_i_ratio": f"{b_total}:{i_total}" if i_total > 0 else "N/A",
            "o_pct": f"{label_counts.get('O', 0)/total*100:.1f}%",
        }

    log.info("Label distribution diagnosis complete")
    return results


def diagnose_model_inference(config: dict[str, Any], log: Any) -> dict[str, Any]:
    """Test trained models on representative inputs."""
    results = {}

    test_cases = [
        {
            "text": "my name is John Smith and my email is john.smith@gmail.com",
            "expected": ["NAME", "EMAIL"],
        },
        {
            "text": "order number ORD-2024-5591 from last tuesday",
            "expected": ["CONTRACT_ID", "ISSUE_DATE"],
        },
        {
            "text": "i bought the iPhone 15 Pro about a week ago",
            "expected": ["PRODUCT", "ISSUE_DATE"],
        },
        {
            "text": "hi uh my email is john at gmail dot com",
            "expected": ["EMAIL"],
        },
        {
            "text": "please transfer me to someone else",
            "expected": [],
        },
    ]

    for model_key in ["bert_tiny", "bert_small"]:
        model_dir = Path(config["training"][model_key]["save_dir"])
        if not model_dir.exists():
            results[model_key] = {"status": "NOT TRAINED"}
            continue

        model = TransformerNER(model_dir)
        model_results = []

        for tc in test_cases:
            preds = model.predict(tc["text"])
            pred_labels = [e["label"] for e in preds]

            # Also check raw logits for B vs I
            encoding = model.tokenizer(
                tc["text"],
                return_offsets_mapping=True,
                return_tensors="pt",
                truncation=True,
                max_length=128,
            )
            offset_mapping = encoding.pop("offset_mapping")[0].tolist()

            with torch.no_grad():
                outputs = model.model(**encoding)
                raw_preds = torch.argmax(outputs.logits, dim=-1)[0]

            raw_labels = [
                ID2LABEL[p.item()]
                for p, (s, e) in zip(raw_preds, offset_mapping)
                if not (s == 0 and e == 0)
            ]
            raw_non_o = [tag for tag in raw_labels if tag != "O"]

            model_results.append({
                "text": tc["text"][:60],
                "expected": tc["expected"],
                "predicted_entities": pred_labels,
                "raw_non_o_labels": raw_non_o,
                "match": set(pred_labels) == set(tc["expected"]),
            })

        results[model_key] = {
            "test_cases": model_results,
            "pass_rate": f"{sum(1 for r in model_results if r['match'])}/{len(model_results)}",
        }

    # Also test rules baseline
    rules_results = []
    for tc in test_cases:
        preds = rules.predict(tc["text"])
        pred_labels = [e["label"] for e in preds]
        rules_results.append({
            "text": tc["text"][:60],
            "expected": tc["expected"],
            "predicted": pred_labels,
        })
    results["rules"] = {"test_cases": rules_results}

    log.info("Model inference diagnosis complete")
    return results


def main() -> None:
    setup_logging()
    log = get_logger(__name__)
    config = load_config()

    output_dir = Path("results/diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Starting full diagnostic pipeline")

    all_results = {}

    # 1. Data diagnosis
    log.info("=" * 50)
    log.info("Phase 1: Data Diagnosis")
    all_results["data"] = diagnose_data(config, log)

    # 2. Tokenizer diagnosis
    log.info("=" * 50)
    log.info("Phase 2: Tokenizer Diagnosis")
    all_results["tokenizer"] = diagnose_tokenizer(config, log)

    # 3. Label distribution
    log.info("=" * 50)
    log.info("Phase 3: Label Distribution Diagnosis")
    all_results["labels"] = diagnose_label_distribution(config, log)

    # 4. Model inference
    log.info("=" * 50)
    log.info("Phase 4: Model Inference Diagnosis")
    all_results["inference"] = diagnose_model_inference(config, log)

    # Save full report
    with open(output_dir / "diagnostic_report.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 65)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 65)

    print("\n--- Data Files ---")
    for name, info in all_results["data"].items():
        status = info.get("status", "?")
        total = info.get("total", "?")
        print(f"  {name:<25} {status:<12} samples={total}")

    print("\n--- Tokenizer ---")
    for model_key, info in all_results["tokenizer"].items():
        without = info["without_lower"]
        with_lower = info["with_lower"]
        print(f"  {model_key} (WITHOUT do_lower_case — diagnostic baseline):")
        print(f"  UNK rate: {without['unk_rate']}")
        for ex in without.get("unk_examples", []):
            print(f"    [UNK] in: \"{ex['text']}\" → {ex['tokens']}")
        print(f"  {model_key} (WITH do_lower_case — training config):")
        print(f"  UNK rate: {with_lower['unk_rate']}")
        for ex in with_lower.get("unk_examples", []):
            print(f"    [UNK] in: \"{ex['text']}\" → {ex['tokens']}")

    print("\n--- Label Distribution (first 200 samples) ---")
    for model_key, info in all_results["labels"].items():
        print(f"  {model_key}: B={info['b_total']} I={info['i_total']} ratio={info['b_i_ratio']} O={info['o_pct']}")

    print("\n--- Model Inference ---")
    for model_key, info in all_results["inference"].items():
        if "pass_rate" in info:
            print(f"  {model_key}: pass_rate={info['pass_rate']}")
        for tc in info.get("test_cases", []):
            status = "✓" if tc.get("match", False) else "✗"
            raw = tc.get("raw_non_o_labels", [])
            print(f"    {status} \"{tc['text']}\"")
            print(f"      expected={tc['expected']} predicted={tc.get('predicted_entities', tc.get('predicted', []))}")
            if raw:
                print(f"      raw_labels={raw}")

    print("=" * 65)

    log.info("Diagnostic report saved", path=str(output_dir / "diagnostic_report.json"))


if __name__ == "__main__":
    main()