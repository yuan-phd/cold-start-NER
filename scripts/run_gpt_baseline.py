"""GPT-4o-mini zero-shot NER baseline on gold test set.

Directly answers the assignment's motivating question: how much does a
lightweight local system gain over calling an LLM directly?

Tracks offset correction rate for transparency.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from src.data.generate import _validate_sample, _try_fix_offsets
from src.evaluation.metrics import evaluate
from src.utils.config import load_config
from src.utils.logger import setup_logging, get_logger

SYSTEM_PROMPT = """You are a Named Entity Recognition system for customer service transcripts.

Extract entities from the given text. Entity types:
- NAME: customer or person names
- EMAIL: email addresses (may be in oral format like "john at gmail dot com")
- CONTRACT_ID: order numbers, reference IDs, account numbers (may be in oral format like "CT dash 55123")
- PRODUCT: product or service names
- ISSUE_DATE: dates or time references (absolute like "February 3rd" or relative like "last Monday")

Return ONLY valid JSON with this exact format, no markdown or explanation:
{"text": "<the input text>", "entities": [{"text": "<entity text>", "label": "<TYPE>", "start": <int>, "end": <int>}]}

CRITICAL: text[start:end] must exactly equal the entity text. Count characters carefully.
If no entities are found, return: {"text": "<the input text>", "entities": []}"""


def run_gpt_baseline(
    gold_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Run GPT-4o-mini zero-shot NER on gold test set."""
    log = get_logger(__name__)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    with open(gold_path) as f:
        gold_samples = json.load(f)

    gold_clean = [s for s in gold_samples if not s.get("noisy", False)]
    gold_noisy = [s for s in gold_samples if s.get("noisy", False)]

    stats = {
        "total": 0,
        "valid_direct": 0,
        "auto_fixed": 0,
        "failed": 0,
        "errors": 0,
        "total_latency_ms": 0,
    }

    predictions = []
    clean_preds = []
    noisy_preds = []

    for sample in tqdm(gold_samples, desc="GPT-4o-mini NER"):
        stats["total"] += 1

        try:
            start_time = time.perf_counter()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": sample["text"]},
                ],
                temperature=0.0,
                max_tokens=500,
            )
            latency_ms = (time.perf_counter() - start_time) * 1000
            stats["total_latency_ms"] += latency_ms

            raw = response.choices[0].message.content.strip()

            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            result = json.loads(raw)

            # Validate offsets
            is_valid, _ = _validate_sample(result)

            if is_valid:
                stats["valid_direct"] += 1
                pred_entities = result.get("entities", [])
            else:
                fixed = _try_fix_offsets(result)
                if fixed:
                    is_valid, _ = _validate_sample(fixed)
                    if is_valid:
                        stats["auto_fixed"] += 1
                        pred_entities = fixed.get("entities", [])
                    else:
                        stats["failed"] += 1
                        pred_entities = []
                        log.debug("GPT sample failed after fix", text=sample["text"][:60])
                else:
                    stats["failed"] += 1
                    pred_entities = []

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            stats["errors"] += 1
            pred_entities = []
            log.debug("GPT parse error", error=str(e))
        except Exception as e:
            stats["errors"] += 1
            pred_entities = []
            log.warning("GPT API error", error=str(e))

        predictions.append(pred_entities)
        if sample.get("noisy", False):
            noisy_preds.append(pred_entities)
        else:
            clean_preds.append(pred_entities)

    # Evaluate
    all_gold = [s["entities"] for s in gold_samples]
    clean_gold = [s["entities"] for s in gold_clean]
    noisy_gold = [s["entities"] for s in gold_noisy]

    results = {
        "overall": {
            "strict": evaluate(predictions, all_gold, mode="strict"),
            "partial": evaluate(predictions, all_gold, mode="partial", iou_threshold=0.5),
        },
        "clean": {
            "strict": evaluate(clean_preds, clean_gold, mode="strict"),
            "partial": evaluate(clean_preds, clean_gold, mode="partial", iou_threshold=0.5),
        },
    }

    if gold_noisy:
        results["noisy"] = {
            "strict": evaluate(noisy_preds, noisy_gold, mode="strict"),
            "partial": evaluate(noisy_preds, noisy_gold, mode="partial", iou_threshold=0.5),
        }

    avg_latency = stats["total_latency_ms"] / max(stats["total"], 1)
    stats["avg_latency_ms"] = round(avg_latency, 1)

    # Summary
    print("\n" + "=" * 70)
    print("GPT-4o-mini ZERO-SHOT NER BASELINE")
    print("=" * 70)

    print("\n--- Generation Stats ---")
    print(f"  Total samples: {stats['total']}")
    print(f"  Valid offsets (direct): {stats['valid_direct']} ({stats['valid_direct']/max(stats['total'],1)*100:.1f}%)")
    print(f"  Auto-fixed offsets: {stats['auto_fixed']} ({stats['auto_fixed']/max(stats['total'],1)*100:.1f}%)")
    print(f"  Failed: {stats['failed']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Avg latency: {avg_latency:.0f} ms/sample")

    print(f"\n--- Results (Gold Clean, n={len(gold_clean)}) ---")
    clean_sf1 = results["clean"]["strict"]["overall"]["f1"]
    clean_pf1 = results["clean"]["partial"]["overall"]["f1"]
    print(f"  Strict F1: {clean_sf1:.3f}")
    print(f"  Partial F1: {clean_pf1:.3f}")

    if "noisy" in results:
        print(f"\n--- Results (Gold Noisy, n={len(gold_noisy)}) ---")
        noisy_sf1 = results["noisy"]["strict"]["overall"]["f1"]
        noisy_pf1 = results["noisy"]["partial"]["overall"]["f1"]
        print(f"  Strict F1: {noisy_sf1:.3f}")
        print(f"  Partial F1: {noisy_pf1:.3f}")

    print("\n--- Comparison ---")
    print(f"  {'Model':<25} {'Clean Strict F1':>15} {'ms/sample':>10}")
    print(f"  {'-'*52}")
    print(f"  {'GPT-4o-mini (zero-shot)':<25} {clean_sf1:>15.3f} {avg_latency:>10.0f}")
    print(f"  {'Ensemble (local)':<25} {'0.854':>15} {'4.0':>10}")
    print(f"  {'BERT-Small (local)':<25} {'0.739':>15} {'3.4':>10}")
    print("=" * 70)

    # Save
    output = {
        "stats": stats,
        "results": results,
        "predictions": [
            {"text": s["text"], "predicted": p, "gold": s["entities"]}
            for s, p in zip(gold_samples, predictions)
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("GPT baseline results saved", path=str(output_path))

    return output


def main() -> None:
    setup_logging()
    config = load_config()
    eval_dir = Path(config["evaluation"]["eval_data_dir"])

    run_gpt_baseline(
        gold_path=eval_dir / "gold_test.json",
        output_path=Path("results/gpt_baseline_results.json"),
    )


if __name__ == "__main__":
    main()