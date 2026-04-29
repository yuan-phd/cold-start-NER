"""ONNX export and INT8 quantization of BERT-Small for production inference.

Exports the trained BERT-Small NER model to ONNX format, applies INT8
dynamic quantization, benchmarks speed, and verifies F1 does not degrade.

Usage:
    python scripts/run_onnx.py
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from src.models.transformer_ner import TransformerNER, ID2LABEL
from src.evaluation.metrics import evaluate
from src.utils.config import load_config
from src.utils.logger import setup_logging, get_logger


def export_to_onnx(model_dir: Path, onnx_dir: Path) -> Path:
    """Export BERT NER model to ONNX format."""
    log = get_logger(__name__)
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "model.onnx"

    # Load PyTorch model
    ner = TransformerNER(model_dir)
    model = ner.model
    tokenizer = ner.tokenizer

    # Create dummy input
    dummy_text = "my name is john smith and my order is ORD-2024-5591"
    inputs = tokenizer(
        dummy_text, return_tensors="pt", truncation=True, max_length=256,
    )

    # Export
    model.eval()
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "token_type_ids": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
        },
        opset_version=14,
    )

    # Copy tokenizer files
    tokenizer.save_pretrained(str(onnx_dir))

    log.info("ONNX export complete", path=str(onnx_path))
    return onnx_path


def quantize_onnx(onnx_path: Path) -> Path:
    """Apply INT8 dynamic quantization to ONNX model."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    log = get_logger(__name__)
    quantized_path = onnx_path.parent / "model_quantized.onnx"

    quantize_dynamic(
        str(onnx_path),
        str(quantized_path),
        weight_type=QuantType.QInt8,
    )

    # Report size reduction
    orig_size = onnx_path.stat().st_size / (1024 * 1024)
    quant_size = quantized_path.stat().st_size / (1024 * 1024)
    log.info(
        "Quantization complete",
        original_mb=f"{orig_size:.1f}",
        quantized_mb=f"{quant_size:.1f}",
        reduction=f"{(1 - quant_size/orig_size)*100:.1f}%",
    )

    return quantized_path


class ONNXNERModel:
    """ONNX-based NER model with same interface as TransformerNER."""

    def __init__(self, onnx_path: Path, tokenizer_dir: Path, max_length: int = 256):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_dir), do_lower_case=True,
        )
        self.max_length = max_length

    def predict(self, text: str) -> list[dict[str, Any]]:
        """Extract entities — same interface as TransformerNER.predict()."""
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="np",
            truncation=True,
            max_length=self.max_length,
        )

        offset_mapping = encoding.pop("offset_mapping")[0].tolist()

        outputs = self.session.run(
            ["logits"],
            {
                "input_ids": encoding["input_ids"].astype(np.int64),
                "attention_mask": encoding["attention_mask"].astype(np.int64),
                "token_type_ids": encoding["token_type_ids"].astype(np.int64),
            },
        )

        logits = outputs[0][0]  # (seq_len, num_labels)
        probs = _softmax(logits)
        predictions = np.argmax(probs, axis=-1)
        confidences = np.max(probs, axis=-1)

        # Same entity extraction logic as TransformerNER.predict()
        entities = []
        current_entity = None
        prev_entity_type = None

        for pred_id, conf, (start, end) in zip(predictions, confidences, offset_mapping):
            if start == 0 and end == 0:
                prev_entity_type = None
                continue

            label = ID2LABEL[int(pred_id)]
            confidence = float(conf)

            if label == "O":
                if current_entity is not None:
                    current_entity["confidence"] = round(
                        current_entity["_conf_sum"] / current_entity["_tok_count"], 4
                    )
                    del current_entity["_conf_sum"]
                    del current_entity["_tok_count"]
                    entities.append(current_entity)
                    current_entity = None
                prev_entity_type = None
                continue

            entity_type = label[2:]

            is_new = False
            if label.startswith("B-"):
                is_new = True
            elif label.startswith("I-"):
                if prev_entity_type is None or prev_entity_type != entity_type:
                    is_new = True

            if is_new:
                if current_entity is not None:
                    current_entity["confidence"] = round(
                        current_entity["_conf_sum"] / current_entity["_tok_count"], 4
                    )
                    del current_entity["_conf_sum"]
                    del current_entity["_tok_count"]
                    entities.append(current_entity)
                current_entity = {
                    "text": text[start:end],
                    "label": entity_type,
                    "start": start,
                    "end": end,
                    "_conf_sum": confidence,
                    "_tok_count": 1,
                    "confidence": 0.0,
                }
            else:
                if current_entity is not None:
                    current_entity["end"] = end
                    current_entity["text"] = text[current_entity["start"]:end]
                    current_entity["_conf_sum"] += confidence
                    current_entity["_tok_count"] += 1

            prev_entity_type = entity_type

        if current_entity is not None:
            current_entity["confidence"] = round(
                current_entity["_conf_sum"] / current_entity["_tok_count"], 4
            )
            del current_entity["_conf_sum"]
            del current_entity["_tok_count"]
            entities.append(current_entity)

        return entities


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numpy softmax along last axis."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def measure_speed(predict_fn, texts: list[str], n_runs: int = 5) -> float:
    """Measure average ms/sample."""
    # Warmup
    for t in texts[:3]:
        predict_fn(t)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        for t in texts:
            predict_fn(t)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed / len(texts))
    return sum(times) / len(times)


def main() -> None:
    setup_logging()
    log = get_logger(__name__)
    config = load_config()

    model_dir = Path(config["training"]["bert_small"]["save_dir"])
    onnx_dir = Path("results/models/bert_small_onnx")

    # Step 1: Export
    log.info("Step 1: Exporting BERT-Small to ONNX")
    onnx_path = export_to_onnx(model_dir, onnx_dir)

    # Step 2: Quantize
    log.info("Step 2: INT8 dynamic quantization")
    quant_path = quantize_onnx(onnx_path)

    # Step 3: Load models for comparison
    log.info("Step 3: Loading models for comparison")
    pytorch_model = TransformerNER(model_dir)
    onnx_fp32 = ONNXNERModel(onnx_path, onnx_dir)
    onnx_int8 = ONNXNERModel(quant_path, onnx_dir)

    # Step 4: Verify F1 equivalence on gold set
    log.info("Step 4: Verifying F1 on gold test set")
    eval_dir = Path(config["evaluation"]["eval_data_dir"])
    with open(eval_dir / "gold_test.json") as f:
        gold = json.load(f)
    gold_clean = [s for s in gold if not s.get("noisy", False)]

    results = {}
    for name, model in [
        ("pytorch", pytorch_model),
        ("onnx_fp32", onnx_fp32),
        ("onnx_int8", onnx_int8),
    ]:
        preds = [model.predict(s["text"]) for s in gold_clean]
        golds = [s["entities"] for s in gold_clean]
        strict = evaluate(preds, golds, mode="strict")
        results[name] = strict["overall"]["f1"]
        log.info(f"F1 ({name})", strict_f1=strict["overall"]["f1"])

    # Step 5: Speed benchmark
    log.info("Step 5: Speed benchmark")
    import random
    rng = random.Random(42)
    train_path = Path("data/synthetic/prepared/train_prepared.json")
    bench_texts = []
    if train_path.exists():
        with open(train_path) as f:
            train_data = json.load(f)
        bench_texts.extend([s["text"] for s in rng.sample(train_data, min(25, len(train_data)))])

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

    speed = {}
    for name, model in [
        ("pytorch", pytorch_model),
        ("onnx_fp32", onnx_fp32),
        ("onnx_int8", onnx_int8),
    ]:
        ms = measure_speed(model.predict, bench_texts)
        speed[name] = round(ms, 2)
        log.info(f"Speed ({name})", ms_per_sample=f"{ms:.2f}")

    # Summary
    print("\n" + "=" * 70)
    print("ONNX QUANTIZATION RESULTS")
    print("=" * 70)

    orig_mb = onnx_path.stat().st_size / (1024 * 1024)
    quant_mb = quant_path.stat().st_size / (1024 * 1024)

    print(f"\n{'Model':<20} {'Strict F1':>10} {'ms/sample':>10} {'Size (MB)':>10}")
    print("-" * 52)
    print(f"{'PyTorch':<20} {results['pytorch']:>10.3f} {speed['pytorch']:>10.2f} {'N/A':>10}")
    print(f"{'ONNX FP32':<20} {results['onnx_fp32']:>10.3f} {speed['onnx_fp32']:>10.2f} {orig_mb:>10.1f}")
    print(f"{'ONNX INT8':<20} {results['onnx_int8']:>10.3f} {speed['onnx_int8']:>10.02f} {quant_mb:>10.1f}")
    print(f"\nSize reduction: {(1 - quant_mb/orig_mb)*100:.1f}%")
    print(f"Speed improvement (INT8 vs PyTorch): {(1 - speed['onnx_int8']/speed['pytorch'])*100:.1f}%")

    f1_diff = abs(results["pytorch"] - results["onnx_int8"])
    if f1_diff < 0.01:
        print(f"F1 degradation: {f1_diff:.4f} (negligible)")
    else:
        print(f"F1 degradation: {f1_diff:.4f} (WARNING: significant)")

    print("=" * 70)

    # Save results
    output = {
        "f1": results,
        "speed_ms": speed,
        "size_mb": {"onnx_fp32": round(orig_mb, 1), "onnx_int8": round(quant_mb, 1)},
        "size_reduction_pct": round((1 - quant_mb / orig_mb) * 100, 1),
    }
    with open(Path("results/onnx_results.json"), "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results saved", path="results/onnx_results.json")


if __name__ == "__main__":
    main()