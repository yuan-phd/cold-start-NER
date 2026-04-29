import argparse
import json
import sys
import time
from pathlib import Path

from src.models import rules
from src.models.transformer_ner import TransformerNER
from src.models.ensemble import predict as ensemble_predict
from src.utils.config import load_config
from src.utils.logger import setup_logging


def load_model(model_name: str, config: dict):
    """Load a model by name and return a predict function."""
    if model_name == "rules":
        return rules.predict

    elif model_name == "bert-tiny":
        model_dir = config["training"]["bert_tiny"]["save_dir"]
        if not Path(model_dir).exists():
            print(f"Error: BERT-Tiny not found at {model_dir}. Run 'make train' first.")
            sys.exit(1)
        model = TransformerNER(model_dir)
        return model.predict

    elif model_name == "bert-small":
        model_dir = config["training"]["bert_small"]["save_dir"]
        if not Path(model_dir).exists():
            print(f"Error: BERT-Small not found at {model_dir}. Run 'make train' first.")
            sys.exit(1)
        model = TransformerNER(model_dir)
        return model.predict

    elif model_name == "ensemble":
        model_dir = config["training"]["bert_small"]["save_dir"]
        if not Path(model_dir).exists():
            print(f"Error: BERT-Small not found at {model_dir}. Run 'make train' first.")
            sys.exit(1)
        transformer = TransformerNER(model_dir)

        def _predict(text: str):
            entities, _ = ensemble_predict(text, transformer)
            return entities

        return _predict

    else:
        print(f"Error: Unknown model '{model_name}'. Choose from: rules, bert-tiny, bert-small, ensemble")
        sys.exit(1)


def format_output(text: str, entities: list[dict], elapsed_ms: float, output_format: str) -> str:
    """Format entities for display."""
    if output_format == "json":
        return json.dumps({
            "text": text,
            "entities": entities,
            "latency_ms": round(elapsed_ms, 1),
        }, indent=2)

    # Pretty print
    lines = [f"Text: {text}", f"Entities ({len(entities)}):"]
    if not entities:
        lines.append("  (none)")
    for e in entities:
        conf = f"confidence={e['confidence']:.2f}" if "confidence" in e else ""
        partial = f" partial={e['partial']}" if e.get("partial") else ""
        lines.append(f"  {e['label']}: \"{e['text']}\" [{e['start']}:{e['end']}] {conf}{partial}")
    lines.append(f"Latency: {elapsed_ms:.1f}ms")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Extract named entities from customer service text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/run_inference.py --text "my name is sarah and my order is ORD-2024-5591"
  python scripts/run_inference.py --text "..." --model bert-small --format json
  python scripts/run_inference.py --file input.txt --model ensemble
""",
    )
    parser.add_argument("--text", type=str, help="Input text to process")
    parser.add_argument("--file", type=str, help="Path to text file (one text per line)")
    parser.add_argument(
        "--model",
        type=str,
        default="ensemble",
        choices=["rules", "bert-tiny", "bert-small", "ensemble"],
        help="Model to use (default: ensemble)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pretty",
        choices=["pretty", "json"],
        help="Output format (default: pretty)",
    )

    args = parser.parse_args()

    if not args.text and not args.file:
        parser.error("Either --text or --file is required")

    setup_logging()
    config = load_config()
    predict_fn = load_model(args.model, config)

    texts = []
    if args.text:
        texts.append(args.text)
    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        texts = [line.strip() for line in file_path.read_text().splitlines() if line.strip()]

    for text in texts:
        start_time = time.perf_counter()
        entities = predict_fn(text)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        print(format_output(text, entities, elapsed_ms, args.format))
        if len(texts) > 1:
            print("---")


if __name__ == "__main__":
    main()