"""Run model training pipeline.

Pipeline step 4: loads noise-augmented data, trains BERT-Tiny and BERT-Small.
"""

import json
from pathlib import Path

from src.models.transformer_ner import train
from src.utils.config import load_config
from src.utils.logger import setup_logging, get_logger


def main() -> None:
    setup_logging()
    log = get_logger(__name__)
    config = load_config()

    noisy_dir = Path(config["noise"]["output_dir"])

    # --- Load prepared + noisy data ---
    log.info("Loading noise-augmented training data")
    with open(noisy_dir / "train_noisy.json") as f:
        noisy_train = json.load(f)
    with open(noisy_dir / "val_noisy.json") as f:
        noisy_val = json.load(f)

    log.info("Data loaded", train=len(noisy_train), val=len(noisy_val))

    # --- Train BERT-Tiny ---
    log.info("Training BERT-Tiny")
    tiny_history = train(
        train_samples=noisy_train,
        val_samples=noisy_val,
        model_key="bert_tiny",
        config=config,
    )

    # --- Train BERT-Small ---
    log.info("Training BERT-Small")
    small_history = train(
        train_samples=noisy_train,
        val_samples=noisy_val,
        model_key="bert_small",
        config=config,
    )

    log.info(
        "All training complete",
        bert_tiny_best_val_loss=f"{tiny_history['best_val_loss']:.4f}",
        bert_small_best_val_loss=f"{small_history['best_val_loss']:.4f}",
    )


if __name__ == "__main__":
    main()