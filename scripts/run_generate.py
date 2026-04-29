"""Run synthetic data generation pipeline."""

from pathlib import Path

from dotenv import load_dotenv

from src.data.generate import generate_samples, generate_negative_samples
from src.utils.config import load_config
from src.utils.logger import setup_logging, get_logger

load_dotenv()


def main() -> None:
    setup_logging()
    log = get_logger(__name__)
    config = load_config()

    gen_config = config["data_generation"]
    output_dir = Path(gen_config["output_dir"])

    # --- Generate training data ---
    log.info("Phase 1: Generating training samples")
    generate_samples(
        n_samples=gen_config["train_samples"],
        output_path=output_dir / "train.json",
        config=config,
    )

    # --- Generate validation data ---
    log.info("Phase 2: Generating validation samples")
    generate_samples(
        n_samples=gen_config["val_samples"],
        output_path=output_dir / "val.json",
        config=config,
    )

    # --- Generate negative samples (no entities) ---
    log.info("Phase 3: Generating negative samples")
    n_negative_train = int(gen_config["train_samples"] * 0.1)  # 10% negatives
    n_negative_val = int(gen_config["val_samples"] * 0.1)
    generate_negative_samples(n_negative_train, output_dir / "train_negative.json")
    generate_negative_samples(n_negative_val, output_dir / "val_negative.json")

    log.info("Data generation complete", output_dir=str(output_dir))


if __name__ == "__main__":
    main()