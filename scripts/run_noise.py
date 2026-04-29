"""Apply ASR noise augmentation to prepared data.

Pipeline step 3: prepared clean data → noise augmentation → ready for training.
"""

import json
from pathlib import Path

from src.data.noise import create_noisy_dataset
from src.data.validate import validate_dataset, print_report
from src.utils.config import load_config
from src.utils.logger import setup_logging, get_logger


def main() -> None:
    setup_logging()
    log = get_logger(__name__)
    config = load_config()

    data_dir = Path(config["data_generation"]["output_dir"])
    prepared_dir = data_dir / "prepared"
    noisy_dir = Path(config["noise"]["output_dir"])
    noisy_dir.mkdir(parents=True, exist_ok=True)

    # --- Load prepared data ---
    log.info("Loading prepared data")
    with open(prepared_dir / "train_prepared.json") as f:
        train_samples = json.load(f)
    with open(prepared_dir / "val_prepared.json") as f:
        val_samples = json.load(f)

    log.info("Prepared data loaded", train=len(train_samples), val=len(val_samples))

    # --- Apply noise augmentation ---
    log.info("Applying noise augmentation (4 levels: clean, mild, moderate, severe)")
    noisy_train = create_noisy_dataset(train_samples, config=config)
    noisy_val = create_noisy_dataset(val_samples, config=config)

    # --- Validate ---
    log.info("Validating noisy data")
    report = validate_dataset(noisy_train)
    print_report(report)

    # --- Save ---
    log.info("Saving noisy data")
    with open(noisy_dir / "train_noisy.json", "w") as f:
        json.dump(noisy_train, f, indent=2)
    with open(noisy_dir / "val_noisy.json", "w") as f:
        json.dump(noisy_val, f, indent=2)

    log.info(
        "Noise augmentation complete",
        train=len(noisy_train),
        val=len(noisy_val),
        output_dir=str(noisy_dir),
    )


if __name__ == "__main__":
    main()