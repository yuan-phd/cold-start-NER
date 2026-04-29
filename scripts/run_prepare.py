"""Prepare training data: merge all sources and apply format transforms.

Pipeline step 2: raw synthetic + oral + negatives → format transforms → merged clean data.
"""

import json
from pathlib import Path

from src.data.noise import apply_oral_format_transforms
from src.data.validate import validate_sample, validate_dataset, print_report
from src.utils.config import load_config
from src.utils.logger import setup_logging, get_logger


def main() -> None:
    setup_logging()
    log = get_logger(__name__)
    config = load_config()

    data_dir = Path(config["data_generation"]["output_dir"])

    # --- Step 1: Load and merge all data sources ---
    log.info("Step 1: Loading and merging data sources")

    with open(data_dir / "train.json") as f:
        train_samples = json.load(f)
    with open(data_dir / "val.json") as f:
        val_samples = json.load(f)

    log.info("Base data loaded", train=len(train_samples), val=len(val_samples))

    # Negative samples
    for neg_path, target in [
        (data_dir / "train_negative.json", train_samples),
        (data_dir / "val_negative.json", val_samples),
    ]:
        if neg_path.exists():
            with open(neg_path) as f:
                negs = json.load(f)
            target.extend(negs)
            log.info("Negative samples merged", file=neg_path.name, count=len(negs))

    # Oral-format samples
    oral_path = data_dir / "train_oral.json"
    if oral_path.exists():
        with open(oral_path) as f:
            oral_samples = json.load(f)
        train_samples.extend(oral_samples)
        log.info("Oral-format samples merged", count=len(oral_samples))

    # Gemini-generated diverse samples (different LLM family for diversity)
    gemini_path = data_dir / "train_gemini.json"
    if gemini_path.exists():
        with open(gemini_path) as f:
            gemini_samples = json.load(f)
        train_samples.extend(gemini_samples)
        log.info("Gemini samples merged", count=len(gemini_samples))

    log.info("After merge", train=len(train_samples), val=len(val_samples))

    # --- Step 2: Apply oral format transforms to 30% of original data ---
    log.info("Step 2: Applying oral format transforms")
    train_samples = apply_oral_format_transforms(train_samples, transform_ratio=0.3)

    # --- Step 3: Validate and filter ---
    log.info("Step 3: Validating transformed data")
    report = validate_dataset(train_samples)
    print_report(report)

    if report["invalid_samples"] > 0:
        log.warning("Filtering out invalid samples", count=report["invalid_samples"])
        train_samples = [s for s in train_samples if validate_sample(s)[0]]
        log.info("After filtering", train=len(train_samples))

    # --- Step 4: Save merged + transformed data ---
    log.info("Step 4: Saving prepared data")
    prepared_dir = data_dir / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)

    with open(prepared_dir / "train_prepared.json", "w") as f:
        json.dump(train_samples, f, indent=2)
    with open(prepared_dir / "val_prepared.json", "w") as f:
        json.dump(val_samples, f, indent=2)

    log.info(
        "Data preparation complete",
        train=len(train_samples),
        val=len(val_samples),
        output_dir=str(prepared_dir),
    )


if __name__ == "__main__":
    main()