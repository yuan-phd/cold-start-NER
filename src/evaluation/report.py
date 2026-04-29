"""Evaluation report generator — produce markdown report with tables and figures."""

import json
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

log = get_logger(__name__)


def generate_report(
    results: dict[str, Any],
    output_path: Path,
) -> None:
    """Generate a markdown evaluation report.

    Args:
        results: Dict containing all evaluation results with keys:
            - model_results: dict mapping model_name to evaluation report
            - ablation_results: optional ablation study results
            - speed_results: optional inference speed results
            - error_report: optional error analysis report
        output_path: Path to save the markdown file.
    """
    lines = []
    lines.append("# Evaluation Report\n")
    lines.append("Auto-generated evaluation results.\n")

    # --- Main results table ---
    if "model_results" in results:
        lines.append("## Model Comparison\n")
        lines.append("### Strict Match\n")

        # Header
        lines.append(
            "| Model | Precision | Recall | F1 | TP | FP | FN |"
        )
        lines.append("|-------|-----------|--------|-----|-----|-----|-----|")

        for model_name, report in results["model_results"].items():
            if "strict" not in report:
                continue
            o = report["strict"]["overall"]
            lines.append(
                f"| {model_name} | {o['precision']:.3f} | {o['recall']:.3f} | "
                f"{o['f1']:.3f} | {o['tp']} | {o['fp']} | {o['fn']} |"
            )

        lines.append("\n### Partial Match\n")
        lines.append(
            "| Model | Precision | Recall | F1 | TP | FP | FN |"
        )
        lines.append("|-------|-----------|--------|-----|-----|-----|-----|")

        for model_name, report in results["model_results"].items():
            if "partial" not in report:
                continue
            o = report["partial"]["overall"]
            lines.append(
                f"| {model_name} | {o['precision']:.3f} | {o['recall']:.3f} | "
                f"{o['f1']:.3f} | {o['tp']} | {o['fp']} | {o['fn']} |"
            )

    # --- Per entity type breakdown for ALL models ---
    if "per_entity_all_models" in results:
        lines.append("\n## Per-Entity-Type Breakdown (All Models, Strict Match)\n")

        entity_types = set()
        for model_data in results["per_entity_all_models"].values():
            entity_types.update(model_data.keys())
        entity_types = sorted(entity_types)

        for etype in entity_types:
            lines.append(f"\n### {etype}\n")
            lines.append("| Model | Precision | Recall | F1 | Support |")
            lines.append("|-------|-----------|--------|-----|---------|")

            for model_name, model_data in results["per_entity_all_models"].items():
                if etype in model_data:
                    m = model_data[etype]
                    lines.append(
                        f"| {model_name} | {m['precision']:.3f} | "
                        f"{m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |"
                    )

    # --- Noise robustness results ---
    if "noise_results" in results and results["noise_results"]:
        lines.append("\n## Noise Robustness (Gold Noisy Subset)\n")

        lines.append("### Strict Match\n")
        lines.append("| Model | Precision | Recall | F1 | TP | FP | FN |")
        lines.append("|-------|-----------|--------|-----|-----|-----|-----|")
        for model_name, report in results["noise_results"].items():
            o = report["strict"]["overall"]
            lines.append(
                f"| {model_name} | {o['precision']:.3f} | {o['recall']:.3f} | "
                f"{o['f1']:.3f} | {o['tp']} | {o['fp']} | {o['fn']} |"
            )

        lines.append("\n### Partial Match\n")
        lines.append("| Model | Precision | Recall | F1 | TP | FP | FN |")
        lines.append("|-------|-----------|--------|-----|-----|-----|-----|")
        for model_name, report in results["noise_results"].items():
            o = report["partial"]["overall"]
            lines.append(
                f"| {model_name} | {o['precision']:.3f} | {o['recall']:.3f} | "
                f"{o['f1']:.3f} | {o['tp']} | {o['fp']} | {o['fn']} |"
            )

        # Per-entity breakdown on noisy subset
        lines.append("\n### Per-Entity Breakdown (Noisy Subset, Strict Match)\n")

        # Collect all entity types
        entity_types = set()
        for model_name, report in results["noise_results"].items():
            if "per_entity_type" in report["strict"]:
                entity_types.update(report["strict"]["per_entity_type"].keys())
        entity_types = sorted(entity_types)

        if entity_types:
            header = "| Model | " + " | ".join(entity_types) + " |"
            separator = "|-------|" + "|".join(["------" for _ in entity_types]) + "|"
            lines.append(header)
            lines.append(separator)

            for model_name, report in results["noise_results"].items():
                pet = report["strict"].get("per_entity_type", {})
                cells = []
                for et in entity_types:
                    if et in pet:
                        cells.append(f"{pet[et]['f1']:.3f}")
                    else:
                        cells.append("—")
                lines.append(f"| {model_name} | " + " | ".join(cells) + " |")

    # --- Secondary test set results ---
    if "secondary_results" in results and results["secondary_results"]:
        lines.append("\n## Secondary Test Set (LLM-Generated, n=110)\n")
        lines.append("**Note:** This test set was generated by GPT-4o-mini with non-overlapping entity pools.")
        lines.append("100% of samples required auto-offset-correction. Results reported separately from gold set")
        lines.append("due to circular evaluation risk (same LLM family generated training and test data).\n")

        lines.append("### Strict Match\n")
        lines.append("| Model | Precision | Recall | F1 | TP | FP | FN |")
        lines.append("|-------|-----------|--------|-----|-----|-----|-----|")
        for model_name, report in results["secondary_results"].items():
            o = report["strict"]["overall"]
            lines.append(
                f"| {model_name} | {o['precision']:.3f} | {o['recall']:.3f} | "
                f"{o['f1']:.3f} | {o['tp']} | {o['fp']} | {o['fn']} |"
            )

        lines.append("\n### Partial Match\n")
        lines.append("| Model | Precision | Recall | F1 | TP | FP | FN |")
        lines.append("|-------|-----------|--------|-----|-----|-----|-----|")
        for model_name, report in results["secondary_results"].items():
            o = report["partial"]["overall"]
            lines.append(
                f"| {model_name} | {o['precision']:.3f} | {o['recall']:.3f} | "
                f"{o['f1']:.3f} | {o['tp']} | {o['fp']} | {o['fn']} |"
            )

    # --- Ablation results ---
    if "ablation_results" in results:
        lines.append("\n## Ablation Studies\n")

        for ablation_name, ablation_data in results["ablation_results"].items():
            lines.append(f"### {ablation_name}\n")
            lines.append("| Variant | F1 (Strict) |")
            lines.append("|---------|-------------|")

            for variant, report in ablation_data.items():
                f1 = report.get("strict", {}).get("overall", {}).get("f1", "--")
                if isinstance(f1, float):
                    f1 = f"{f1:.3f}"
                lines.append(f"| {variant} | {f1} |")
            lines.append("")

    # --- Inference speed ---
    if "speed_results" in results:
        lines.append("\n## Inference Speed (CPU)\n")
        lines.append("| Model | Avg ms/sample |")
        lines.append("|-------|---------------|")

        for model_name, ms in results["speed_results"].items():
            lines.append(f"| {model_name} | {ms:.1f} |")

    # --- Error analysis summary ---
    if "error_report" in results:
        er = results["error_report"]
        lines.append("\n## Error Analysis Summary\n")
        lines.append(f"Total errors: {er['total_errors']}\n")

        if er.get("error_type_counts"):
            lines.append("| Error Type | Count |")
            lines.append("|------------|-------|")
            for etype, count in er["error_type_counts"].items():
                lines.append(f"| {etype} | {count} |")

    # --- Figures ---
    lines.append("\n## Figures\n")
    figures_dir = Path("results/figures")
    if figures_dir.exists():
        for fig_path in sorted(figures_dir.glob("*.png")):
            lines.append(f"### {fig_path.stem.replace('_', ' ').title()}\n")
            lines.append(f"![{fig_path.stem}](figures/{fig_path.name})\n")

    # Write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(content)

    # Also save raw results as JSON
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info("Evaluation report generated", path=str(output_path))