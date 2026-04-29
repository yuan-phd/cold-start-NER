"""Error-driven iterative data refinement — analyze failures, regenerate targeted data.

Implements the core loop: train → evaluate → analyze errors → generate targeted
data → retrain. This mirrors Diabolocom Research's Curate-Train-Refine methodology
applied to NER.
"""

import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from src.data.generate import _validate_sample, _try_fix_offsets
from src.utils.config import load_config
from src.utils.logger import get_logger

log = get_logger(__name__)


def analyze_weaknesses(error_report: dict[str, Any]) -> dict[str, Any]:
    """Analyze error report to identify systematic weaknesses.

    Args:
        error_report: Output from evaluation.analyze.analyze_errors().

    Returns:
        Weakness analysis with targeted generation recommendations.
    """
    weaknesses = {
        "weak_entity_types": [],
        "dominant_error_types": [],
        "noise_sensitive_types": [],
        "recommendations": [],
    }

    # Find entity types with most errors
    entity_errors = error_report.get("error_by_entity_type", {})
    for etype, error_counts in entity_errors.items():
        total = sum(error_counts.values())
        if total > 0:
            weaknesses["weak_entity_types"].append({
                "entity_type": etype,
                "total_errors": total,
                "breakdown": error_counts,
            })

    # Sort by error count
    weaknesses["weak_entity_types"].sort(key=lambda x: x["total_errors"], reverse=True)

    # Find dominant error type
    error_counts = error_report.get("error_type_counts", {})
    if error_counts:
        dominant = max(error_counts.items(), key=lambda x: x[1])
        weaknesses["dominant_error_types"].append({
            "type": dominant[0],
            "count": dominant[1],
        })

    # Generate recommendations
    for weak in weaknesses["weak_entity_types"][:3]:
        etype = weak["entity_type"]
        breakdown = weak["breakdown"]

        if breakdown.get("COMPLETE_MISS", 0) > breakdown.get("BOUNDARY", 0):
            weaknesses["recommendations"].append(
                f"Generate more samples with {etype} entities in diverse contexts — "
                f"model is missing them entirely ({breakdown.get('COMPLETE_MISS', 0)} misses)"
            )
        elif breakdown.get("BOUNDARY", 0) > 0:
            weaknesses["recommendations"].append(
                f"Generate samples with {etype} entities at sentence boundaries and "
                f"adjacent to other entities — model has boundary issues ({breakdown.get('BOUNDARY', 0)} errors)"
            )
        if breakdown.get("TYPE_CONFUSION", 0) > 0:
            weaknesses["recommendations"].append(
                f"Generate samples that disambiguate {etype} from similar types — "
                f"model confuses types ({breakdown.get('TYPE_CONFUSION', 0)} confusions)"
            )

    log.info(
        "Weakness analysis complete",
        weak_types=[w["entity_type"] for w in weaknesses["weak_entity_types"][:3]],
        recommendations=len(weaknesses["recommendations"]),
    )

    return weaknesses


def generate_targeted_samples(
    weaknesses: dict[str, Any],
    n_samples: int,
    output_path: Path,
) -> dict[str, Any]:
    """Generate targeted training data to address identified weaknesses.

    Args:
        weaknesses: Output from analyze_weaknesses().
        n_samples: Number of targeted samples to generate.
        output_path: Path to save generated data.

    Returns:
        Generation statistics.
    """
    config = load_config()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    gen_config = config["data_generation"]

    # Build targeted prompts based on weaknesses
    weak_types = [w["entity_type"] for w in weaknesses["weak_entity_types"][:3]]
    if not weak_types:
        weak_types = ["PRODUCT", "ISSUE_DATE", "NAME"]  # Default focus

    recommendations = "\n".join(f"- {r}" for r in weaknesses.get("recommendations", []))

    prompt_template = """Generate a realistic customer service phone transcript snippet
that is specifically designed to test NER extraction for these challenging cases:

Focus entity types: {focus_types}
Specific challenges to include:
{recommendations}

Additional requirements:
- Include at least one entity from the focus types
- Make the context challenging: entity at start/end of sentence, adjacent to other entities,
  in noisy/colloquial speech
- Use natural spoken language with fillers (uh, um, like)
- Vary the entity placement and surrounding context

Output valid JSON only: {{"text": "...", "entities": [{{"text": "...", "label": "...", "start": int, "end": int}}]}}
Entity labels must be one of: NAME, EMAIL, CONTRACT_ID, PRODUCT, ISSUE_DATE
Offsets must satisfy: text[start:end] == entity text"""

    samples = []
    stats = {"attempts": 0, "generated": 0, "fixed": 0, "rejected": 0}

    with tqdm(total=n_samples, desc="Generating targeted samples") as pbar:
        while len(samples) < n_samples:
            focus = ", ".join(weak_types)
            prompt = prompt_template.format(
                focus_types=focus,
                recommendations=recommendations or "- Generate diverse challenging examples",
            )

            stats["attempts"] += 1
            try:
                response = client.chat.completions.create(
                    model=gen_config["api_model"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.9,
                    max_tokens=500,
                )
                raw = response.choices[0].message.content.strip()

                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                    if raw.endswith("```"):
                        raw = raw[:-3]
                    raw = raw.strip()

                sample = json.loads(raw)
                is_valid, reason = _validate_sample(sample)

                if not is_valid:
                    fixed = _try_fix_offsets(sample)
                    if fixed:
                        is_valid, reason = _validate_sample(fixed)
                        if is_valid:
                            sample = fixed
                            stats["fixed"] += 1

                if is_valid:
                    sample["metadata"] = {
                        "scenario": "targeted_refinement",
                        "focus_types": weak_types,
                    }
                    samples.append(sample)
                    stats["generated"] += 1
                    pbar.update(1)
                else:
                    stats["rejected"] += 1

            except Exception as e:
                log.debug("Targeted generation error", error=str(e))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)

    log.info(
        "Targeted generation complete",
        samples=len(samples),
        stats=stats,
        output=str(output_path),
    )
    return stats