"""Synthetic NER data generation via LLM for customer service dialogues.

Generates realistic customer service transcript snippets with entity annotations.
Each sample includes text and a list of entities with span offsets.
Uses OpenAI API for generation with structured JSON output.
"""

import json
import os
import random
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from src.utils.config import load_config, get_seed
from src.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Entity value pools — used to seed the LLM with realistic example values
# ---------------------------------------------------------------------------

NAMES = [
    "John Smith", "Sarah Johnson", "Michael Brown", "Emily Davis", "Robert Wilson",
    "Jessica Martinez", "David Anderson", "Jennifer Taylor", "James Thomas", "Maria Garcia",
    "William Jackson", "Linda White", "Richard Harris", "Susan Clark", "Joseph Lewis",
    "Karen Robinson", "Charles Walker", "Nancy Hall", "Daniel Allen", "Betty Young",
    "Hank Mueller", "Bob Patterson", "Alice Chen", "Mohammed Al-Rashid", "Priya Patel",
    "Yuki Tanaka", "Carlos Rivera", "Fatima Osei", "Liam O'Brien", "Ananya Sharma",
]

EMAILS = [
    "john.smith@gmail.com", "sarah.j@yahoo.com", "mbrown42@outlook.com",
    "emily.davis@hotmail.com", "rwilson@protonmail.com", "jess.m@gmail.com",
    "david.a@company.com", "jen.taylor@icloud.com", "jthomas99@gmail.com",
    "maria.garcia@outlook.com", "w.jackson@email.com", "linda.w@yahoo.com",
    "r.harris@gmail.com", "sclark@fastmail.com", "joe.lewis@outlook.com",
    "krobinson@gmail.com", "c.walker@protonmail.com", "nhall@company.org",
    "d.allen@email.com", "b.young@hotmail.com",
]

CONTRACT_IDS = [
    "CT-78432", "ORD-2024-5591", "ACC-00192", "REF-44821", "INV-2023-887",
    "POL-55123", "TKT-90274", "SUB-33018", "CLM-77659", "SRV-11042",
    "A1B2C3", "X9Y8Z7", "2024-ABX-001", "NB-449921", "HD-2025-03312",
    "CUS-88710", "RET-22045", "WO-67893", "PRJ-10284", "LIC-55901",
]

PRODUCTS = [
    "iPhone 15 Pro", "Samsung Galaxy S24", "MacBook Air M3", "Dell XPS 15",
    "Sony WH-1000XM5 headphones", "AirPods Pro", "iPad Air", "Kindle Paperwhite",
    "Dyson V15 vacuum", "Nespresso Vertuo", "Ring doorbell", "Fitbit Charge 6",
    "Nintendo Switch OLED", "PlayStation 5", "LG OLED TV 65 inch",
    "Premium subscription plan", "Fiber optic 500 Mbps plan", "Cloud storage 2TB plan",
    "Annual maintenance contract", "Extended warranty package",
]

ISSUE_DATES_ABSOLUTE = [
    "January 15th", "March 3rd 2024", "02/28/2025", "December 1st",
    "July 22 2024", "November 30th", "2024-09-15", "August 8th",
    "April 5th 2025", "June 12", "October 21st 2024", "May 17",
]

ISSUE_DATES_RELATIVE = [
    "last Tuesday", "three days ago", "about a week ago", "yesterday",
    "two weeks ago", "last month", "a few days back", "the day before yesterday",
    "last Friday", "earlier this week", "sometime last week", "about ten days ago",
    "last Monday morning", "a couple of weeks back", "end of last month",
]

SCENARIO_TEMPLATES = {
    "order_status_inquiry": {
        "description": "Customer calling to check the status of an order",
        "required_entities": ["NAME", "CONTRACT_ID"],
        "optional_entities": ["EMAIL", "PRODUCT"],
        "styles": ["polite", "impatient", "neutral"],
    },
    "product_complaint": {
        "description": "Customer complaining about a product issue",
        "required_entities": ["NAME", "PRODUCT"],
        "optional_entities": ["CONTRACT_ID", "ISSUE_DATE"],
        "styles": ["frustrated", "angry", "calm"],
    },
    "return_request": {
        "description": "Customer requesting a product return or exchange",
        "required_entities": ["NAME", "PRODUCT", "ISSUE_DATE"],
        "optional_entities": ["CONTRACT_ID", "EMAIL"],
        "styles": ["polite", "disappointed", "neutral"],
    },
    "account_update": {
        "description": "Customer updating their account information",
        "required_entities": ["NAME", "EMAIL"],
        "optional_entities": ["CONTRACT_ID"],
        "styles": ["neutral", "polite", "rushed"],
    },
    "technical_support": {
        "description": "Customer calling for technical help with a product",
        "required_entities": ["NAME", "PRODUCT"],
        "optional_entities": ["CONTRACT_ID", "ISSUE_DATE"],
        "styles": ["confused", "frustrated", "patient"],
    },
    "appointment_scheduling": {
        "description": "Customer scheduling or rescheduling a service appointment",
        "required_entities": ["NAME", "ISSUE_DATE"],
        "optional_entities": ["EMAIL", "CONTRACT_ID"],
        "styles": ["polite", "rushed", "neutral"],
    },
    "billing_dispute": {
        "description": "Customer disputing a charge or billing issue",
        "required_entities": ["NAME", "CONTRACT_ID", "ISSUE_DATE"],
        "optional_entities": ["EMAIL", "PRODUCT"],
        "styles": ["angry", "confused", "assertive"],
    },
    "general_inquiry": {
        "description": "Customer making a general inquiry",
        "required_entities": ["NAME"],
        "optional_entities": ["EMAIL", "PRODUCT"],
        "styles": ["polite", "casual", "neutral"],
    },
}

INDUSTRIES = [
    "telecommunications", "retail", "e-commerce", "banking",
    "insurance", "healthcare", "subscription services", "utilities",
    "electronics", "home appliances",
]


def _pick_entity_values(
    required: list[str], optional: list[str], rng: random.Random
) -> dict[str, str]:
    """Select concrete entity values for a sample.

    Args:
        required: Entity types that must appear.
        optional: Entity types that may appear (50% chance each).
        rng: Random number generator for reproducibility.

    Returns:
        Dict mapping entity type to a concrete value.
    """
    pools = {
        "NAME": NAMES,
        "EMAIL": EMAILS,
        "CONTRACT_ID": CONTRACT_IDS,
        "PRODUCT": PRODUCTS,
        "ISSUE_DATE": ISSUE_DATES_ABSOLUTE + ISSUE_DATES_RELATIVE,
    }

    entities = {}
    for etype in required:
        entities[etype] = rng.choice(pools[etype])
    for etype in optional:
        if rng.random() < 0.5:
            entities[etype] = rng.choice(pools[etype])
    return entities


def _build_prompt(
    scenario_key: str,
    entity_values: dict[str, str],
    style: str,
    industry: str,
) -> str:
    """Build the LLM prompt for generating one NER-annotated sample.

    Args:
        scenario_key: Key into SCENARIO_TEMPLATES.
        entity_values: Mapping of entity type to value to include.
        style: Conversation style (e.g. "polite", "angry").
        industry: Industry context (e.g. "telecommunications").

    Returns:
        Formatted prompt string.
    """
    scenario = SCENARIO_TEMPLATES[scenario_key]
    entity_instructions = "\n".join(
        f"  - {etype}: \"{value}\"" for etype, value in entity_values.items()
    )

    return f"""Generate a realistic customer service phone transcript snippet.

Context:
- Scenario: {scenario["description"]}
- Industry: {industry}
- Conversation style: {style}
- This is a transcription of spoken language — use natural, conversational tone.
  Include speech patterns like "um", "uh", "so", "well", partial sentences, and self-corrections.

The text MUST contain these exact entity values (use them verbatim):
{entity_instructions}

Requirements:
1. Output valid JSON only. No markdown, no backticks, no explanation.
2. The JSON must have exactly two keys:
   - "text": a string containing the transcript snippet (1-4 sentences)
   - "entities": an array of objects, each with:
     - "text": the exact entity string as it appears in the text
     - "label": one of NAME, EMAIL, CONTRACT_ID, PRODUCT, ISSUE_DATE
     - "start": character offset where the entity begins in the text
     - "end": character offset where the entity ends in the text
3. Every entity value listed above MUST appear in the text and be annotated.
4. start/end offsets must be exact: text[start:end] == entity text.
5. Do not add entities beyond those listed above.

Example output format:
{{"text": "Hi my name is John Smith and I am calling about order CT-78432", "entities": [{{"text": "John Smith", "label": "NAME", "start": 14, "end": 24}}, {{"text": "CT-78432", "label": "CONTRACT_ID", "start": 54, "end": 62}}]}}"""


def _validate_sample(sample: dict[str, Any]) -> tuple[bool, str]:
    """Validate a single generated sample for correctness.

    Args:
        sample: Parsed JSON with "text" and "entities" keys.

    Returns:
        (is_valid, reason) tuple.
    """
    if "text" not in sample or "entities" not in sample:
        return False, "Missing 'text' or 'entities' key"

    text = sample["text"]
    if not isinstance(text, str) or len(text) < 10:
        return False, "Text too short or not a string"

    valid_labels = {"NAME", "EMAIL", "CONTRACT_ID", "PRODUCT", "ISSUE_DATE"}

    for i, ent in enumerate(sample["entities"]):
        # Check required fields
        for field in ("text", "label", "start", "end"):
            if field not in ent:
                return False, f"Entity {i} missing field '{field}'"

        # Check label validity
        if ent["label"] not in valid_labels:
            return False, f"Entity {i} has invalid label '{ent['label']}'"

        # Check offset types
        if not isinstance(ent["start"], int) or not isinstance(ent["end"], int):
            return False, f"Entity {i} has non-integer offsets"

        # Check offset bounds
        if ent["start"] < 0 or ent["end"] > len(text) or ent["start"] >= ent["end"]:
            return False, f"Entity {i} has invalid offset range [{ent['start']}:{ent['end']}]"

        # Check span alignment — the critical validation
        extracted = text[ent["start"]:ent["end"]]
        if extracted != ent["text"]:
            return False, (
                f"Entity {i} span mismatch: "
                f"text[{ent['start']}:{ent['end']}] = '{extracted}' != '{ent['text']}'"
            )

    return True, "OK"


def _try_fix_offsets(sample: dict[str, Any]) -> dict[str, Any] | None:
    """Attempt to fix misaligned entity offsets by searching for the entity text.

    LLMs frequently get offsets wrong but include the correct entity text.
    This function searches for the entity text in the sample text and fixes offsets.

    Args:
        sample: Sample with potentially misaligned offsets.

    Returns:
        Fixed sample, or None if unfixable.
    """
    text = sample["text"]
    fixed_entities = []

    for ent in sample["entities"]:
        entity_text = ent["text"].strip()  # Strip whitespace from entity text
        # Search for exact match in text
        idx = text.find(entity_text)
        if idx == -1:
            # Try case-insensitive search
            idx = text.lower().find(entity_text.lower())
            if idx != -1:
                # Use the actual text from the source
                entity_text = text[idx:idx + len(entity_text)]

        if idx == -1:
            return None  # Entity text not found at all

        fixed_entities.append({
            "text": entity_text,
            "label": ent["label"],
            "start": idx,
            "end": idx + len(entity_text),
        })

    return {"text": text, "entities": fixed_entities}

def generate_samples(
    n_samples: int,
    output_path: Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate n synthetic NER samples and save to JSON.

    Args:
        n_samples: Number of samples to generate.
        output_path: Path to save the output JSON file.
        config: Optional config override.

    Returns:
        Statistics dict with counts of generated, fixed, and rejected samples.
    """
    if config is None:
        config = load_config()

    rng = random.Random(get_seed())
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    gen_config = config["data_generation"]

    # Build weighted scenario list
    scenarios = config["data_generation"]["scenarios"]
    scenario_keys = [s["type"] for s in scenarios]
    scenario_weights = [s["weight"] for s in scenarios]

    samples: list[dict[str, Any]] = []
    stats = {"total_attempts": 0, "generated": 0, "fixed": 0, "rejected": 0, "errors": 0}

    with tqdm(total=n_samples, desc="Generating samples") as pbar:
        while len(samples) < n_samples:
            # Pick scenario, style, industry, entity values
            scenario_key = rng.choices(scenario_keys, weights=scenario_weights, k=1)[0]
            scenario = SCENARIO_TEMPLATES[scenario_key]
            style = rng.choice(scenario["styles"])
            industry = rng.choice(INDUSTRIES)
            entity_values = _pick_entity_values(
                scenario["required_entities"],
                scenario["optional_entities"],
                rng,
            )

            prompt = _build_prompt(scenario_key, entity_values, style, industry)
            stats["total_attempts"] += 1

            try:
                response = client.chat.completions.create(
                    model=gen_config["api_model"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=gen_config["temperature"],
                    max_tokens=500,
                )
                raw = response.choices[0].message.content.strip()

                # Clean potential markdown wrapping
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                    if raw.endswith("```"):
                        raw = raw[:-3]
                    raw = raw.strip()

                sample = json.loads(raw)

                # Validate
                is_valid, reason = _validate_sample(sample)

                if not is_valid:
                    # Try to fix offsets
                    fixed = _try_fix_offsets(sample)
                    if fixed is not None:
                        is_valid, reason = _validate_sample(fixed)
                        if is_valid:
                            sample = fixed
                            stats["fixed"] += 1

                if is_valid:
                    sample["metadata"] = {
                        "scenario": scenario_key,
                        "style": style,
                        "industry": industry,
                    }
                    samples.append(sample)
                    stats["generated"] += 1
                    pbar.update(1)
                else:
                    stats["rejected"] += 1
                    log.debug("Rejected sample", reason=reason)

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                stats["errors"] += 1
                log.debug("Generation error", error=str(e))
            except Exception as e:
                stats["errors"] += 1
                log.warning("Unexpected error", error=str(e))
                time.sleep(1)  # Back off on unexpected errors

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)

    log.info(
        "Generation complete",
        samples=len(samples),
        attempts=stats["total_attempts"],
        fixed=stats["fixed"],
        rejected=stats["rejected"],
        errors=stats["errors"],
        acceptance_rate=f"{len(samples) / stats['total_attempts']:.1%}",
        output=str(output_path),
    )

    return stats


def generate_negative_samples(
    n_samples: int,
    output_path: Path,
) -> int:
    """Generate samples with NO entities — important for reducing false positives.

    Args:
        n_samples: Number of negative samples.
        output_path: Path to save output JSON.

    Returns:
        Number of samples generated.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    config = load_config()
    gen_config = config["data_generation"]

    prompt = """Generate a realistic customer service phone transcript snippet (1-3 sentences).
The transcript should be general conversation that does NOT contain any of the following:
- Person names
- Email addresses
- Contract/order IDs
- Product names
- Dates

Examples of valid output: greetings, hold messages, transfer messages, general questions.

Output valid JSON only: {"text": "...", "entities": []}"""

    samples = []
    for _ in tqdm(range(n_samples), desc="Generating negative samples"):
        try:
            response = client.chat.completions.create(
                model=gen_config["api_model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=200,
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            sample = json.loads(raw)
            if "text" in sample and isinstance(sample.get("entities"), list):
                sample["entities"] = []  # Force empty
                sample["metadata"] = {"scenario": "negative", "style": "neutral", "industry": "any"}
                samples.append(sample)
        except Exception as e:
            log.debug("Negative sample error", error=str(e))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)

    log.info("Negative samples generated", count=len(samples), output=str(output_path))
    return len(samples)

def generate_oral_samples(
    n_samples: int,
    output_path: Path,
) -> dict[str, Any]:
    """Generate samples with oral/ASR-style entity formatting.

    These samples feature entities as they would be spoken aloud:
    - Emails: "john at gmail dot com" instead of "john@gmail.com"
    - Contract IDs: "CT dash 55123" instead of "CT-55123"
    - Products: informal names ("iphone fifteen" vs "iPhone 15")
    - Dates: spoken format ("february twenty eighth" vs "02/28")
    - Names: with spelling ("priya thats p r i y a")

    Args:
        n_samples: Number of samples to generate.
        output_path: Path to save output JSON.

    Returns:
        Generation statistics.
    """
    config = load_config()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    gen_config = config["data_generation"]
    rng = random.Random(get_seed() + 999)

    prompt_template = """Generate a realistic customer service PHONE CALL transcript snippet.

CRITICAL: This is SPOKEN language transcribed by speech recognition. Entities must appear
in their ORAL/SPOKEN form, not written form. Examples:
- Email: "john at gmail dot com" NOT "john@gmail.com"
- Contract ID: "CT dash 55123" or "CT dash five five one two three" NOT "CT-55123"
- Product: "iphone fifteen pro" or "the new samsung galaxy" NOT "iPhone 15 Pro"
- Date: "february twenty eighth" or "oh two twenty eight" NOT "02/28/2025"
- Name with spelling: "priya thats p r i y a patel" or just "priya patel"

Scenario: {scenario}
Style: {style}
Industry: {industry}

Requirements:
- The entities MUST be in spoken/oral form as described above
- Entity text in the annotation must match EXACTLY what appears in the transcript
- {entity_instruction}
- {position_instruction}
- Include natural speech disfluencies (uh, um, like, you know)

Output valid JSON only: {{"text": "...", "entities": [{{"text": "...", "label": "...", "start": int, "end": int}}]}}
Labels: NAME, EMAIL, CONTRACT_ID, PRODUCT, ISSUE_DATE
Offsets must satisfy: text[start:end] == entity text"""

    scenarios = [
        ("Customer calling to check order status", "order_status_inquiry"),
        ("Customer complaining about a product", "product_complaint"),
        ("Customer requesting a return", "return_request"),
        ("Customer updating account info", "account_update"),
        ("Customer calling for technical support", "technical_support"),
        ("Customer disputing a bill", "billing_dispute"),
    ]

    entity_instructions = [
        "Include an EMAIL in spoken form (use 'at' and 'dot com')",
        "Include a CONTRACT_ID in spoken form (spell out or say 'dash')",
        "Include a PRODUCT in informal spoken form (lowercase, no model numbers written formally)",
        "Include an ISSUE_DATE in spoken form (say the date out loud)",
        "Include a NAME and have the caller spell it out",
        "Include both an EMAIL (spoken form) and a CONTRACT_ID (spoken form)",
    ]

    position_instructions = [
        "Start the text with the caller's name (entity at the very beginning)",
        "End the text with an entity (email or contract ID at the very end)",
        "Place two entities adjacent to each other with minimal gap",
        "Use a natural entity placement",
        "Start the text directly with an entity",
    ]

    industries = [
        "telecommunications", "retail", "e-commerce", "banking",
        "insurance", "healthcare", "subscription services",
    ]

    styles = ["frustrated", "polite", "rushed", "confused", "neutral", "angry", "casual"]

    samples = []
    stats = {"attempts": 0, "generated": 0, "fixed": 0, "rejected": 0, "errors": 0}

    with tqdm(total=n_samples, desc="Generating oral-style samples") as pbar:
        while len(samples) < n_samples:
            scenario_desc, scenario_key = rng.choice(scenarios)
            entity_instr = rng.choice(entity_instructions)
            position_instr = rng.choice(position_instructions)
            industry = rng.choice(industries)
            style = rng.choice(styles)

            prompt = prompt_template.format(
                scenario=scenario_desc,
                style=style,
                industry=industry,
                entity_instruction=entity_instr,
                position_instruction=position_instr,
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
                        "scenario": scenario_key,
                        "style": style,
                        "industry": industry,
                        "data_type": "oral_format",
                    }
                    samples.append(sample)
                    stats["generated"] += 1
                    pbar.update(1)
                else:
                    stats["rejected"] += 1
                    log.debug("Rejected oral sample", reason=reason)

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                stats["errors"] += 1
                log.debug("Oral generation error", error=str(e))
            except Exception as e:
                stats["errors"] += 1
                log.warning("Unexpected error", error=str(e))
                time.sleep(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)

    log.info(
        "Oral-style generation complete",
        samples=len(samples),
        stats=stats,
        output=str(output_path),
    )
    return stats