"""Generate secondary LLM test set for evaluation.

Uses different prompts, different entity values, and different scenario
distribution than training data. Results reported separately from
hand-written gold set to highlight the circular evaluation limitation.

Tracks offset auto-fix rate for transparency.
"""

import json
import os
import random
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from src.data.generate import _validate_sample, _try_fix_offsets
from src.utils.logger import setup_logging, get_logger

log = get_logger(__name__)

# Entity pools deliberately NON-OVERLAPPING with training data pools
# Sized to support 100 samples with minimal repetition

NAMES = [
    "Emma Watson", "James Rivera", "Sophie Chen", "Marcus Johnson",
    "Aisha Patel", "Oliver Brown", "Yuna Kim", "Diego Martinez",
    "Hannah Mueller", "Raj Krishnan", "Elena Volkov", "Tyler Washington",
    "Mei Lin", "Oscar Diaz", "Freya Andersen", "Kwame Osei",
    "Isabelle Dupont", "Hiroshi Sato", "Amara Okafor", "Noah Fischer",
    "Chloe Bennett", "Andre Williams", "Zara Hussein", "Patrick O'Malley",
    "Leila Ahmadi", "Samuel Okonkwo", "Ingrid Larsson", "Rafael Costa",
    "Nadia Petrov", "Ethan Park", "Clara Hoffman", "Victor Reyes",
    "Ayumi Nakamura", "Luca Romano", "Fatou Diallo", "Brendan Murphy",
    "Simone Beaumont", "Tariq Hassan", "Julia Kowalski", "Devon Clarke",
]

EMAILS = [
    "emma.w@live.com", "jrivera@mail.com", "sophie.chen@outlook.com",
    "m.johnson@aol.com", "aisha.p@zoho.com", "oliver.b@tutanota.com",
    "yuna.kim@naver.com", "diego.m@mail.com", "h.mueller@web.de",
    "raj.k@rediffmail.com", "elena.v@yandex.com", "tyler.w@msn.com",
    "mei.lin@qq.com", "oscar.d@mail.com", "freya.a@proton.me",
    "k.osei@afrimail.com", "i.dupont@orange.fr", "h.sato@docomo.ne.jp",
    "amara.o@mail.com", "noah.f@gmx.de", "chloe.b@icloud.com",
    "andre.w@outlook.com", "zara.h@mail.com", "p.omalley@eircom.net",
    "leila.a@mail.com", "s.okonkwo@mail.com", "ingrid.l@telia.se",
    "rafael.c@uol.com.br", "nadia.p@mail.ru", "ethan.p@daum.net",
]

CONTRACT_IDS = [
    "SVC-10293", "RTN-2025-0041", "BK-88712", "ORD-2025-3387",
    "INS-44201", "TEL-90032", "MNT-2024-771", "CAS-55109",
    "WRK-33802", "PLN-2025-0089", "DLV-71123", "RPR-2024-445",
    "CLN-99281", "UPG-2025-112", "SHP-60334", "TRF-2024-887",
    "EXT-40019", "RNW-2025-0056", "PMT-85521", "ACT-2024-303",
    "DSP-70442", "QRY-2025-0128", "RSV-33019", "CHG-2024-667",
    "VLD-50883", "TST-2025-0199", "REQ-44570", "FLW-2024-221",
    "ADJ-90134", "CMP-2025-0077",
]

PRODUCTS = [
    "Google Pixel 9", "Bose QuietComfort headphones", "Lenovo ThinkPad X1",
    "Roomba j7 robot vacuum", "Apple Watch Ultra", "Canon EOS R6 camera",
    "Herman Miller Aeron chair", "Sonos Era 300 speaker",
    "Breville Barista Express", "Garmin Fenix 8 watch",
    "Samsung Frame TV 55 inch", "Dyson Airwrap styler",
    "Logitech MX Master mouse", "Theragun Elite massager",
    "Weber Spirit grill", "Peloton Bike Plus",
    "Vitamix A3500 blender", "Philips Hue starter kit",
    "Ember travel mug", "Oura Ring Gen 3",
    "Steam Deck OLED", "DJI Mini 4 Pro drone",
    "Anker PowerCore battery pack", "Kobo Libra Colour ereader",
    "Instant Pot Duo Plus", "JBL Flip 6 speaker",
    "TP-Link Deco mesh router", "Ninja Creami ice cream maker",
    "Eufy security camera", "Cricut Maker 3",
]

DATES_ABSOLUTE = [
    "September 22nd", "April 10th 2025", "11/15/2024", "July 4th",
    "2025-01-20", "December 28th", "March 8th 2025", "October 3rd",
    "January 31st", "August 19th 2024", "06/02/2025", "November 11th",
    "February 14th", "May 27th 2025", "2024-12-05", "June 30th",
]

DATES_RELATIVE = [
    "last Wednesday", "four days ago", "about two weeks ago",
    "yesterday morning", "last month", "a couple of days back",
    "earlier today", "sometime last weekend", "the other day",
    "last Thursday evening", "roughly a week ago", "three weeks back",
    "the beginning of this month", "a few days ago", "last Saturday",
]

SCENARIOS = [
    "Customer checking order delivery status",
    "Customer requesting refund for damaged product",
    "Customer reporting a technical malfunction",
    "Customer updating contact information on account",
    "Customer disputing an unexpected charge",
    "Customer asking about warranty coverage",
    "Customer scheduling a repair appointment",
    "Customer asking about product availability",
]


class NonRepeatingPool:
    """Sample from a pool with minimal repetition.

    Exhausts the pool before recycling, ensuring maximum diversity.
    """

    def __init__(self, values: list[str], rng: random.Random) -> None:
        self._all = list(values)
        self._remaining = []
        self._rng = rng

    def pick(self) -> str:
        if not self._remaining:
            self._remaining = list(self._all)
            self._rng.shuffle(self._remaining)
        return self._remaining.pop()


def generate_secondary_test(
    n_samples: int,
    output_path: Path,
) -> dict[str, Any]:
    """Generate secondary test samples with non-overlapping entity values."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    rng = random.Random(12345)

    # Non-repeating pools for maximum diversity
    pools = {
        "NAME": NonRepeatingPool(NAMES, rng),
        "EMAIL": NonRepeatingPool(EMAILS, rng),
        "CONTRACT_ID": NonRepeatingPool(CONTRACT_IDS, rng),
        "PRODUCT": NonRepeatingPool(PRODUCTS, rng),
        "ISSUE_DATE": NonRepeatingPool(DATES_ABSOLUTE + DATES_RELATIVE, rng),
    }

    stats = {
        "attempts": 0,
        "generated": 0,
        "auto_fixed": 0,
        "rejected": 0,
        "errors": 0,
    }

    prompt_template = """Generate a realistic customer service phone call transcript snippet.

Scenario: {scenario}
Style: {style}

The text MUST contain these exact entity values verbatim:
{entity_list}

CRITICAL RULES:
1. Output valid JSON only. No markdown, no explanation.
2. JSON has "text" (string) and "entities" (array of objects with "text", "label", "start", "end").
3. Labels: NAME, EMAIL, CONTRACT_ID, PRODUCT, ISSUE_DATE
4. text[start:end] must exactly equal the entity text.
5. Use natural spoken language with occasional fillers (uh, um).
6. Do NOT include entities beyond those listed above.

Example: {{"text": "Hi this is Sophie Chen calling about order SVC-10293", "entities": [{{"text": "Sophie Chen", "label": "NAME", "start": 11, "end": 22}}, {{"text": "SVC-10293", "label": "CONTRACT_ID", "start": 43, "end": 52}}]}}"""

    styles = ["polite", "frustrated", "neutral", "rushed", "confused", "casual"]

    samples = []

    with tqdm(total=n_samples, desc="Generating secondary test set") as pbar:
        while len(samples) < n_samples:
            n_entities = rng.choices([1, 2, 3], weights=[0.2, 0.5, 0.3])[0]
            entity_types = rng.sample(
                ["NAME", "EMAIL", "CONTRACT_ID", "PRODUCT", "ISSUE_DATE"],
                k=min(n_entities, 5),
            )

            entity_values = {}
            for etype in entity_types:
                entity_values[etype] = pools[etype].pick()

            entity_list = "\n".join(
                f"  - {etype}: \"{val}\"" for etype, val in entity_values.items()
            )

            prompt = prompt_template.format(
                scenario=rng.choice(SCENARIOS),
                style=rng.choice(styles),
                entity_list=entity_list,
            )

            stats["attempts"] += 1

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=400,
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
                            stats["auto_fixed"] += 1

                if is_valid:
                    samples.append(sample)
                    stats["generated"] += 1
                    pbar.update(1)
                else:
                    stats["rejected"] += 1
                    log.debug("Rejected secondary sample", reason=reason)

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                stats["errors"] += 1
                log.debug("Secondary generation error", error=str(e))
            except Exception as e:
                stats["errors"] += 1
                log.warning("Unexpected error", error=str(e))

    # --- Add negative samples (no entities) ---
    log.info("Generating negative samples for secondary set")
    neg_prompt = """Generate a realistic customer service phone call transcript snippet (1-2 sentences).
It should be general conversation with NO names, emails, order numbers, product names, or dates.
Examples: greetings, hold messages, transfer requests, general questions.
Output valid JSON only: {"text": "...", "entities": []}"""

    n_negatives = 10
    neg_generated = 0
    neg_attempts = 0
    max_neg_attempts = n_negatives * 5
    while neg_generated < n_negatives and neg_attempts < max_neg_attempts:
        neg_attempts += 1
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": neg_prompt}],
                temperature=0.9,
                max_tokens=150,
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            sample = json.loads(raw)
            if "text" in sample and isinstance(sample.get("entities"), list):
                sample["entities"] = []
                samples.append(sample)
                neg_generated += 1
        except Exception as e:
            log.debug("Negative sample error", error=str(e))

    if neg_generated < n_negatives:
        log.warning("Could not generate all negative samples", got=neg_generated, target=n_negatives)
    stats["negative_samples"] = neg_generated

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)

    # Report
    fix_rate = stats["auto_fixed"] / max(stats["generated"], 1) * 100
    accept_rate = stats["generated"] / max(stats["attempts"], 1) * 100

    print("\n" + "=" * 60)
    print("SECONDARY TEST SET GENERATION REPORT")
    print("=" * 60)
    print(f"  Entity samples generated: {stats['generated']}")
    print(f"  Negative samples: {stats['negative_samples']}")
    print(f"  Total: {len(samples)}")
    print(f"  Attempts: {stats['attempts']}")
    print(f"  Acceptance rate: {accept_rate:.1f}%")
    print(f"  Auto-fixed offsets: {stats['auto_fixed']} ({fix_rate:.1f}% of entity samples)")
    print(f"  Rejected: {stats['rejected']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Saved to: {output_path}")
    print("=" * 60)

    log.info(
        "Secondary test set generated",
        total=len(samples),
        fix_rate=f"{fix_rate:.1f}%",
    )

    stats["fix_rate_pct"] = round(fix_rate, 1)
    stats["accept_rate_pct"] = round(accept_rate, 1)
    stats["total_samples"] = len(samples)
    return stats


def main() -> None:
    setup_logging()
    stats = generate_secondary_test(
        n_samples=100,
        output_path=Path("data/eval/secondary_test.json"),
    )
    with open(Path("results/secondary_test_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()