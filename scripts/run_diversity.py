"""Training data diversity analysis — n-gram repetition and structural diversity.

Measures how diverse the GPT-4o-mini generated training data is by computing
n-gram repetition rates, sentence structure patterns, and entity context variety.
High repetition indicates the model may learn rigid patterns rather than
generalizable features.

Output: results/diversity_analysis.json

Usage:
    python scripts/run_diversity.py
"""

import json
from collections import Counter
from pathlib import Path

from src.utils.logger import setup_logging, get_logger


def tokenize_simple(text: str) -> list[str]:
    """Lowercase whitespace tokenization."""
    return text.lower().split()


def compute_ngram_stats(texts: list[str], n: int) -> dict:
    """Compute n-gram repetition statistics across a corpus."""
    ngram_counts = Counter()
    total_ngrams = 0

    for text in texts:
        tokens = tokenize_simple(text)
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngram_counts[ngram] += 1
            total_ngrams += 1

    unique_ngrams = len(ngram_counts)
    repeated_ngrams = sum(1 for c in ngram_counts.values() if c > 1)
    repeated_instances = sum(c for c in ngram_counts.values() if c > 1)

    return {
        "n": n,
        "total_ngrams": total_ngrams,
        "unique_ngrams": unique_ngrams,
        "unique_ratio": round(unique_ngrams / max(total_ngrams, 1), 4),
        "repeated_ngrams": repeated_ngrams,
        "repeated_ratio": round(repeated_ngrams / max(unique_ngrams, 1), 4),
        "repeated_instance_ratio": round(repeated_instances / max(total_ngrams, 1), 4),
        "top_20": [
            {"ngram": " ".join(ng), "count": c}
            for ng, c in ngram_counts.most_common(20)
        ],
    }


def compute_opening_diversity(texts: list[str]) -> dict:
    """Analyze diversity of sentence openings (first 3 tokens)."""
    openings = Counter()
    for text in texts:
        tokens = tokenize_simple(text)
        if len(tokens) >= 3:
            opening = " ".join(tokens[:3])
            openings[opening] += 1

    total = sum(openings.values())
    unique = len(openings)

    return {
        "total_samples": total,
        "unique_openings": unique,
        "unique_ratio": round(unique / max(total, 1), 4),
        "top_20": [
            {"opening": o, "count": c}
            for o, c in openings.most_common(20)
        ],
    }


def compute_entity_context_diversity(samples: list[dict]) -> dict:
    """Analyze diversity of words surrounding entities."""
    context_patterns = Counter()
    entity_trigger_phrases = Counter()

    for sample in samples:
        text = sample["text"].lower()
        tokens = text.split()

        for entity in sample.get("entities", []):
            start_char = entity["start"]
            # Find the token index closest to entity start
            char_pos = 0
            token_idx = 0
            for i, tok in enumerate(tokens):
                if char_pos >= start_char:
                    token_idx = i
                    break
                char_pos += len(tok) + 1

            # Get 2 tokens before entity as context
            if token_idx >= 2:
                context = " ".join(tokens[token_idx - 2:token_idx])
                context_patterns[f"{entity['label']}: ...{context} [ENTITY]"] += 1

            # Get 1 token before as trigger
            if token_idx >= 1:
                entity_trigger_phrases[f"{entity['label']}: {tokens[token_idx - 1]}"] += 1

    return {
        "unique_context_patterns": len(context_patterns),
        "top_20_contexts": [
            {"pattern": p, "count": c}
            for p, c in context_patterns.most_common(20)
        ],
        "unique_triggers": len(entity_trigger_phrases),
        "top_20_triggers": [
            {"trigger": t, "count": c}
            for t, c in entity_trigger_phrases.most_common(20)
        ],
    }


def main() -> None:
    setup_logging()
    log = get_logger(__name__)

    # Load base prepared data (before noise, to measure generation diversity)
    prepared_path = Path("data/synthetic/prepared/train_prepared.json")
    if not prepared_path.exists():
        log.error("Prepared training data not found. Run 'make prepare' first.")
        return

    with open(prepared_path) as f:
        samples = json.load(f)

    texts = [s["text"] for s in samples]
    log.info("Loaded prepared data", samples=len(samples))

    # N-gram analysis
    results = {"total_samples": len(samples)}

    for n in [2, 3, 4]:
        log.info(f"Computing {n}-gram statistics")
        results[f"{n}gram"] = compute_ngram_stats(texts, n)

    # Opening diversity
    log.info("Computing opening diversity")
    results["openings"] = compute_opening_diversity(texts)

    # Entity context diversity
    log.info("Computing entity context diversity")
    results["entity_context"] = compute_entity_context_diversity(samples)

    # Save
    output_path = Path("results/diversity_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved", path=str(output_path))

    # Print summary
    print("\n" + "=" * 65)
    print("TRAINING DATA DIVERSITY ANALYSIS")
    print("=" * 65)
    print(f"Samples analyzed: {len(samples)} (prepared, pre-noise)")
    print()

    for n in [2, 3, 4]:
        stats = results[f"{n}gram"]
        print(f"--- {n}-grams ---")
        print(f"  Total: {stats['total_ngrams']}, Unique: {stats['unique_ngrams']} ({stats['unique_ratio']*100:.1f}%)")
        print(f"  Repeated n-grams: {stats['repeated_ngrams']} ({stats['repeated_ratio']*100:.1f}% of unique)")
        print(f"  Repeated instances: {stats['repeated_instance_ratio']*100:.1f}% of all n-grams")
        print("  Top 5:")
        for item in stats["top_20"][:5]:
            print(f"    '{item['ngram']}' — {item['count']}x")
        print()

    print("--- Sentence Openings ---")
    op = results["openings"]
    print(f"  Unique openings: {op['unique_openings']} / {op['total_samples']} ({op['unique_ratio']*100:.1f}%)")
    print("  Top 5:")
    for item in op["top_20"][:5]:
        print(f"    '{item['opening']}' — {item['count']}x")
    print()

    print("--- Entity Context Triggers ---")
    ec = results["entity_context"]
    print(f"  Unique context patterns: {ec['unique_context_patterns']}")
    print("  Top 10 trigger words:")
    for item in ec["top_20_triggers"][:10]:
        print(f"    {item['trigger']} — {item['count']}x")
    print("=" * 65)


if __name__ == "__main__":
    main()
