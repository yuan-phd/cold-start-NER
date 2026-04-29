"""Exploratory Data Analysis — validate synthetic data quality before training.

Covers five dimensions:
A. Text layer (lengths, tokenization, truncation, vocabulary coverage)
B. Entity layer (distribution, density, position, format diversity, co-occurrence)
C. BIO label layer (distribution, B/I ratio, alignment spot-check)
D. Noise layer (text length change, partial entities, UNK rate change)
E. Metadata layer (scenario, style, industry)

All results saved to results/eda/eda_report.json and results/eda/figures/.
"""

import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from transformers import AutoTokenizer

from src.models.transformer_ner import NERDataset, ID2LABEL
from src.utils.config import load_config, get_seed
from src.utils.logger import setup_logging, get_logger

log = get_logger(__name__)

FIGURES_DIR = Path("results/eda/figures")
REPORT_PATH = Path("results/eda/eda_report.json")


def _save_fig(fig: plt.Figure, name: str) -> None:
    """Save a figure and close it."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Figure saved", name=name)


# =========================================================================
# A. TEXT LAYER
# =========================================================================

def analyze_text_layer(
    prepared_samples: list[dict],
    tokenizer: AutoTokenizer,
    tokenizer_no_lc: AutoTokenizer,
) -> dict[str, Any]:
    """Analyze text characteristics and tokenizer behavior."""
    log.info("A. Analyzing text layer")
    results = {}

    # --- A1: Character length distribution ---
    char_lengths = [len(s["text"]) for s in prepared_samples]
    results["a1_char_lengths"] = {
        "min": min(char_lengths),
        "max": max(char_lengths),
        "mean": round(statistics.mean(char_lengths), 1),
        "median": round(statistics.median(char_lengths), 1),
        "stdev": round(statistics.stdev(char_lengths), 1),
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(char_lengths, bins=50, edgecolor="black", alpha=0.7)
    ax.set_title("A1: Text Length Distribution (Characters)")
    ax.set_xlabel("Character Count")
    ax.set_ylabel("Frequency")
    ax.axvline(statistics.mean(char_lengths), color="red", linestyle="--", label=f"Mean: {statistics.mean(char_lengths):.0f}")
    ax.legend()
    _save_fig(fig, "a1_char_length_distribution.png")

    # --- A2: Token length distribution ---
    token_lengths = [
        len(tokenizer(s["text"], truncation=False)["input_ids"])
        for s in prepared_samples
    ]
    percentiles = {
        "P90": round(np.percentile(token_lengths, 90), 0),
        "P95": round(np.percentile(token_lengths, 95), 0),
        "P99": round(np.percentile(token_lengths, 99), 0),
    }
    results["a2_token_lengths"] = {
        "min": min(token_lengths),
        "max": max(token_lengths),
        "mean": round(statistics.mean(token_lengths), 1),
        "median": round(statistics.median(token_lengths), 1),
        **percentiles,
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(token_lengths, bins=50, edgecolor="black", alpha=0.7)
    ax.set_title("A2: Token Length Distribution (BERT Tokenizer, do_lower_case=True)")
    ax.set_xlabel("Token Count")
    ax.set_ylabel("Frequency")
    for threshold in [128, 192, 256]:
        count = sum(1 for tl in token_lengths if tl > threshold)
        ax.axvline(threshold, color="red" if threshold == 128 else "orange", linestyle="--",
                    label=f">{threshold}: {count} samples ({count/len(token_lengths)*100:.1f}%)")
    ax.legend()
    _save_fig(fig, "a2_token_length_distribution.png")

    # --- A3: Truncation impact analysis ---
    results["a3_truncation"] = {}
    for max_len in [128, 192, 256, 512]:
        truncated = 0
        entities_lost = 0
        entity_types_lost = Counter()

        for s, n_tokens in zip(prepared_samples, token_lengths):
            if n_tokens <= max_len:
                continue
            truncated += 1

            # Find character position of last token within max_len
            enc = tokenizer(
                s["text"], max_length=max_len, truncation=True,
                return_offsets_mapping=True,
            )
            offsets = enc["offset_mapping"]
            # Last non-special-token offset
            last_char = 0
            for start, end in offsets:
                if end > last_char:
                    last_char = end

            for e in s.get("entities", []):
                if e["start"] >= last_char:
                    entities_lost += 1
                    entity_types_lost[e["label"]] += 1

        results["a3_truncation"][f"max_{max_len}"] = {
            "samples_truncated": truncated,
            "pct_truncated": f"{truncated/len(prepared_samples)*100:.1f}%",
            "entities_lost": entities_lost,
            "entity_types_lost": dict(entity_types_lost),
        }

    # --- A4: Vocabulary coverage (UNK rate) ---
    unk_with_lc = {"total_tokens": 0, "unk_tokens": 0, "unk_words": Counter()}
    unk_without_lc = {"total_tokens": 0, "unk_tokens": 0, "unk_words": Counter()}

    for s in prepared_samples:
        text = s["text"]
        words = text.split()

        # With do_lower_case
        tokens_lc = tokenizer.tokenize(text)
        unk_with_lc["total_tokens"] += len(tokens_lc)
        for tok in tokens_lc:
            if tok == "[UNK]":
                unk_with_lc["unk_tokens"] += 1

        # Without do_lower_case
        tokens_no_lc = tokenizer_no_lc.tokenize(text)
        unk_without_lc["total_tokens"] += len(tokens_no_lc)
        for i, tok in enumerate(tokens_no_lc):
            if tok == "[UNK]":
                unk_without_lc["unk_tokens"] += 1
                # Try to find which word caused it
                for w in words:
                    if tokenizer_no_lc.tokenize(w) == ["[UNK]"]:
                        unk_without_lc["unk_words"][w] += 1

    results["a4_vocab_coverage"] = {
        "with_lower_case": {
            "total_tokens": unk_with_lc["total_tokens"],
            "unk_tokens": unk_with_lc["unk_tokens"],
            "unk_rate": f"{unk_with_lc['unk_tokens']/max(unk_with_lc['total_tokens'],1)*100:.3f}%",
        },
        "without_lower_case": {
            "total_tokens": unk_without_lc["total_tokens"],
            "unk_tokens": unk_without_lc["unk_tokens"],
            "unk_rate": f"{unk_without_lc['unk_tokens']/max(unk_without_lc['total_tokens'],1)*100:.3f}%",
            "top_unk_words": dict(unk_without_lc["unk_words"].most_common(20)),
        },
    }

    # --- A5: UNK inside entities ---
    unk_in_entities = defaultdict(lambda: {"total_tokens": 0, "unk_tokens": 0})

    for s in prepared_samples:
        for e in s.get("entities", []):
            entity_text = e["text"]
            label = e["label"]

            tokens_lc = tokenizer.tokenize(entity_text)
            unk_in_entities[label]["total_tokens"] += len(tokens_lc)
            unk_in_entities[label]["unk_tokens"] += sum(1 for t in tokens_lc if t == "[UNK]")

    results["a5_unk_in_entities"] = {}
    for label, counts in unk_in_entities.items():
        rate = counts["unk_tokens"] / max(counts["total_tokens"], 1)
        results["a5_unk_in_entities"][label] = {
            "total_tokens": counts["total_tokens"],
            "unk_tokens": counts["unk_tokens"],
            "unk_rate": f"{rate*100:.2f}%",
        }

    return results


# =========================================================================
# B. ENTITY LAYER
# =========================================================================

def analyze_entity_layer(prepared_samples: list[dict], tokenizer: AutoTokenizer) -> dict[str, Any]:
    """Analyze entity characteristics."""
    log.info("B. Analyzing entity layer")
    results = {}

    all_entities = [e for s in prepared_samples for e in s.get("entities", [])]

    # --- B1: Entity type distribution ---
    type_counts = Counter(e["label"] for e in all_entities)
    results["b1_type_distribution"] = dict(type_counts.most_common())

    fig, ax = plt.subplots(figsize=(8, 5))
    types = list(type_counts.keys())
    counts = list(type_counts.values())
    ax.bar(types, counts, edgecolor="black", alpha=0.7)
    ax.set_title("B1: Entity Type Distribution")
    ax.set_ylabel("Count")
    for i, (t, c) in enumerate(zip(types, counts)):
        ax.text(i, c + 5, str(c), ha="center", fontweight="bold")
    _save_fig(fig, "b1_entity_type_distribution.png")

    # --- B2: Per-sample entity density ---
    densities = [len(s.get("entities", [])) for s in prepared_samples]
    density_counts = Counter(densities)
    results["b2_entity_density"] = {
        "distribution": dict(sorted(density_counts.items())),
        "mean": round(statistics.mean(densities), 2),
        "zero_entity_samples": density_counts.get(0, 0),
        "zero_entity_pct": f"{density_counts.get(0, 0)/len(prepared_samples)*100:.1f}%",
    }

    # --- B3: Entity character length by type ---
    char_lengths_by_type = defaultdict(list)
    for e in all_entities:
        char_lengths_by_type[e["label"]].append(len(e["text"]))

    results["b3_entity_char_lengths"] = {}
    for label, lengths in char_lengths_by_type.items():
        results["b3_entity_char_lengths"][label] = {
            "min": min(lengths),
            "max": max(lengths),
            "mean": round(statistics.mean(lengths), 1),
        }

    fig, ax = plt.subplots(figsize=(10, 5))
    data_for_box = [char_lengths_by_type.get(t, []) for t in type_counts.keys()]
    ax.boxplot(data_for_box, labels=list(type_counts.keys()))
    ax.set_title("B3: Entity Character Length by Type")
    ax.set_ylabel("Characters")
    _save_fig(fig, "b3_entity_char_lengths.png")

    # --- B4: Entity token length by type ---
    token_lengths_by_type = defaultdict(list)
    for e in all_entities:
        tokens = tokenizer.tokenize(e["text"])
        token_lengths_by_type[e["label"]].append(len(tokens))

    results["b4_entity_token_lengths"] = {}
    for label, lengths in token_lengths_by_type.items():
        results["b4_entity_token_lengths"][label] = {
            "min": min(lengths),
            "max": max(lengths),
            "mean": round(statistics.mean(lengths), 1),
            "note": f"Avg B:I ratio would be 1:{round(statistics.mean(lengths)-1, 1)}",
        }

    # --- B5: Entity position in text ---
    position_by_type = defaultdict(lambda: {"0-25%": 0, "25-50%": 0, "50-75%": 0, "75-100%": 0})
    for s in prepared_samples:
        text_len = max(len(s["text"]), 1)
        for e in s.get("entities", []):
            rel_pos = e["start"] / text_len
            if rel_pos < 0.25:
                bucket = "0-25%"
            elif rel_pos < 0.5:
                bucket = "25-50%"
            elif rel_pos < 0.75:
                bucket = "50-75%"
            else:
                bucket = "75-100%"
            position_by_type[e["label"]][bucket] += 1

    results["b5_entity_positions"] = {k: dict(v) for k, v in position_by_type.items()}

    # Heatmap
    entity_types = sorted(position_by_type.keys())
    buckets = ["0-25%", "25-50%", "50-75%", "75-100%"]
    matrix = np.array([
        [position_by_type[t][b] for b in buckets]
        for t in entity_types
    ], dtype=float)
    # Normalize per row
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_pct = np.where(row_sums > 0, matrix / row_sums * 100, 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(matrix_pct, annot=True, fmt=".0f", xticklabels=buckets,
                yticklabels=entity_types, cmap="YlOrRd", ax=ax)
    ax.set_title("B5: Entity Position in Text (% per type)")
    _save_fig(fig, "b5_entity_position_heatmap.png")

    # --- B6: Entities at text boundaries ---
    at_start = sum(1 for s in prepared_samples for e in s.get("entities", []) if e["start"] < 10)
    at_end = sum(
        1 for s in prepared_samples
        for e in s.get("entities", [])
        if e["end"] > len(s["text"]) - 10
    )
    results["b6_boundary_entities"] = {
        "at_start_lt_10_chars": at_start,
        "at_end_last_10_chars": at_end,
    }

    # --- B7: Entity format diversity ---
    email_formats = {"standard": 0, "oral": 0}
    contract_formats = {"standard": 0, "oral": 0}
    date_formats = {"absolute": 0, "relative": 0}
    name_casing = {"capitalized": 0, "lowercase": 0, "mixed": 0}

    relative_keywords = ["ago", "last", "yesterday", "earlier", "week", "month", "back", "before", "sometime"]

    for e in all_entities:
        if e["label"] == "EMAIL":
            if "@" in e["text"]:
                email_formats["standard"] += 1
            elif " at " in e["text"].lower():
                email_formats["oral"] += 1
        elif e["label"] == "CONTRACT_ID":
            if " dash " in e["text"].lower():
                contract_formats["oral"] += 1
            elif "-" in e["text"]:
                contract_formats["standard"] += 1
        elif e["label"] == "ISSUE_DATE":
            if any(k in e["text"].lower() for k in relative_keywords):
                date_formats["relative"] += 1
            else:
                date_formats["absolute"] += 1
        elif e["label"] == "NAME":
            if e["text"][0].isupper():
                name_casing["capitalized"] += 1
            elif e["text"].islower():
                name_casing["lowercase"] += 1
            else:
                name_casing["mixed"] += 1

    results["b7_format_diversity"] = {
        "email": email_formats,
        "contract_id": contract_formats,
        "issue_date": date_formats,
        "name_casing": name_casing,
    }

    # --- B8: Entity value uniqueness ---
    values_by_type = defaultdict(list)
    for e in all_entities:
        values_by_type[e["label"]].append(e["text"])

    results["b8_uniqueness"] = {}
    for label, values in values_by_type.items():
        counter = Counter(values)
        results["b8_uniqueness"][label] = {
            "total": len(values),
            "unique": len(set(values)),
            "top_5": counter.most_common(5),
        }

    # --- B9: Entity co-occurrence ---
    cooccurrence = defaultdict(int)
    for s in prepared_samples:
        types_in_sample = set(e["label"] for e in s.get("entities", []))
        for t1 in types_in_sample:
            for t2 in types_in_sample:
                if t1 <= t2:
                    cooccurrence[(t1, t2)] += 1

    results["b9_cooccurrence"] = {f"{t1}+{t2}": c for (t1, t2), c in cooccurrence.items()}

    # --- B10: Adjacent entity distance ---
    distances = []
    for s in prepared_samples:
        ents = sorted(s.get("entities", []), key=lambda e: e["start"])
        for i in range(len(ents) - 1):
            gap = ents[i + 1]["start"] - ents[i]["end"]
            distances.append(gap)

    if distances:
        close_pairs = sum(1 for d in distances if 0 <= d < 5)
        results["b10_adjacency"] = {
            "total_adjacent_pairs": len(distances),
            "mean_distance": round(statistics.mean(distances), 1),
            "min_distance": min(distances),
            "pairs_within_5_chars": close_pairs,
        }
    else:
        results["b10_adjacency"] = {"total_adjacent_pairs": 0}

    return results


# =========================================================================
# C. BIO LABEL LAYER
# =========================================================================

def analyze_bio_layer(prepared_samples: list[dict], tokenizer: AutoTokenizer) -> dict[str, Any]:
    """Analyze BIO label distribution using actual tokenizer and alignment."""
    log.info("C. Analyzing BIO label layer")
    results = {}

    # Encode a subset for efficiency
    subset = prepared_samples[:300]
    dataset = NERDataset(subset, tokenizer, max_length=256)

    # --- C1: Full BIO label distribution ---
    label_counts = Counter()
    for i in range(len(dataset)):
        labels = dataset[i]["labels"]
        for label_id in labels.tolist():
            if label_id == -100:
                continue
            label_counts[ID2LABEL[label_id]] += 1

    total = sum(label_counts.values())
    results["c1_label_distribution"] = {
        label: {"count": count, "pct": f"{count/total*100:.2f}%"}
        for label, count in label_counts.most_common()
    }

    # --- C2: B vs I ratio per entity type ---
    entity_types = ["NAME", "EMAIL", "CONTRACT_ID", "PRODUCT", "ISSUE_DATE"]
    results["c2_bi_ratio"] = {}
    for etype in entity_types:
        b_count = label_counts.get(f"B-{etype}", 0)
        i_count = label_counts.get(f"I-{etype}", 0)
        results["c2_bi_ratio"][etype] = {
            "B": b_count,
            "I": i_count,
            "ratio": f"1:{round(i_count/max(b_count,1), 1)}",
            "avg_entity_tokens": round((b_count + i_count) / max(b_count, 1), 1),
        }

    # --- C3: Class imbalance ---
    non_o_counts = {tag: c for tag, c in label_counts.items() if tag != "O"}
    if non_o_counts:
        max_class = max(non_o_counts.values())
        min_class = min(non_o_counts.values())
        results["c3_imbalance"] = {
            "O_pct": f"{label_counts.get('O', 0)/total*100:.1f}%",
            "max_non_O": max(non_o_counts, key=non_o_counts.get),
            "min_non_O": min(non_o_counts, key=non_o_counts.get),
            "max_min_ratio": f"{max_class/max(min_class,1):.1f}x",
        }

    # --- C4: Alignment spot-check ---
    rng = random.Random(get_seed())
    entity_samples = [s for s in subset if len(s.get("entities", [])) >= 2]
    spot_checks = []

    for s in rng.sample(entity_samples, min(10, len(entity_samples))):
        encoding = tokenizer(
            s["text"], max_length=256, truncation=True,
            return_offsets_mapping=True, return_tensors="pt",
        )
        offset_list = encoding.offset_mapping[0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

        # Recreate labels using dataset logic
        ds = NERDataset([s], tokenizer, max_length=256)
        labels = ds[0]["labels"].tolist()

        aligned = []
        for tok, lab_id, (start, end) in zip(tokens, labels, offset_list):
            if lab_id == -100:
                continue
            lab = ID2LABEL[lab_id]
            if lab != "O":
                aligned.append({
                    "token": tok,
                    "label": lab,
                    "char_span": f"[{start}:{end}]",
                    "source_text": s["text"][start:end] if end <= len(s["text"]) else "?",
                })

        spot_checks.append({
            "text_preview": s["text"][:80] + "...",
            "entities": [{"text": e["text"], "label": e["label"]} for e in s["entities"]],
            "token_label_alignment": aligned,
        })

    results["c4_spot_checks"] = spot_checks

    return results


# =========================================================================
# D. NOISE LAYER
# =========================================================================

def analyze_noise_layer(
    prepared_samples: list[dict],
    noisy_samples: list[dict],
    tokenizer: AutoTokenizer,
) -> dict[str, Any]:
    """Analyze noise impact on text and entities."""
    log.info("D. Analyzing noise layer")
    results = {}

    by_level = defaultdict(list)
    for s in noisy_samples:
        by_level[s.get("noise_level", "unknown")].append(s)

    # --- D1: Text length change across noise levels ---
    results["d1_length_change"] = {}
    prepared_lengths = [len(s["text"]) for s in prepared_samples]
    prepared_mean = statistics.mean(prepared_lengths)

    for level in ["clean", "mild", "moderate", "severe"]:
        if level not in by_level:
            continue
        lengths = [len(s["text"]) for s in by_level[level]]
        results["d1_length_change"][level] = {
            "mean_length": round(statistics.mean(lengths), 1),
            "change_from_prepared": f"{(statistics.mean(lengths) - prepared_mean)/prepared_mean*100:+.1f}%",
        }

    # --- D2: Partial entity rate by noise level × entity type ---
    results["d2_partial_entities"] = {}
    entity_types = ["NAME", "EMAIL", "CONTRACT_ID", "PRODUCT", "ISSUE_DATE"]

    for level in ["clean", "mild", "moderate", "severe"]:
        if level not in by_level:
            continue
        level_data = {}
        for etype in entity_types:
            total = 0
            partial = 0
            for s in by_level[level]:
                for e in s.get("entities", []):
                    if e["label"] == etype:
                        total += 1
                        if e.get("partial", False):
                            partial += 1
            level_data[etype] = {
                "total": total,
                "partial": partial,
                "partial_rate": f"{partial/max(total,1)*100:.1f}%",
            }
        results["d2_partial_entities"][level] = level_data

    # D2 heatmap
    levels = ["clean", "mild", "moderate", "severe"]
    matrix = np.zeros((len(levels), len(entity_types)))
    for i, level in enumerate(levels):
        for j, etype in enumerate(entity_types):
            data = results["d2_partial_entities"].get(level, {}).get(etype, {})
            total = data.get("total", 0)
            partial = data.get("partial", 0)
            matrix[i, j] = partial / max(total, 1) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(matrix, annot=True, fmt=".1f", xticklabels=entity_types,
                yticklabels=levels, cmap="Reds", ax=ax)
    ax.set_title("D2: Partial Entity Rate (%) by Noise Level × Entity Type")
    _save_fig(fig, "d2_partial_entity_heatmap.png")

    # --- D3: UNK rate change across noise levels ---
    results["d3_unk_by_noise"] = {}
    for level in ["clean", "mild", "moderate", "severe"]:
        if level not in by_level:
            continue
        total_tokens = 0
        unk_tokens = 0
        for s in by_level[level][:200]:  # Sample for speed
            tokens = tokenizer.tokenize(s["text"])
            total_tokens += len(tokens)
            unk_tokens += sum(1 for t in tokens if t == "[UNK]")
        results["d3_unk_by_noise"][level] = {
            "total_tokens": total_tokens,
            "unk_tokens": unk_tokens,
            "unk_rate": f"{unk_tokens/max(total_tokens,1)*100:.3f}%",
        }

    return results


# =========================================================================
# E. METADATA LAYER
# =========================================================================

def analyze_metadata(prepared_samples: list[dict]) -> dict[str, Any]:
    """Analyze scenario, style, and industry distribution."""
    log.info("E. Analyzing metadata layer")
    results = {}

    samples_with_meta = [s for s in prepared_samples if "metadata" in s]

    results["e1_scenarios"] = dict(Counter(
        s["metadata"].get("scenario", "unknown") for s in samples_with_meta
    ).most_common())

    results["e2_styles"] = dict(Counter(
        s["metadata"].get("style", "unknown") for s in samples_with_meta
    ).most_common())

    results["e3_industries"] = dict(Counter(
        s["metadata"].get("industry", "unknown") for s in samples_with_meta
    ).most_common())

    return results


# =========================================================================
# MAIN
# =========================================================================

def main() -> None:
    setup_logging()
    log = get_logger(__name__)
    config = load_config()

    model_name = config["training"]["bert_tiny"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
    tokenizer_no_lc = AutoTokenizer.from_pretrained(model_name)

    # Load data
    prepared_path = Path("data/synthetic/prepared/train_prepared.json")
    noisy_path = Path("data/noisy/train_noisy.json")

    with open(prepared_path) as f:
        prepared_samples = json.load(f)
    log.info("Prepared data loaded", samples=len(prepared_samples))

    noisy_samples = []
    if noisy_path.exists():
        with open(noisy_path) as f:
            noisy_samples = json.load(f)
        log.info("Noisy data loaded", samples=len(noisy_samples))

    # Run all analyses
    report = {}
    report["text"] = analyze_text_layer(prepared_samples, tokenizer, tokenizer_no_lc)
    report["entity"] = analyze_entity_layer(prepared_samples, tokenizer)
    report["bio"] = analyze_bio_layer(prepared_samples, tokenizer)
    if noisy_samples:
        report["noise"] = analyze_noise_layer(prepared_samples, noisy_samples, tokenizer)
    report["metadata"] = analyze_metadata(prepared_samples)

    # Save report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 70)
    print("EDA SUMMARY")
    print("=" * 70)

    # Text layer
    t = report["text"]
    print("\n--- A. TEXT LAYER ---")
    print(f"  A1 Char lengths: min={t['a1_char_lengths']['min']} max={t['a1_char_lengths']['max']} mean={t['a1_char_lengths']['mean']}")
    print(f"  A2 Token lengths: min={t['a2_token_lengths']['min']} max={t['a2_token_lengths']['max']} mean={t['a2_token_lengths']['mean']} P95={t['a2_token_lengths']['P95']}")
    print("  A3 Truncation:")
    for k, v in t["a3_truncation"].items():
        print(f"     {k}: {v['samples_truncated']} samples, {v['entities_lost']} entities lost")
    print(f"  A4 UNK rate (with lowercase): {t['a4_vocab_coverage']['with_lower_case']['unk_rate']}")
    print(f"  A4 UNK rate (without lowercase): {t['a4_vocab_coverage']['without_lower_case']['unk_rate']}")
    if t['a4_vocab_coverage']['without_lower_case']['top_unk_words']:
        top_unk = list(t['a4_vocab_coverage']['without_lower_case']['top_unk_words'].items())[:5]
        print(f"     Top UNK words (no lowercase): {top_unk}")
    print("  A5 UNK in entities (with lowercase):")
    for label, info in t["a5_unk_in_entities"].items():
        print(f"     {label}: {info['unk_rate']}")

    # Entity layer
    e = report["entity"]
    print("\n--- B. ENTITY LAYER ---")
    print(f"  B1 Type distribution: {e['b1_type_distribution']}")
    print(f"  B2 Density: mean={e['b2_entity_density']['mean']} zero={e['b2_entity_density']['zero_entity_pct']}")
    print(f"  B6 Boundary: start={e['b6_boundary_entities']['at_start_lt_10_chars']} end={e['b6_boundary_entities']['at_end_last_10_chars']}")
    print("  B7 Format diversity:")
    for fmt_type, fmt_data in e["b7_format_diversity"].items():
        print(f"     {fmt_type}: {fmt_data}")
    print("  B8 Uniqueness:")
    for label, info in e["b8_uniqueness"].items():
        print(f"     {label}: {info['unique']} unique / {info['total']} total")

    # BIO layer
    b = report["bio"]
    print("\n--- C. BIO LABEL LAYER ---")
    print(f"  C1 Distribution (top 5): {dict(list(b['c1_label_distribution'].items())[:5])}")
    print("  C2 B:I ratio:")
    for etype, info in b["c2_bi_ratio"].items():
        print(f"     {etype}: {info['ratio']} (avg {info['avg_entity_tokens']} tokens/entity)")
    print(f"  C3 Imbalance: O={b['c3_imbalance']['O_pct']} max/min={b['c3_imbalance']['max_min_ratio']}")
    print(f"  C4 Spot checks: {len(b['c4_spot_checks'])} samples checked")
    # Print first spot check
    if b["c4_spot_checks"]:
        sc = b["c4_spot_checks"][0]
        print(f"     Sample: {sc['text_preview']}")
        print(f"     Entities: {sc['entities']}")
        print("     Alignment:")
        for a in sc["token_label_alignment"][:8]:
            print(f"       {a['token']:<15} → {a['label']:<15} {a['source_text']}")

    # Noise layer
    if "noise" in report:
        n = report["noise"]
        print("\n--- D. NOISE LAYER ---")
        print(f"  D1 Length change: {n['d1_length_change']}")
        print("  D3 UNK by noise:")
        for level, info in n["d3_unk_by_noise"].items():
            print(f"     {level}: {info['unk_rate']}")

    # Metadata
    m = report["metadata"]
    print("\n--- E. METADATA ---")
    print(f"  E1 Scenarios: {m['e1_scenarios']}")
    print(f"  E2 Styles: {m['e2_styles']}")

    # Flag anomalies
    print("\n--- ANOMALIES / ACTION ITEMS ---")
    anomalies = []

    trunc_128 = t["a3_truncation"]["max_128"]
    if trunc_128["samples_truncated"] > 0:
        anomalies.append(f"⚠️  {trunc_128['samples_truncated']} samples truncated at 128 tokens, {trunc_128['entities_lost']} entities lost")

    unk_rate_lc = float(t["a4_vocab_coverage"]["with_lower_case"]["unk_rate"].strip("%"))
    unk_rate_no_lc = float(t["a4_vocab_coverage"]["without_lower_case"]["unk_rate"].strip("%"))
    if unk_rate_no_lc > 1.0:
        anomalies.append(f"⚠️  UNK rate without lowercase: {unk_rate_no_lc}% — do_lower_case=True is critical")
    if unk_rate_lc > 0.5:
        anomalies.append(f"⚠️  UNK rate with lowercase: {unk_rate_lc}% — some words still unknown")

    if e["b6_boundary_entities"]["at_start_lt_10_chars"] < 5:
        anomalies.append(f"⚠️  Only {e['b6_boundary_entities']['at_start_lt_10_chars']} entities at text start — expand gold set with start-position entities")

    if not anomalies:
        print("  ✓ No critical anomalies detected")
    else:
        for a in anomalies:
            print(f"  {a}")

    print("=" * 70 + "\n")
    log.info("EDA complete", report_path=str(REPORT_PATH))


if __name__ == "__main__":
    main()