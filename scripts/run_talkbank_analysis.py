"""Calibrate noise engine filler parameters against real conversational speech.

Analyzes the public TalkBank dataset on HuggingFace (English split) to extract
real-world filler word frequencies and disfluency patterns from human-transcribed
phone conversations. Uses these statistics to calibrate the noise engine's filler
insertion parameters.

Depends on: datasets library, HuggingFace access to the TalkBank dataset
Output: results/talkbank_analysis.json

Usage:
    python scripts/run_talkbank_analysis.py
"""

import json
import re
from collections import Counter
from pathlib import Path

from src.utils.logger import setup_logging, get_logger


# Filler words to search for — covers common English conversational fillers
FILLER_WORDS = {
    "uh", "uhh", "uhhh", "um", "umm", "ummm", "uhm",
    "er", "erm", "eh",
    "hm", "hmm", "hmmm",
    "mhm", "mmhm",
    "huh", "hhh", "hha",
    "like", "right", "well", "so",
}

# Multi-word fillers
MULTI_WORD_FILLERS = [
    "you know", "i mean", "uh huh", "kind of", "sort of",
]

# Disfluency markers in TalkBank CHAT notation
CHAT_MARKERS = {
    "⌈": "overlap_start",
    "⌉": "overlap_end",
    "::": "lengthening",
}

# Repetition/stutter pattern
STUTTER_PATTERN = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)


def load_talkbank_transcripts() -> list[str]:
    """Load English transcripts from TalkBank dataset."""
    from datasets import load_dataset
    ds = load_dataset("diabolocom/talkbank_4_stt", "en")
    seg = ds["segment"].select_columns(["transcript"])
    return [seg[i]["transcript"] for i in range(len(seg))]


def analyze_fillers(transcripts: list[str]) -> dict:
    """Count single-word and multi-word filler frequencies."""
    total_words = 0
    filler_counts = Counter()

    for t in transcripts:
        words = t.lower().split()
        total_words += len(words)

        # Single-word fillers
        for w in words:
            clean = re.sub(r"[^a-z]", "", w)
            if clean in FILLER_WORDS:
                filler_counts[clean] += 1

    # Multi-word fillers (scan raw text)
    multi_counts = Counter()
    full_text = " ".join(t.lower() for t in transcripts)
    for phrase in MULTI_WORD_FILLERS:
        count = full_text.count(phrase)
        if count > 0:
            multi_counts[phrase] = count

    total_fillers = sum(filler_counts.values()) + sum(multi_counts.values())

    return {
        "total_words": total_words,
        "total_fillers": total_fillers,
        "filler_rate": round(total_fillers / max(total_words, 1), 5),
        "single_word_fillers": dict(filler_counts.most_common(30)),
        "multi_word_fillers": dict(multi_counts.most_common(10)),
    }


def analyze_disfluencies(transcripts: list[str]) -> dict:
    """Analyze disfluency patterns: repetitions, false starts, overlaps."""
    stutter_count = 0
    overlap_count = 0
    lengthening_count = 0
    total_transcripts = len(transcripts)

    for t in transcripts:
        # Stutters / word repetitions
        stutter_count += len(STUTTER_PATTERN.findall(t))
        # Overlap markers
        overlap_count += t.count("⌈")
        # Lengthening
        lengthening_count += t.count("::")

    return {
        "total_transcripts": total_transcripts,
        "stutters": stutter_count,
        "stutter_rate_per_transcript": round(stutter_count / max(total_transcripts, 1), 4),
        "overlaps": overlap_count,
        "overlap_rate_per_transcript": round(overlap_count / max(total_transcripts, 1), 4),
        "lengthenings": lengthening_count,
        "lengthening_rate_per_transcript": round(lengthening_count / max(total_transcripts, 1), 4),
    }


def analyze_transcript_structure(transcripts: list[str]) -> dict:
    """Analyze basic transcript structure: length, word count distributions."""
    lengths = [len(t) for t in transcripts]
    word_counts = [len(t.split()) for t in transcripts]

    def percentiles(values: list) -> dict:
        s = sorted(values)
        n = len(s)
        return {
            "min": s[0],
            "p25": s[n // 4],
            "median": s[n // 2],
            "p75": s[3 * n // 4],
            "max": s[-1],
            "mean": round(sum(s) / n, 1),
        }

    return {
        "char_length": percentiles(lengths),
        "word_count": percentiles(word_counts),
    }


def compute_noise_params(filler_analysis: dict) -> dict:
    """Derive recommended noise engine parameters from TalkBank statistics.

    Compares observed rates against current heuristic settings and
    provides calibrated values.
    """
    observed_filler_rate = filler_analysis["filler_rate"]

    # Current heuristic settings from config/settings.yaml
    current_params = {
        "mild_filler_prob": 0.1,
        "moderate_filler_prob": 0.2,
        "severe_filler_prob": 0.3,
    }

    # Top fillers by frequency — use for weighted filler selection
    top_fillers = list(filler_analysis["single_word_fillers"].keys())[:10]

    # Recommended: scale filler probs relative to observed rate
    # Observed rate is the "natural" baseline. Mild = 1x, moderate = 2x, severe = 3x
    recommended_params = {
        "mild_filler_prob": round(observed_filler_rate, 4),
        "moderate_filler_prob": round(observed_filler_rate * 2, 4),
        "severe_filler_prob": round(observed_filler_rate * 3, 4),
        "filler_words_weighted": top_fillers,
    }

    return {
        "observed_filler_rate": observed_filler_rate,
        "current_heuristic_params": current_params,
        "recommended_params": recommended_params,
    }


def main() -> None:
    setup_logging()
    log = get_logger(__name__)

    log.info("Loading TalkBank English transcripts")
    transcripts = load_talkbank_transcripts()
    log.info("Loaded transcripts", count=len(transcripts))

    # Filler analysis
    log.info("Analyzing filler words")
    filler_results = analyze_fillers(transcripts)

    # Disfluency analysis
    log.info("Analyzing disfluency patterns")
    disfluency_results = analyze_disfluencies(transcripts)

    # Transcript structure
    log.info("Analyzing transcript structure")
    structure_results = analyze_transcript_structure(transcripts)

    # Derive noise params
    log.info("Computing calibrated noise parameters")
    noise_params = compute_noise_params(filler_results)

    # Compile results
    results = {
        "dataset": "TalkBank (en, segment split, HuggingFace)",
        "total_transcripts": len(transcripts),
        "fillers": filler_results,
        "disfluencies": disfluency_results,
        "transcript_structure": structure_results,
        "noise_calibration": noise_params,
    }

    # Save
    output_path = Path("results/talkbank_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved", path=str(output_path))

    # Print summary
    print("\n" + "=" * 65)
    print("TALKBANK FILLER / DISFLUENCY ANALYSIS")
    print("=" * 65)
    print(f"Transcripts analyzed: {len(transcripts)}")
    print(f"Total words: {filler_results['total_words']}")
    print()

    print("--- Filler Words ---")
    print(f"Total fillers: {filler_results['total_fillers']}")
    print(f"Filler rate: {filler_results['filler_rate']:.4f} ({filler_results['filler_rate']*100:.2f}%)")
    print()
    print("Top 15 single-word fillers:")
    for filler, count in list(filler_results["single_word_fillers"].items())[:15]:
        pct = count / filler_results["total_words"] * 100
        print(f"  {filler:<12} {count:>6} ({pct:.2f}%)")
    print()
    if filler_results["multi_word_fillers"]:
        print("Multi-word fillers:")
        for phrase, count in filler_results["multi_word_fillers"].items():
            print(f"  {phrase:<15} {count:>6}")
        print()

    print("--- Disfluencies ---")
    print(f"Stutters (word repetitions): {disfluency_results['stutters']} ({disfluency_results['stutter_rate_per_transcript']:.3f}/transcript)")
    print(f"Overlaps: {disfluency_results['overlaps']} ({disfluency_results['overlap_rate_per_transcript']:.3f}/transcript)")
    print(f"Lengthenings: {disfluency_results['lengthenings']} ({disfluency_results['lengthening_rate_per_transcript']:.3f}/transcript)")
    print()

    print("--- Noise Calibration ---")
    obs = noise_params["observed_filler_rate"]
    print(f"Observed filler rate: {obs:.4f}")
    print(f"Current heuristic:    mild={noise_params['current_heuristic_params']['mild_filler_prob']}, moderate={noise_params['current_heuristic_params']['moderate_filler_prob']}, severe={noise_params['current_heuristic_params']['severe_filler_prob']}")
    rec = noise_params["recommended_params"]
    print(f"Recommended:          mild={rec['mild_filler_prob']}, moderate={rec['moderate_filler_prob']}, severe={rec['severe_filler_prob']}")
    print(f"Top fillers (weighted): {rec['filler_words_weighted']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
