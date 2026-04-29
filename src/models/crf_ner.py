"""CRF baseline NER model — feature-engineered Conditional Random Field.

Provides a traditional ML baseline for comparison with BERT fine-tuning.
Uses hand-crafted word-level features (shape, prefix/suffix, context)
with sklearn-crfsuite.

The CRF operates at word level (space-tokenized), not subword level.
Character offsets are reconstructed from word positions after prediction.
"""

import re
from typing import Any

import sklearn_crfsuite

from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _word_shape(word: str) -> str:
    """Classify word shape: all_lower, all_upper, title_case, mixed, digit, other."""
    if word.isdigit():
        return "digit"
    if word.isalpha():
        if word.islower():
            return "all_lower"
        if word.isupper():
            return "all_upper"
        if word[0].isupper() and word[1:].islower():
            return "title_case"
        return "mixed"
    if any(c.isdigit() for c in word) and any(c.isalpha() for c in word):
        return "alnum_mixed"
    return "other"


def _has_hyphen(word: str) -> bool:
    return "-" in word


def _has_at(word: str) -> bool:
    return "@" in word


def _has_dot(word: str) -> bool:
    return "." in word


def _word_features(sent: list[str], i: int) -> dict[str, Any]:
    """Extract features for word at position i in sentence."""
    word = sent[i]
    word_lower = word.lower()

    features = {
        "bias": 1.0,
        "word.lower()": word_lower,
        "word[-3:]": word_lower[-3:] if len(word_lower) >= 3 else word_lower,
        "word[-2:]": word_lower[-2:] if len(word_lower) >= 2 else word_lower,
        "word[:3]": word_lower[:3] if len(word_lower) >= 3 else word_lower,
        "word[:2]": word_lower[:2] if len(word_lower) >= 2 else word_lower,
        "word.shape": _word_shape(word),
        "word.isdigit": word.isdigit(),
        "word.isalpha": word.isalpha(),
        "word.istitle": word.istitle(),
        "word.isupper": word.isupper(),
        "word.has_hyphen": _has_hyphen(word),
        "word.has_at": _has_at(word),
        "word.has_dot": _has_dot(word),
        "word.length": min(len(word), 20),
    }

    # Contextual features: previous word
    if i > 0:
        prev = sent[i - 1]
        prev_lower = prev.lower()
        features.update({
            "-1:word.lower()": prev_lower,
            "-1:word.shape": _word_shape(prev),
            "-1:word.istitle": prev.istitle(),
        })
    else:
        features["BOS"] = True

    # Contextual features: next word
    if i < len(sent) - 1:
        nxt = sent[i + 1]
        nxt_lower = nxt.lower()
        features.update({
            "+1:word.lower()": nxt_lower,
            "+1:word.shape": _word_shape(nxt),
            "+1:word.istitle": nxt.istitle(),
        })
    else:
        features["EOS"] = True

    # Second-order context
    if i > 1:
        features["-2:word.lower()"] = sent[i - 2].lower()
    if i < len(sent) - 2:
        features["+2:word.lower()"] = sent[i + 2].lower()

    # Trigger phrase features
    if i > 0:
        bigram = sent[i - 1].lower() + " " + word_lower
        features["bigram"] = bigram
        # Common NER triggers
        if sent[i - 1].lower() in ("name", "called", "is", "im", "am"):
            features["after_name_trigger"] = True
        if sent[i - 1].lower() in ("order", "reference", "number", "id", "contract"):
            features["after_id_trigger"] = True
        if sent[i - 1].lower() in ("email", "mail", "address"):
            features["after_email_trigger"] = True
        if word_lower in ("at", "dash", "dot"):
            features["oral_format_word"] = True

    return features


# ---------------------------------------------------------------------------
# Data conversion: char-offset NER samples → word-level BIO sequences
# ---------------------------------------------------------------------------

def _tokenize_simple(text: str) -> list[tuple[str, int, int]]:
    """Simple space-based tokenization with character offsets.

    Returns list of (word, start_char, end_char).
    """
    tokens = []
    for match in re.finditer(r"\S+", text):
        tokens.append((match.group(), match.start(), match.end()))
    return tokens


def _assign_bio_labels(
    tokens: list[tuple[str, int, int]],
    entities: list[dict],
) -> list[str]:
    """Assign BIO labels to word tokens based on character-level entity spans.

    A word is labeled as part of an entity if it overlaps with the entity span.
    The first overlapping word gets B-, subsequent words get I-.
    """
    labels = ["O"] * len(tokens)

    for ent in entities:
        ent_start = ent["start"]
        ent_end = ent["end"]
        ent_label = ent["label"]
        first = True

        for i, (word, w_start, w_end) in enumerate(tokens):
            # Check overlap
            overlap = max(0, min(w_end, ent_end) - max(w_start, ent_start))
            word_len = w_end - w_start
            # Word is part of entity if >50% of the word overlaps
            if overlap > word_len * 0.5:
                if first:
                    labels[i] = f"B-{ent_label}"
                    first = False
                else:
                    labels[i] = f"I-{ent_label}"

    return labels


def _samples_to_sequences(
    samples: list[dict],
) -> tuple[list[list[dict]], list[list[str]]]:
    """Convert NER samples to CRF feature sequences and label sequences.

    Returns:
        X: list of feature sequences (each is list of feature dicts)
        y: list of label sequences (each is list of BIO strings)
    """
    X = []
    y = []

    for sample in samples:
        text = sample["text"]
        entities = sample.get("entities", [])
        # Keep all entities including partial — matches BERT training decision

        tokens = _tokenize_simple(text)
        if not tokens:
            continue

        words = [t[0] for t in tokens]
        labels = _assign_bio_labels(tokens, entities)

        features = [_word_features(words, i) for i in range(len(words))]
        X.append(features)
        y.append(labels)

    return X, y


# ---------------------------------------------------------------------------
# CRF model class
# ---------------------------------------------------------------------------

class CRFModel:
    """CRF NER model with word-level features."""

    def __init__(self) -> None:
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True,
        )
        self._trained = False

    def train(
        self,
        train_samples: list[dict],
        val_samples: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Train CRF on NER samples.

        Args:
            train_samples: Training data with text and entities.
            val_samples: Optional validation data.

        Returns:
            Training statistics dict.
        """
        log.info("Converting training data to CRF sequences")
        X_train, y_train = _samples_to_sequences(train_samples)
        log.info("Training CRF", sequences=len(X_train))

        self.model.fit(X_train, y_train)
        self._trained = True

        stats = {
            "train_sequences": len(X_train),
            "labels": list(self.model.classes_),
        }

        if val_samples:
            X_val, y_val = _samples_to_sequences(val_samples)
            y_pred = self.model.predict(X_val)

            # Overall token-level accuracy
            correct = sum(
                p == g
                for pred_seq, gold_seq in zip(y_pred, y_val)
                for p, g in zip(pred_seq, gold_seq)
            )
            total = sum(len(seq) for seq in y_val)
            stats["val_token_accuracy"] = round(correct / max(total, 1), 4)
            stats["val_sequences"] = len(X_val)
            log.info("Validation", token_accuracy=stats["val_token_accuracy"])

        log.info("CRF training complete", labels=len(self.model.classes_))
        return stats

    def predict(self, text: str) -> list[dict[str, Any]]:
        """Extract entities from text.

        Args:
            text: Input text string.

        Returns:
            List of entity dicts with text, label, start, end.
        """
        if not self._trained:
            raise RuntimeError("CRF model not trained. Call train() first.")

        tokens = _tokenize_simple(text)
        if not tokens:
            return []

        words = [t[0] for t in tokens]
        features = [_word_features(words, i) for i in range(len(words))]
        predictions = self.model.predict([features])[0]

        # Convert BIO predictions back to entity spans
        entities = []
        current_entity = None

        for (word, w_start, w_end), label in zip(tokens, predictions):
            if label == "O":
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
                continue

            entity_type = label[2:]  # Strip B- or I-

            is_new = False
            if label.startswith("B-"):
                is_new = True
            elif label.startswith("I-"):
                if current_entity is None:
                    is_new = True
                elif current_entity["label"] != entity_type:
                    is_new = True

            if is_new:
                if current_entity is not None:
                    entities.append(current_entity)
                current_entity = {
                    "text": text[w_start:w_end],
                    "label": entity_type,
                    "start": w_start,
                    "end": w_end,
                    "confidence": 1.0,  # CRF has no probability output
                }
            else:
                if current_entity is not None:
                    current_entity["end"] = w_end
                    current_entity["text"] = text[current_entity["start"]:w_end]

        if current_entity is not None:
            entities.append(current_entity)

        return entities
