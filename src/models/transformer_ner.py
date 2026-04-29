"""Fine-tuned transformer NER — BERT-Tiny / BERT-Small token classification.

Trains a lightweight token classification model on synthetic NER data.
Uses BIOES tagging scheme: B-ENTITY for beginning, I-ENTITY for inside, E-ENTITY for end, S-ENTITY for single-token, O for outside.
Designed for CPU inference with minimal latency.
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

from src.utils.config import load_config, get_seed
from src.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Label scheme
# ---------------------------------------------------------------------------

BIOES_LABELS = [
    "O",
    "B-NAME", "I-NAME", "E-NAME", "S-NAME",
    "B-EMAIL", "I-EMAIL", "E-EMAIL", "S-EMAIL",
    "B-CONTRACT_ID", "I-CONTRACT_ID", "E-CONTRACT_ID", "S-CONTRACT_ID",
    "B-PRODUCT", "I-PRODUCT", "E-PRODUCT", "S-PRODUCT",
    "B-ISSUE_DATE", "I-ISSUE_DATE", "E-ISSUE_DATE", "S-ISSUE_DATE",
]

LABEL2ID = {label: i for i, label in enumerate(BIOES_LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NERDataset(Dataset):
    """Token classification dataset with BIO tagging.

    Converts character-level entity annotations to token-level BIO labels
    aligned with the tokenizer's sub-word tokenization.
    """

    def __init__(
        self,
        samples: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 128,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._encoded = self._encode_all()

    def _entities_to_char_labels(self, text: str, entities: list[dict]) -> list[str]:
        """Convert entity annotations to per-character BIOES labels.

        Args:
            text: The input text.
            entities: List of entity dicts with start, end, label.

        Returns:
            List of BIOES labels, one per character.
            S-X for single-character entities, B-X/I-X/E-X for multi-character.
        """
        char_labels = ["O"] * len(text)

        for ent in entities:
            start = ent["start"]
            end = ent["end"]
            label = ent["label"]

            if start < 0 or end > len(text) or start >= end:
                continue

            span_len = end - start
            if span_len == 1:
                char_labels[start] = f"S-{label}"
            else:
                char_labels[start] = f"B-{label}"
                for i in range(start + 1, end - 1):
                    char_labels[i] = f"I-{label}"
                char_labels[end - 1] = f"E-{label}"

        return char_labels

    def _align_labels_with_tokens(
        self,
        offset_mapping: list[list[int]],
        char_labels: list[str],
    ) -> list[int]:
        """Align character-level BIOES labels to sub-word tokens.

        Uses a two-step approach:
        1. For each token, determine entity type and whether it starts a new
           entity (B- or S- at first character = new entity start).
        2. Group consecutive same-entity tokens into spans and assign correct
           BIOES labels at the token level.

        This handles subword tokenization correctly: the last token of an
        entity gets E- even if its first character was I- at char level.
        Adjacent same-type entities are separated by the is_start flag.
        """
        # Step 1: Determine entity type and boundary for each token
        token_info: list[tuple] = []  # (entity_type_or_O_or_None, is_new_start)

        for start, end in offset_mapping:
            if start == 0 and end == 0:
                token_info.append((None, False))
                continue

            char_label = char_labels[start] if start < len(char_labels) else "O"

            if char_label == "O":
                token_info.append(("O", False))
            elif char_label.startswith("B-") or char_label.startswith("S-"):
                entity_type = char_label.split("-", 1)[1]
                token_info.append((entity_type, True))
            else:  # I- or E-
                entity_type = char_label.split("-", 1)[1]
                token_info.append((entity_type, False))

        # Step 2: Group consecutive same-entity tokens, assign BIOES
        token_labels: list[int] = []
        n = len(token_info)
        i = 0

        while i < n:
            etype, is_start = token_info[i]

            if etype is None:
                token_labels.append(-100)
                i += 1
                continue

            if etype == "O":
                token_labels.append(LABEL2ID["O"])
                i += 1
                continue

            # Entity token — find span extent
            # Stop at: O, special, different type, or new start of same type
            span_start = i
            j = i + 1
            while j < n:
                next_etype, next_is_start = token_info[j]
                if next_etype != etype or next_is_start:
                    break
                j += 1
            span_end = j

            span_len = span_end - span_start
            if span_len == 1:
                token_labels.append(LABEL2ID.get(f"S-{etype}", 0))
            else:
                for k in range(span_start, span_end):
                    if k == span_start:
                        token_labels.append(LABEL2ID.get(f"B-{etype}", 0))
                    elif k == span_end - 1:
                        token_labels.append(LABEL2ID.get(f"E-{etype}", 0))
                    else:
                        token_labels.append(LABEL2ID.get(f"I-{etype}", 0))

            i = span_end

        return token_labels

    def _encode_all(self) -> list[dict[str, torch.Tensor]]:
        """Pre-encode all samples."""
        encoded = []

        for sample in self.samples:
            text = sample["text"]
            entities = sample.get("entities", [])

            # Keep all entities including partial (right-boundary truncations
            # caused by noise injection offset drift — entities remain
            # semantically recognizable and provide valid training signal)
            clean_entities = entities

            char_labels = self._entities_to_char_labels(text, clean_entities)

            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_offsets_mapping=True,
                return_tensors="pt",
            )

            # Convert offset_mapping to list of tuples for iteration
            offset_list = encoding.offset_mapping[0].tolist()
            token_labels = self._align_labels_with_tokens(offset_list, char_labels)

            encoded.append({
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": torch.tensor(token_labels, dtype=torch.long),
                "offset_mapping": encoding["offset_mapping"].squeeze(0),
            })

        return encoded

    def __len__(self) -> int:
        return len(self._encoded)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self._encoded[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["labels"],
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    train_samples: list[dict[str, Any]],
    val_samples: list[dict[str, Any]],
    model_key: str = "bert_tiny",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a token classification model on NER data.

    Args:
        train_samples: Training NER samples.
        val_samples: Validation NER samples.
        model_key: Key in config["training"] (bert_tiny or bert_small).
        config: Optional config override.

    Returns:
        Training results dict with loss history and best model path.
    """
    if config is None:
        config = load_config()

    torch.manual_seed(get_seed())
    np.random.seed(get_seed())

    train_config = config["training"][model_key]
    model_name = train_config["model_name"]
    save_dir = Path(train_config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    log.info("Starting training", model=model_name, model_key=model_key)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(BIOES_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    log.info(
        "Model loaded",
        params=f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M",
    )

    # Create datasets
    log.info("Encoding training data", samples=len(train_samples))
    train_dataset = NERDataset(train_samples, tokenizer, train_config["max_seq_length"])
    log.info("Encoding validation data", samples=len(val_samples))
    val_dataset = NERDataset(val_samples, tokenizer, train_config["max_seq_length"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
    )

    # Optimizer with weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": train_config["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=train_config["learning_rate"])

    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * train_config["num_epochs"]
    warmup_steps = int(total_steps * train_config["warmup_ratio"])

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return max(0.0, (total_steps - step) / max(total_steps - warmup_steps, 1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    history = {"train_loss": [], "val_loss": [], "epoch_times": []}
    best_val_loss = float("inf")

    for epoch in range(train_config["num_epochs"]):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss_sum += loss.item()
            train_steps += 1

        avg_train_loss = train_loss_sum / max(train_steps, 1)

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                val_loss_sum += outputs.loss.item()
                val_steps += 1

        avg_val_loss = val_loss_sum / max(val_steps, 1)
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["epoch_times"].append(epoch_time)

        log.info(
            "Epoch complete",
            epoch=epoch + 1,
            train_loss=f"{avg_train_loss:.4f}",
            val_loss=f"{avg_val_loss:.4f}",
            time=f"{epoch_time:.1f}s",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            log.info("Best model saved", val_loss=f"{best_val_loss:.4f}")

    # Save training history
    history["best_val_loss"] = best_val_loss
    history["model_key"] = model_key
    history["model_name"] = model_name
    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    log.info(
        "Training complete",
        model_key=model_key,
        best_val_loss=f"{best_val_loss:.4f}",
        total_time=f"{sum(history['epoch_times']):.0f}s",
        save_dir=str(save_dir),
    )

    return history


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class TransformerNER:
    """Inference wrapper for a trained token classification model."""

    def __init__(self, model_dir: str | Path, max_length: int = 256) -> None:
        """Load a trained model from disk.

        Args:
            model_dir: Path to saved model directory.
        """
        self.model_dir = Path(model_dir)
        log.info("Loading TransformerNER", path=str(self.model_dir))

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, do_lower_case=True)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_dir)
        self.model.eval()
        self.max_length = max_length

        log.info(
            "TransformerNER loaded",
            params=f"{sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M",
        )

    def predict(self, text: str) -> list[dict[str, Any]]:
        """Extract entities from text using BIOES decoding.

        Tag interpretation:
        - S-X: single-token entity, emit immediately
        - B-X: start of multi-token entity
        - I-X: inside multi-token entity (continues B/I of same type)
        - E-X: end of multi-token entity, close and emit
        - O: outside, close any open entity

        Fallback handling for invalid sequences:
        - I-X without preceding B/I of same type → treat as B-X (new entity)
        - E-X without preceding B/I of same type → treat as S-X (single token)
        """
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        offset_mapping = encoding.pop("offset_mapping")[0].tolist()

        with torch.no_grad():
            outputs = self.model(**encoding)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            predictions = torch.argmax(probs, dim=-1)
            confidences = probs.max(dim=-1).values

        entities: list[dict[str, Any]] = []
        current_entity: dict[str, Any] | None = None

        for pred_id, conf, (start, end) in zip(predictions, confidences, offset_mapping):
            if start == 0 and end == 0:
                continue

            label = ID2LABEL[pred_id.item()]
            confidence = conf.item()

            if label == "O":
                if current_entity is not None:
                    self._finalize_entity(current_entity)
                    entities.append(current_entity)
                    current_entity = None
                continue

            prefix = label[:2]
            entity_type = label[2:]

            if prefix == "S-":
                if current_entity is not None:
                    self._finalize_entity(current_entity)
                    entities.append(current_entity)
                    current_entity = None
                entities.append({
                    "text": text[start:end],
                    "label": entity_type,
                    "start": start,
                    "end": end,
                    "confidence": round(confidence, 4),
                })

            elif prefix == "B-":
                if current_entity is not None:
                    self._finalize_entity(current_entity)
                    entities.append(current_entity)
                current_entity = {
                    "text": text[start:end],
                    "label": entity_type,
                    "start": start,
                    "end": end,
                    "_conf_sum": confidence,
                    "_tok_count": 1,
                }

            elif prefix == "I-":
                if (current_entity is not None
                        and current_entity["label"] == entity_type):
                    current_entity["end"] = end
                    current_entity["text"] = text[current_entity["start"]:end]
                    current_entity["_conf_sum"] += confidence
                    current_entity["_tok_count"] += 1
                else:
                    if current_entity is not None:
                        self._finalize_entity(current_entity)
                        entities.append(current_entity)
                    current_entity = {
                        "text": text[start:end],
                        "label": entity_type,
                        "start": start,
                        "end": end,
                        "_conf_sum": confidence,
                        "_tok_count": 1,
                    }

            elif prefix == "E-":
                if (current_entity is not None
                        and current_entity["label"] == entity_type):
                    current_entity["end"] = end
                    current_entity["text"] = text[current_entity["start"]:end]
                    current_entity["_conf_sum"] += confidence
                    current_entity["_tok_count"] += 1
                    self._finalize_entity(current_entity)
                    entities.append(current_entity)
                    current_entity = None
                else:
                    if current_entity is not None:
                        self._finalize_entity(current_entity)
                        entities.append(current_entity)
                        current_entity = None
                    entities.append({
                        "text": text[start:end],
                        "label": entity_type,
                        "start": start,
                        "end": end,
                        "confidence": round(confidence, 4),
                    })

        if current_entity is not None:
            self._finalize_entity(current_entity)
            entities.append(current_entity)

        return entities

    @staticmethod
    def _finalize_entity(entity: dict[str, Any]) -> None:
        """Compute final confidence and remove internal tracking fields."""
        if "_conf_sum" in entity and "_tok_count" in entity:
            entity["confidence"] = round(
                entity["_conf_sum"] / entity["_tok_count"], 4
            )
            del entity["_conf_sum"]
            del entity["_tok_count"]