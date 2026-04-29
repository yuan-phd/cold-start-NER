"""GLiNER zero-shot NER baseline — no training data required.

Uses a pre-trained GLiNER model to extract entities without any fine-tuning.
This serves as the true cold-start baseline: what can you achieve with zero
labeled data? GLiNER uses a bidirectional transformer encoder with learned
entity type representations for span-based zero-shot NER.
"""

from typing import Any

from gliner import GLiNER

from src.utils.config import load_config
from src.utils.logger import get_logger

log = get_logger(__name__)

_model: GLiNER | None = None


def _get_model(config: dict[str, Any] | None = None) -> GLiNER:
    """Load and cache the GLiNER model (singleton pattern).

    Args:
        config: Optional config override.

    Returns:
        Loaded GLiNER model.
    """
    global _model
    if _model is not None:
        return _model

    if config is None:
        config = load_config()

    gliner_config = config["gliner"]
    log.info("Loading GLiNER model", model=gliner_config["model_name"])
    _model = GLiNER.from_pretrained(gliner_config["model_name"])
    log.info("GLiNER model loaded")
    return _model


def predict(
    text: str,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Extract entities from text using GLiNER zero-shot NER.

    Args:
        text: Input text string.
        config: Optional config override.

    Returns:
        List of entity dicts with text, label, start, end, confidence.
    """
    if config is None:
        config = load_config()

    gliner_config = config["gliner"]
    model = _get_model(config)

    # GLiNER prediction
    raw_entities = model.predict_entities(
        text,
        gliner_config["labels"],
        threshold=gliner_config["threshold"],
    )

    # Map GLiNER labels to our entity types
    label_map = gliner_config["label_map"]
    entities = []

    for ent in raw_entities:
        mapped_label = label_map.get(ent["label"])
        if mapped_label is None:
            log.debug("Unmapped GLiNER label", label=ent["label"])
            continue

        entities.append({
            "text": ent["text"],
            "label": mapped_label,
            "start": ent["start"],
            "end": ent["end"],
            "confidence": round(ent["score"], 4),
        })

    return entities


def predict_batch(
    texts: list[str],
    config: dict[str, Any] | None = None,
) -> list[list[dict[str, Any]]]:
    """Extract entities from a batch of texts.

    Args:
        texts: List of input text strings.
        config: Optional config override.

    Returns:
        List of entity lists, one per input text.
    """
    return [predict(text, config) for text in texts]