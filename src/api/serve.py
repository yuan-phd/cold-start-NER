"""FastAPI NER service — expose entity extraction as a REST endpoint."""

import time
from typing import Any

from fastapi import FastAPI, HTTPException

from src.api.schemas import NERRequest, NERResponse, Entity
from src.models import rules
from src.utils.config import load_config
from src.utils.logger import setup_logging, get_logger

setup_logging()
log = get_logger(__name__)

app = FastAPI(
    title="Diabolocom NER API",
    description="Lightweight NER for noisy customer service transcriptions",
    version="0.1.0",
)

# Model cache — loaded on first request
_models: dict[str, Any] = {}


def _get_model(model_name: str) -> Any:
    """Load and cache a model by name.

    Args:
        model_name: One of "rules", "gliner", "bert-tiny", "bert-small", "ensemble".

    Returns:
        Model object with a predict() method.
    """
    if model_name in _models:
        return _models[model_name]

    config = load_config()

    if model_name == "rules":
        _models[model_name] = rules
        return rules

    elif model_name == "gliner":
        from src.models import gliner_ner
        _get_model.__doc__  # Force module load
        _models[model_name] = gliner_ner
        return gliner_ner

    elif model_name == "bert-tiny":
        from src.models.transformer_ner import TransformerNER
        model_dir = config["training"]["bert_tiny"]["save_dir"]
        model = TransformerNER(model_dir)
        _models[model_name] = model
        return model

    elif model_name == "bert-small":
        from src.models.transformer_ner import TransformerNER
        model_dir = config["training"]["bert_small"]["save_dir"]
        model = TransformerNER(model_dir)
        _models[model_name] = model
        return model

    elif model_name == "ensemble":
        from src.models.transformer_ner import TransformerNER
        from src.models import ensemble
        model_dir = config["training"]["bert_small"]["save_dir"]
        transformer = TransformerNER(model_dir)
        _models["_ensemble_transformer"] = transformer
        # Return both entities and stats — caller unpacks
        _models[model_name] = lambda text: ensemble.predict(text, transformer)
        return _models[model_name]

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model_name}. Choose from: rules, gliner, bert-tiny, bert-small, ensemble",
        )


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/extract", response_model=NERResponse)
def extract_entities(request: NERRequest) -> NERResponse:
    """Extract named entities from text.

    Args:
        request: NER request with text and model selection.

    Returns:
        NER response with extracted entities and timing.
    """
    log.info("Extraction request", model=request.model, text_length=len(request.text))

    model = _get_model(request.model)

    start_time = time.perf_counter()

    routing_stats = None
    if request.model == "ensemble":
        raw_entities, routing_stats = model(request.text)
    elif hasattr(model, "predict"):
        raw_entities = model.predict(request.text)
    else:
        raise HTTPException(status_code=500, detail="Model has no predict method")

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    entities = [
        Entity(
            text=e["text"],
            label=e["label"],
            start=e["start"],
            end=e["end"],
            confidence=e.get("confidence", 0.0),
            partial=e.get("partial", False),
        )
        for e in raw_entities
    ]

    return NERResponse(
        entities=entities,
        model_used=request.model,
        inference_time_ms=round(elapsed_ms, 2),
        routing_stats=routing_stats,
    )