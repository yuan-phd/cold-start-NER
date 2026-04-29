"""Pydantic request/response schemas for the NER API."""

from pydantic import BaseModel


class Entity(BaseModel):
    """A single extracted entity."""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    partial: bool = False


class NERRequest(BaseModel):
    """Input to the NER endpoint."""
    text: str
    model: str = "ensemble"


class NERResponse(BaseModel):
    """Output from the NER endpoint."""
    entities: list[Entity]
    model_used: str
    inference_time_ms: float
    routing_stats: dict[str, int] | None = None
