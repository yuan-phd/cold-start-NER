"""Tests for V3 API features: partial entity detection field and routing stats.

Extends the existing test_api.py with V3-specific endpoint behavior.
"""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from src.api.serve import app

client = TestClient(app)

MODEL_DIR = Path("results/models/bert_small")


@pytest.mark.skipif(
    not MODEL_DIR.exists(),
    reason="Trained BERT-Small model not found — run 'make train' first",
)
class TestEnsembleV3API:
    """Test V3 ensemble API response includes new fields."""

    def test_ensemble_returns_partial_field(self):
        response = client.post(
            "/extract",
            json={"text": "order ORD-2024-5591 thanks", "model": "ensemble"},
        )
        assert response.status_code == 200
        data = response.json()
        for entity in data["entities"]:
            assert "partial" in entity
            assert isinstance(entity["partial"], bool)

    def test_ensemble_returns_routing_stats(self):
        response = client.post(
            "/extract",
            json={"text": "order ORD-2024-5591 thanks", "model": "ensemble"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "routing_stats" in data
        assert data["routing_stats"] is not None
        stats = data["routing_stats"]
        assert "conflicts" in stats
        assert "model_fallback" in stats
        assert "rules_fallback" in stats
        assert "correction_removals" in stats
        assert "total_entities" in stats

    def test_ensemble_routing_stats_types(self):
        response = client.post(
            "/extract",
            json={"text": "hi my name is john", "model": "ensemble"},
        )
        data = response.json()
        if data["routing_stats"]:
            for key, value in data["routing_stats"].items():
                assert isinstance(value, int), f"{key} should be int, got {type(value)}"

    def test_non_ensemble_model_no_routing_stats(self):
        response = client.post(
            "/extract",
            json={"text": "order ORD-2024-5591", "model": "rules"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["routing_stats"] is None

    def test_complete_contract_id_not_partial(self):
        response = client.post(
            "/extract",
            json={"text": "order ORD-2024-5591 thanks", "model": "ensemble"},
        )
        data = response.json()
        cids = [e for e in data["entities"] if e["label"] == "CONTRACT_ID"]
        assert len(cids) >= 1
        assert cids[0]["partial"] is False

    def test_oral_contract_id_detected(self):
        response = client.post(
            "/extract",
            json={
                "text": "order ORD dash 2025 dash 0091 thanks",
                "model": "ensemble",
            },
        )
        data = response.json()
        cids = [e for e in data["entities"] if e["label"] == "CONTRACT_ID"]
        assert len(cids) >= 1
