"""Unit tests for NER API endpoint."""

from fastapi.testclient import TestClient

from src.api.serve import app

client = TestClient(app)


class TestHealthCheck:
    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestExtractEndpoint:
    def test_rules_model(self):
        response = client.post(
            "/extract",
            json={
                "text": "My name is John Smith and my email is john@test.com",
                "model": "rules",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "entities" in data
        assert data["model_used"] == "rules"
        assert "inference_time_ms" in data

    def test_rules_finds_email(self):
        response = client.post(
            "/extract",
            json={"text": "Contact me at alice@example.com", "model": "rules"},
        )
        data = response.json()
        labels = [e["label"] for e in data["entities"]]
        assert "EMAIL" in labels

    def test_invalid_model(self):
        response = client.post(
            "/extract",
            json={"text": "Hello", "model": "nonexistent"},
        )
        assert response.status_code == 400