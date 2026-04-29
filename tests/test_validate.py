"""Unit tests for data quality validation."""

from src.data.validate import validate_sample, validate_dataset


class TestValidateSample:
    def test_valid_sample(self):
        sample = {
            "text": "My name is John Smith.",
            "entities": [
                {"text": "John Smith", "label": "NAME", "start": 11, "end": 21},
            ],
        }
        is_valid, issues = validate_sample(sample)
        assert is_valid
        assert len(issues) == 0

    def test_missing_text(self):
        sample = {"entities": []}
        is_valid, issues = validate_sample(sample)
        assert not is_valid

    def test_missing_entities(self):
        sample = {"text": "Hello world"}
        is_valid, issues = validate_sample(sample)
        assert not is_valid

    def test_span_mismatch(self):
        sample = {
            "text": "My name is John Smith.",
            "entities": [
                {"text": "Jane Smith", "label": "NAME", "start": 11, "end": 21},
            ],
        }
        is_valid, issues = validate_sample(sample)
        assert not is_valid
        assert any("span mismatch" in issue for issue in issues)

    def test_invalid_label(self):
        sample = {
            "text": "My name is John Smith.",
            "entities": [
                {"text": "John Smith", "label": "PERSON", "start": 11, "end": 21},
            ],
        }
        is_valid, issues = validate_sample(sample)
        assert not is_valid

    def test_empty_entities_valid(self):
        sample = {
            "text": "Hello, how can I help you today?",
            "entities": [],
        }
        is_valid, issues = validate_sample(sample)
        assert is_valid


class TestValidateDataset:
    def test_basic_stats(self):
        samples = [
            {
                "text": "My name is John Smith.",
                "entities": [
                    {"text": "John Smith", "label": "NAME", "start": 11, "end": 21},
                ],
            },
            {
                "text": "Email me at test@gmail.com please.",
                "entities": [
                    {"text": "test@gmail.com", "label": "EMAIL", "start": 12, "end": 26},
                ],
            },
        ]
        report = validate_dataset(samples)
        assert report["total_samples"] == 2
        assert report["valid_samples"] == 2
        assert report["total_entities"] == 2