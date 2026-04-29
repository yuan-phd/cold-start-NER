"""Unit tests for ASR noise engine."""

import random

from src.data.noise import inject_noise, OffsetTracker, create_noisy_dataset


def _make_sample(text, entities):
    return {"text": text, "entities": entities}


class TestOffsetTracker:
    def test_no_shifts(self):
        tracker = OffsetTracker()
        assert tracker.apply(10) == 10

    def test_insertion_before(self):
        tracker = OffsetTracker()
        tracker.record(5, +3)
        assert tracker.apply(10) == 13
        assert tracker.apply(3) == 3

    def test_deletion_before(self):
        tracker = OffsetTracker()
        tracker.record(5, -2)
        assert tracker.apply(10) == 8
        assert tracker.apply(3) == 3

    def test_multiple_shifts(self):
        tracker = OffsetTracker()
        tracker.record(5, +2)
        tracker.record(10, -1)
        assert tracker.apply(15) == 16


class TestInjectNoise:
    def test_clean_level_unchanged(self):
        sample = _make_sample(
            "My name is John Smith and my email is john@test.com",
            [
                {"text": "John Smith", "label": "NAME", "start": 11, "end": 21},
                {"text": "john@test.com", "label": "EMAIL", "start": 38, "end": 51},
            ],
        )
        result = inject_noise(sample, noise_level="clean")
        assert result["text"] == sample["text"]
        assert result["noise_level"] == "clean"

    def test_noisy_preserves_entity_count(self):
        sample = _make_sample(
            "My name is John Smith and my email is john@test.com",
            [
                {"text": "John Smith", "label": "NAME", "start": 11, "end": 21},
                {"text": "john@test.com", "label": "EMAIL", "start": 38, "end": 51},
            ],
        )
        rng = random.Random(42)
        result = inject_noise(sample, noise_level="mild", rng=rng)
        assert len(result["entities"]) == 2
        assert result["noise_level"] == "mild"

    def test_noise_level_in_output(self):
        sample = _make_sample("Hello world", [])
        for level in ["clean", "mild", "moderate", "severe"]:
            result = inject_noise(sample, noise_level=level)
            assert result["noise_level"] == level


class TestCreateNoisyDataset:
    def test_multiplies_samples(self):
        samples = [
            _make_sample("Test sentence one.", []),
            _make_sample("Test sentence two.", []),
        ]
        result = create_noisy_dataset(samples)
        assert len(result) == 8

    def test_all_levels_present(self):
        samples = [_make_sample("Test sentence.", [])]
        result = create_noisy_dataset(samples)
        levels = {s["noise_level"] for s in result}
        assert levels == {"clean", "mild", "moderate", "severe"}