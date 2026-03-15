"""Tests for hearts.ml.train — training pipeline."""

import json
from pathlib import Path

import numpy as np
import pytest

from hearts.bots.rule_bot import RuleBot
from hearts.data.generator import generate_game_data
from hearts.ml.features import BasicFeatureExtractor
from hearts.ml.train import (
    build_dataset,
    load_data,
    load_model,
    save_model,
    train_model,
)


@pytest.fixture
def small_dataset(tmp_path):
    """Generate a small dataset for testing (5 games)."""
    play_file = tmp_path / "play_decisions.jsonl"
    all_play_records = []
    for seed in range(5):
        bots = [RuleBot() for _ in range(4)]
        _, play_recs, _ = generate_game_data(bots, seed=seed)
        all_play_records.extend(play_recs)
    with open(play_file, "w") as f:
        for rec in all_play_records:
            f.write(json.dumps(rec) + "\n")
    return tmp_path, all_play_records


class TestLoadData:
    """load_data reads JSONL files correctly."""

    def test_loads_all_records(self, small_dataset):
        data_dir, expected_records = small_dataset
        records = load_data(data_dir)
        assert len(records) == len(expected_records)

    def test_records_are_dicts(self, small_dataset):
        data_dir, _ = small_dataset
        records = load_data(data_dir)
        assert all(isinstance(r, dict) for r in records)

    def test_records_have_card_played(self, small_dataset):
        data_dir, _ = small_dataset
        records = load_data(data_dir)
        assert all("card_played" in r for r in records)


class TestBuildDataset:
    """build_dataset converts records to X, y arrays."""

    def test_shapes(self, small_dataset):
        data_dir, _ = small_dataset
        records = load_data(data_dir)
        extractor = BasicFeatureExtractor()
        X, y = build_dataset(records, extractor)
        assert X.shape[0] == len(records)
        assert X.shape[1] == len(extractor.feature_names())
        assert y.shape == (len(records),)

    def test_labels_in_range(self, small_dataset):
        data_dir, _ = small_dataset
        records = load_data(data_dir)
        extractor = BasicFeatureExtractor()
        _, y = build_dataset(records, extractor)
        assert np.all(y >= 0)
        assert np.all(y <= 51)

    def test_no_nan_features(self, small_dataset):
        data_dir, _ = small_dataset
        records = load_data(data_dir)
        extractor = BasicFeatureExtractor()
        X, _ = build_dataset(records, extractor)
        assert not np.any(np.isnan(X))
        assert not np.any(np.isinf(X))


class TestTrainModel:
    """train_model trains and returns metrics."""

    def test_returns_model_and_metrics(self, small_dataset):
        data_dir, _ = small_dataset
        records = load_data(data_dir)
        extractor = BasicFeatureExtractor()
        X, y = build_dataset(records, extractor)
        model, metrics = train_model(
            X, y, extractor.feature_names(), seed=42
        )
        assert model is not None
        assert isinstance(metrics, dict)

    def test_metrics_keys(self, small_dataset):
        data_dir, _ = small_dataset
        records = load_data(data_dir)
        extractor = BasicFeatureExtractor()
        X, y = build_dataset(records, extractor)
        _, metrics = train_model(
            X, y, extractor.feature_names(), seed=42
        )
        expected_keys = {
            "train_accuracy",
            "test_accuracy",
            "feature_importances",
            "n_train",
            "n_test",
            "n_features",
        }
        assert expected_keys.issubset(metrics.keys())

    def test_accuracies_in_range(self, small_dataset):
        data_dir, _ = small_dataset
        records = load_data(data_dir)
        extractor = BasicFeatureExtractor()
        X, y = build_dataset(records, extractor)
        _, metrics = train_model(
            X, y, extractor.feature_names(), seed=42
        )
        assert 0.0 <= metrics["train_accuracy"] <= 1.0
        assert 0.0 <= metrics["test_accuracy"] <= 1.0

    def test_feature_importances_sorted(self, small_dataset):
        data_dir, _ = small_dataset
        records = load_data(data_dir)
        extractor = BasicFeatureExtractor()
        X, y = build_dataset(records, extractor)
        _, metrics = train_model(
            X, y, extractor.feature_names(), seed=42
        )
        importances = metrics["feature_importances"]
        values = [v for _, v in importances]
        assert values == sorted(values, reverse=True)

    def test_feature_importances_count(self, small_dataset):
        data_dir, _ = small_dataset
        records = load_data(data_dir)
        extractor = BasicFeatureExtractor()
        X, y = build_dataset(records, extractor)
        _, metrics = train_model(
            X, y, extractor.feature_names(), seed=42
        )
        assert len(metrics["feature_importances"]) == X.shape[1]

    def test_predictions_valid_card_indices(self, small_dataset):
        data_dir, _ = small_dataset
        records = load_data(data_dir)
        extractor = BasicFeatureExtractor()
        X, y = build_dataset(records, extractor)
        model, _ = train_model(
            X, y, extractor.feature_names(), seed=42
        )
        preds = model.predict(X[:10])
        assert all(0 <= p <= 51 for p in preds)

    def test_deterministic_with_seed(self, small_dataset):
        data_dir, _ = small_dataset
        records = load_data(data_dir)
        extractor = BasicFeatureExtractor()
        X, y = build_dataset(records, extractor)
        _, m1 = train_model(
            X, y, extractor.feature_names(), seed=42
        )
        _, m2 = train_model(
            X, y, extractor.feature_names(), seed=42
        )
        assert m1["train_accuracy"] == m2["train_accuracy"]
        assert m1["test_accuracy"] == m2["test_accuracy"]


class TestSaveLoadModel:
    """save_model and load_model round-trip correctly."""

    def test_round_trip(self, small_dataset, tmp_path):
        data_dir, _ = small_dataset
        records = load_data(data_dir)
        extractor = BasicFeatureExtractor()
        X, y = build_dataset(records, extractor)
        model, metrics = train_model(
            X, y, extractor.feature_names(), seed=42
        )

        model_path = tmp_path / "test_model.pkl"
        save_model(model, "BasicFeatureExtractor", metrics, model_path)
        assert model_path.exists()

        loaded_model, loaded_name, loaded_metrics = load_model(
            model_path
        )
        assert loaded_name == "BasicFeatureExtractor"
        assert (
            loaded_metrics["train_accuracy"]
            == metrics["train_accuracy"]
        )
        assert (
            loaded_metrics["test_accuracy"]
            == metrics["test_accuracy"]
        )

        # Model should produce same predictions
        preds_orig = model.predict(X[:5])
        preds_loaded = loaded_model.predict(X[:5])
        np.testing.assert_array_equal(preds_orig, preds_loaded)

    def test_creates_parent_dirs(self, tmp_path):
        """save_model should create parent directories if needed."""
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=1)
        model.fit([[0, 1], [1, 0]], [0, 1])
        path = tmp_path / "nested" / "dir" / "model.pkl"
        save_model(model, "Test", {}, path)
        assert path.exists()
