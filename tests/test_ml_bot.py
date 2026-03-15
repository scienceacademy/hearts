"""Tests for hearts.bots.ml_bot — ML-powered bot."""

import json
import random
from pathlib import Path

import numpy as np
import pytest

from hearts.bot import Bot
from hearts.bots.ml_bot import MLBot, load_ml_bot
from hearts.bots.rule_bot import RuleBot
from hearts.data.generator import generate_game_data
from hearts.game import Game
from hearts.ml.features import BasicFeatureExtractor
from hearts.ml.train import (
    build_dataset,
    load_data,
    save_model,
    train_model,
)
from hearts.state import PassView, PlayerView


@pytest.fixture
def trained_model(tmp_path):
    """Train a small model and save it for testing."""
    # Generate data
    play_file = tmp_path / "play_decisions.jsonl"
    all_records = []
    for seed in range(5):
        bots = [RuleBot() for _ in range(4)]
        _, play_recs, _ = generate_game_data(bots, seed=seed)
        all_records.extend(play_recs)
    with open(play_file, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    # Train
    records = load_data(tmp_path)
    extractor = BasicFeatureExtractor()
    X, y = build_dataset(records, extractor)
    model, metrics = train_model(
        X, y, extractor.feature_names(), seed=42
    )

    # Save
    model_path = tmp_path / "model.pkl"
    save_model(model, "BasicFeatureExtractor", metrics, model_path)
    return model, extractor, model_path


class TestMLBot:
    """MLBot plays legal cards using a trained model."""

    def test_always_plays_legal(self, trained_model):
        """Fuzz test: MLBot should always return a legal card."""
        model, extractor, _ = trained_model
        ml_bot = MLBot(model, extractor)
        bots = [ml_bot, RuleBot(), RuleBot(), RuleBot()]

        for seed in range(10):
            rng = random.Random(seed)
            game = Game(bots, rng=rng)
            # Should complete without ValueError from illegal plays
            game.play()

    def test_completes_full_game(self, trained_model):
        model, extractor, _ = trained_model
        ml_bot = MLBot(model, extractor)
        bots = [ml_bot, RuleBot(), RuleBot(), RuleBot()]
        result = Game(bots, rng=random.Random(42)).play()
        assert result.num_rounds > 0
        assert len(result.final_scores) == 4

    def test_delegates_passing_to_strategy(self, trained_model):
        """MLBot should use the pass_strategy for passing."""
        model, extractor, _ = trained_model

        class TrackingBot(RuleBot):
            def __init__(self):
                super().__init__()
                self.pass_called = False

            def pass_cards(self, view):
                self.pass_called = True
                return super().pass_cards(view)

        tracker = TrackingBot()
        ml_bot = MLBot(model, extractor, pass_strategy=tracker)
        bots = [ml_bot, RuleBot(), RuleBot(), RuleBot()]
        Game(bots, rng=random.Random(42)).play()
        assert tracker.pass_called

    def test_uses_predict_proba(self, trained_model):
        """MLBot should prefer predict_proba when available."""
        model, extractor, _ = trained_model
        assert hasattr(model, "predict_proba")
        ml_bot = MLBot(model, extractor)
        bots = [ml_bot, RuleBot(), RuleBot(), RuleBot()]
        # Just verify it runs — predict_proba path is exercised
        Game(bots, rng=random.Random(99)).play()


class TestMLBotFallback:
    """MLBot handles edge cases gracefully."""

    def test_fallback_to_random_on_bad_model(self):
        """If model predicts only illegal cards, fall back to random."""

        class BadModel:
            """A model that always predicts card index 0."""
            classes_ = np.array([0])

            def predict_proba(self, X):
                return np.array([[1.0]])

            def predict(self, X):
                return np.array([0])

        extractor = BasicFeatureExtractor()
        ml_bot = MLBot(
            BadModel(), extractor, rng=random.Random(42)
        )
        bots = [ml_bot, RuleBot(), RuleBot(), RuleBot()]
        # Should complete without error even with a terrible model
        result = Game(bots, rng=random.Random(42)).play()
        assert result.num_rounds > 0


class TestLoadMLBot:
    """load_ml_bot loads a saved model and returns a working bot."""

    def test_loads_and_plays(self, trained_model):
        _, _, model_path = trained_model
        ml_bot = load_ml_bot(model_path)
        assert isinstance(ml_bot, MLBot)
        bots = [ml_bot, RuleBot(), RuleBot(), RuleBot()]
        result = Game(bots, rng=random.Random(42)).play()
        assert result.num_rounds > 0

    def test_custom_pass_strategy(self, trained_model):
        _, _, model_path = trained_model

        class TrackingBot(RuleBot):
            def __init__(self):
                super().__init__()
                self.pass_called = False

            def pass_cards(self, view):
                self.pass_called = True
                return super().pass_cards(view)

        tracker = TrackingBot()
        ml_bot = load_ml_bot(model_path, pass_strategy=tracker)
        bots = [ml_bot, RuleBot(), RuleBot(), RuleBot()]
        Game(bots, rng=random.Random(42)).play()
        assert tracker.pass_called

    def test_unknown_extractor_raises(self, tmp_path):
        """Should raise if the saved extractor name is unknown."""
        import pickle

        model_path = tmp_path / "bad_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "model": None,
                    "extractor_class_name": "NonexistentExtractor",
                    "metrics": {},
                },
                f,
            )
        with pytest.raises(ValueError, match="Unknown extractor"):
            load_ml_bot(model_path)
