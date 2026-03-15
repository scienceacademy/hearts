"""Tests for hearts.ml.evaluate — model evaluation pipeline."""

import json
import random

import pytest

from hearts.bots.rule_bot import RuleBot
from hearts.data.generator import generate_game_data
from hearts.ml.evaluate import (
    EvaluationResult,
    print_report,
    run_evaluation,
)
from hearts.ml.features import BasicFeatureExtractor
from hearts.ml.train import (
    build_dataset,
    load_data,
    save_model,
    train_model,
)


@pytest.fixture
def saved_model(tmp_path):
    """Generate data, train, and save a model for evaluation tests."""
    play_file = tmp_path / "play_decisions.jsonl"
    all_records = []
    for seed in range(5):
        bots = [RuleBot() for _ in range(4)]
        _, play_recs, _ = generate_game_data(bots, seed=seed)
        all_records.extend(play_recs)
    with open(play_file, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    records = load_data(tmp_path)
    extractor = BasicFeatureExtractor()
    X, y = build_dataset(records, extractor)
    model, metrics = train_model(
        X, y, extractor.feature_names(), seed=42
    )

    model_path = tmp_path / "model.pkl"
    save_model(model, "BasicFeatureExtractor", metrics, model_path)
    return model_path


class TestRunEvaluation:
    """run_evaluation returns a complete EvaluationResult."""

    def test_returns_evaluation_result(self, saved_model):
        result = run_evaluation(
            saved_model, n_games=10, seed=42
        )
        assert isinstance(result, EvaluationResult)

    def test_training_metrics_populated(self, saved_model):
        result = run_evaluation(
            saved_model, n_games=10, seed=42
        )
        assert result.extractor_name == "BasicFeatureExtractor"
        assert result.n_features > 0
        assert 0.0 <= result.train_accuracy <= 1.0
        assert 0.0 <= result.test_accuracy <= 1.0

    def test_feature_importances_present(self, saved_model):
        result = run_evaluation(
            saved_model, n_games=10, seed=42
        )
        assert len(result.feature_importances) > 0
        # Each entry is (name, importance)
        name, imp = result.feature_importances[0]
        assert isinstance(name, str)
        assert isinstance(imp, float)

    def test_tournament_has_all_bots(self, saved_model):
        result = run_evaluation(
            saved_model, n_games=10, seed=42
        )
        stats = result.tournament.stats
        assert "MLBot" in stats
        assert "RuleBot-1" in stats
        assert "RuleBot-2" in stats
        assert "RuleBot-3" in stats

    def test_win_rates_sum_approximately(self, saved_model):
        result = run_evaluation(
            saved_model, n_games=20, seed=42
        )
        stats = result.tournament.stats
        total_wins = sum(s.wins for s in stats.values())
        assert total_wins == 20

    def test_tournament_uses_seat_rotation(self, saved_model):
        result = run_evaluation(
            saved_model, n_games=12, seed=42
        )
        stats = result.tournament.stats
        # With rotation, each bot should play multiple games
        for bot_stats in stats.values():
            assert bot_stats.games > 0

    def test_n_games_matches(self, saved_model):
        result = run_evaluation(
            saved_model, n_games=15, seed=42
        )
        assert result.n_games == 15
        assert result.tournament.num_games == 15


class TestPrintReport:
    """print_report produces output without errors."""

    def test_prints_without_error(self, saved_model, capsys):
        result = run_evaluation(
            saved_model, n_games=10, seed=42
        )
        print_report(result)
        captured = capsys.readouterr()
        assert "Training Metrics" in captured.out
        assert "Feature Importance" in captured.out
        assert "Tournament Results" in captured.out
        assert "MLBot" in captured.out

    def test_contains_baseline_targets(self, saved_model, capsys):
        result = run_evaluation(
            saved_model, n_games=10, seed=42
        )
        print_report(result)
        captured = capsys.readouterr()
        assert "Baseline" in captured.out
        assert "Target" in captured.out
