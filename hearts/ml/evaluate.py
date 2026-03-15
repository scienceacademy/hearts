"""Evaluation of trained Hearts ML models.

Loads a trained model, runs a tournament against RuleBot, and
produces a unified report combining training metrics, feature
importance, and tournament results.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from hearts.bots.ml_bot import MLBot, load_ml_bot
from hearts.bots.rule_bot import RuleBot
from hearts.ml.train import load_model
from hearts.tournament import Tournament, TournamentResult


class _MLBotFactory:
    """Factory for MLBot instances that caches the loaded model.

    Loads the model from disk once on first call, then reuses
    the same model object for all subsequent bot instances.
    """

    def __init__(
        self, model_path: str | Path, seed: int
    ) -> None:
        self._model_path = model_path
        self._seed = seed
        self._counter = 0
        self._cached_bot: Optional[MLBot] = None

    def __call__(self) -> MLBot:
        # Load model once, then create new MLBot instances
        # sharing the same model and extractor
        if self._cached_bot is None:
            bot_rng = random.Random(self._seed + self._counter)
            self._counter += 1
            self._cached_bot = load_ml_bot(
                self._model_path, rng=bot_rng
            )
            return self._cached_bot

        bot_rng = random.Random(self._seed + self._counter)
        self._counter += 1
        return MLBot(
            model=self._cached_bot._model,
            extractor=self._cached_bot._extractor,
            rng=bot_rng,
        )


@dataclass
class EvaluationResult:
    """Complete evaluation output: training metrics + tournament."""

    # Training metrics (from saved model)
    extractor_name: str
    n_features: int
    train_accuracy: float
    test_accuracy: float
    feature_importances: list[tuple[str, float]]

    # Tournament results
    tournament: TournamentResult
    n_games: int


def run_evaluation(
    model_path: str | Path,
    n_games: int = 200,
    seed: Optional[int] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    parallel: bool = False,
) -> EvaluationResult:
    """Load a model, run a tournament, and return full results.

    Runs 1x MLBot vs 3x RuleBot with seat rotation.

    Args:
        model_path: path to saved model pickle
        n_games: number of tournament games
        seed: random seed for reproducibility
        progress: optional callback(current, total) for progress
        parallel: if True, run tournament games in parallel

    Returns:
        EvaluationResult with training metrics and tournament stats.
    """
    _, extractor_name, metrics = load_model(model_path)

    rng = random.Random(seed)

    bot_factories = {
        "MLBot": _MLBotFactory(
            model_path, seed=rng.randint(0, 2**63)
        ),
        "RuleBot-1": RuleBot,
        "RuleBot-2": RuleBot,
        "RuleBot-3": RuleBot,
    }

    tournament = Tournament(
        bot_factories=bot_factories,
        num_games=n_games,
        rng=random.Random(rng.randint(0, 2**63)),
    )
    tournament_result = tournament.run(
        progress=progress, parallel=parallel
    )

    return EvaluationResult(
        extractor_name=extractor_name,
        n_features=metrics.get("n_features", 0),
        train_accuracy=metrics.get("train_accuracy", 0.0),
        test_accuracy=metrics.get("test_accuracy", 0.0),
        feature_importances=metrics.get(
            "feature_importances", []
        ),
        tournament=tournament_result,
        n_games=n_games,
    )


def print_report(result: EvaluationResult) -> None:
    """Print a formatted evaluation report to stdout."""
    print("=== Training Metrics ===")
    print(f"Feature extractor:    {result.extractor_name}")
    print(f"Feature vector size:  {result.n_features} features")
    print(f"Train accuracy:       {result.train_accuracy:.1%}")
    print(f"Test accuracy:        {result.test_accuracy:.1%}")
    print()

    print("=== Feature Importance (top 10) ===")
    for i, (name, importance) in enumerate(
        result.feature_importances[:10]
    ):
        print(f"  {i + 1:>2}. {name:<24} {importance:.4f}")
    print()

    print(f"=== Tournament Results ({result.n_games} games) ===")
    print(
        f"{'Player':<16} {'Win Rate':>10} "
        f"{'Avg Score':>10} {'Avg Rank':>10}"
    )

    stats = result.tournament.stats
    # Compute average rank from scores
    # (lower score = better rank within each game)
    for name, bot_stats in sorted(
        stats.items(),
        key=lambda x: x[1].avg_score,
    ):
        print(
            f"{name:<16} {bot_stats.win_rate:>9.1%} "
            f"{bot_stats.avg_score:>10.1f}"
            f"{'':>10}"
        )

    print()
    print("Baseline (RandomBot):  ~1% win rate, ~104 avg score")
    print("Target: >15% win rate with good features")
