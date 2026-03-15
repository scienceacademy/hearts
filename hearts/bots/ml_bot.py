"""ML-powered Hearts bot that uses a trained model to pick cards."""

from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Optional

import numpy as np

from hearts.bot import Bot
from hearts.bots.rule_bot import RuleBot
from hearts.data.schema import card_to_index, index_to_card
from hearts.ml.features import (
    BasicFeatureExtractor,
    FeatureExtractor,
    StudentFeatureExtractor,
)
from hearts.state import PassView, PlayerView, RoundResult, TrickResult
from hearts.types import Card

EXTRACTOR_REGISTRY: dict[str, type[FeatureExtractor]] = {
    "BasicFeatureExtractor": BasicFeatureExtractor,
    "StudentFeatureExtractor": StudentFeatureExtractor,
}


class MLBot(Bot):
    """A bot that uses a trained ML model to select cards.

    Uses the model's predict_proba() to rank cards, then masks to
    legal moves only and picks the highest-probability legal card.
    Falls back to predict() if predict_proba is unavailable.

    Pass strategy is handled by composition — delegates to a separate
    bot instance (defaults to RuleBot).

    Args:
        model: a trained scikit-learn classifier
        extractor: the feature extractor matching the model's training
        pass_strategy: bot to delegate pass_cards to (default: RuleBot)
        rng: optional RNG for tiebreaking fallbacks
    """

    def __init__(
        self,
        model: object,
        extractor: FeatureExtractor,
        pass_strategy: Optional[Bot] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self._model = model
        self._extractor = extractor
        self._pass_strategy = pass_strategy or RuleBot()
        self._rng = rng or random.Random()

    def pass_cards(self, view: PassView) -> list[Card]:
        """Delegate passing to the pass strategy bot."""
        return self._pass_strategy.pass_cards(view)

    def play_card(self, view: PlayerView) -> Card:
        """Select the best legal card using the trained model."""
        features = self._extractor.extract(view)
        X = features.reshape(1, -1)

        legal_indices = {
            card_to_index(c) for c in view.legal_plays
        }

        # Try predict_proba first for probability-based selection
        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(X)[0]
            # The model's classes_ may not cover all 52 card indices
            classes = self._model.classes_
            best_card_idx = None
            best_prob = -1.0
            for i, cls in enumerate(classes):
                if int(cls) in legal_indices and proba[i] > best_prob:
                    best_prob = proba[i]
                    best_card_idx = int(cls)
            if best_card_idx is not None:
                return index_to_card(best_card_idx)

        # Fallback: use predict() and check legality
        pred = self._model.predict(X)[0]
        if pred in legal_indices:
            return index_to_card(pred)

        # Last resort: random legal card
        return self._rng.choice(view.legal_plays)

    def on_trick_complete(self, result: TrickResult) -> None:
        """Forward to pass strategy bot."""
        self._pass_strategy.on_trick_complete(result)

    def on_round_complete(self, result: RoundResult) -> None:
        """Forward to pass strategy bot."""
        self._pass_strategy.on_round_complete(result)


def load_ml_bot(
    model_path: str | Path,
    pass_strategy: Optional[Bot] = None,
    rng: Optional[random.Random] = None,
) -> MLBot:
    """Load a trained model from disk and return an MLBot.

    Args:
        model_path: path to the pickled model file
        pass_strategy: bot to delegate pass_cards to
        rng: optional RNG for fallbacks

    Returns:
        An MLBot ready to play.
    """
    with open(model_path, "rb") as f:
        data = pickle.load(f)

    model = data["model"]
    extractor_name = data["extractor_class_name"]

    if extractor_name not in EXTRACTOR_REGISTRY:
        raise ValueError(
            f"Unknown extractor: {extractor_name}. "
            f"Available: {list(EXTRACTOR_REGISTRY.keys())}"
        )

    extractor = EXTRACTOR_REGISTRY[extractor_name]()
    return MLBot(
        model=model,
        extractor=extractor,
        pass_strategy=pass_strategy,
        rng=rng,
    )
