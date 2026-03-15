"""Feature extraction from PlayerView for ML models.

This is THE FILE students edit. The goal is to convert a PlayerView
(the game state a bot sees when choosing a card) into a numeric vector
that a machine learning model can learn from.

Think of it this way: the model sees a list of numbers, not cards.
Your feature extractor is the translator. A good translator tells the
model what matters; a bad one buries the signal in noise.

Students should modify StudentFeatureExtractor to add new features.
The model and training pipeline are fixed — feature engineering is the
entire assignment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from hearts.state import PlayerView
from hearts.types import Card, Suit


class FeatureExtractor(ABC):
    """Base class for feature extractors.

    Subclasses must implement extract() and feature_names(). The two
    must be consistent: feature_names() returns one name per dimension
    of the vector that extract() produces.
    """

    @abstractmethod
    def extract(self, view: PlayerView) -> np.ndarray:
        """Convert a PlayerView into a numeric feature vector."""

    @abstractmethod
    def feature_names(self) -> list[str]:
        """Return human-readable names for each feature dimension.

        Must match the length of the array returned by extract().
        """

    def extract_from_record(self, record: dict) -> np.ndarray | None:
        """Optional fast path: extract features directly from a raw dict.

        Returns None to fall back to deserialize + extract(). Subclasses
        can override this to avoid the overhead of creating Card objects
        when the record already contains integer card indices.
        """
        return None


def _extract_basic_features(
    view: PlayerView,
) -> tuple[list[float], list[str]]:
    """Extract the basic feature set shared by both extractors.

    Returns a (features, names) tuple so callers can extend them.
    """
    features: list[float] = []
    names: list[str] = []

    # ---- Hand (52 features) ----
    # Binary vector: 1 if the card is in your hand, 0 otherwise.
    # This tells the model what cards you could potentially play.
    hand_indices = {
        suit.value * 13 + (rank - 2)
        for suit in Suit
        for rank in range(2, 15)
        if Card(suit, rank) in view.hand
    }
    for i in range(52):
        features.append(1.0 if i in hand_indices else 0.0)
        suit = Suit(i // 13)
        rank = (i % 13) + 2
        rank_name = {11: "J", 12: "Q", 13: "K", 14: "A"}.get(
            rank, str(rank)
        )
        names.append(f"hand_{rank_name}{suit.name[0]}")

    # ---- Legal plays (52 features) ----
    # Binary vector: 1 if the card is a legal play right now.
    # This encodes the rules of Hearts (follow suit, first-trick
    # restrictions, hearts leading) so the model knows which cards
    # it's actually choosing between.
    legal_indices = {
        c.suit.value * 13 + (c.rank - 2)
        for c in view.legal_plays
    }
    for i in range(52):
        features.append(1.0 if i in legal_indices else 0.0)
        suit = Suit(i // 13)
        rank = (i % 13) + 2
        rank_name = {11: "J", 12: "Q", 13: "K", 14: "A"}.get(
            rank, str(rank)
        )
        names.append(f"legal_{rank_name}{suit.name[0]}")

    # ---- Cards played this round (52 features) ----
    # Binary vector: 1 if the card has been played in a previous trick.
    # This tells the model what cards are gone and helps it reason about
    # what other players might still hold.
    played_indices: set[int] = set()
    for _, _, card in view.cards_played_this_round:
        played_indices.add(
            card.suit.value * 13 + (card.rank - 2)
        )
    for i in range(52):
        features.append(1.0 if i in played_indices else 0.0)
        suit = Suit(i // 13)
        rank = (i % 13) + 2
        rank_name = {11: "J", 12: "Q", 13: "K", 14: "A"}.get(
            rank, str(rank)
        )
        names.append(f"played_{rank_name}{suit.name[0]}")

    # ---- Current trick (52 features) ----
    # Positional encoding: each card played in the current trick gets
    # a value based on when it was played (0.25 for first, 0.5 for
    # second, 0.75 for third, 1.0 for fourth). This tells the model
    # both WHAT was played and in WHAT ORDER.
    trick_vec = [0.0] * 52
    for position, (_, card) in enumerate(view.trick_so_far):
        idx = card.suit.value * 13 + (card.rank - 2)
        trick_vec[idx] = (position + 1) * 0.25
    for i in range(52):
        features.append(trick_vec[i])
        suit = Suit(i // 13)
        rank = (i % 13) + 2
        rank_name = {11: "J", 12: "Q", 13: "K", 14: "A"}.get(
            rank, str(rank)
        )
        names.append(f"trick_{rank_name}{suit.name[0]}")

    # ---- Suit counts in hand (4 features) ----
    # Number of cards of each suit, normalized by 13. Gives the model
    # direct access to suit lengths — critical for strategy (short suit
    # = close to voiding = can dump point cards).
    for suit in Suit:
        count = sum(1 for c in view.hand if c.suit == suit)
        features.append(count / 13.0)
        names.append(f"suit_count_{suit.name}")

    # ---- Trick position (1 feature) ----
    # How many cards have been played in the current trick (0-3),
    # normalized to 0.0-1.0. Playing first (leading) vs last is a
    # huge strategic difference: when you lead, you're guessing;
    # when you play last, you know exactly what you'll take.
    features.append(len(view.trick_so_far) / 3.0)
    names.append("trick_position")

    # ---- Hearts broken (1 feature) ----
    # Binary flag: 1 if hearts have been played this round.
    # Until hearts are broken, you can't lead with hearts.
    features.append(1.0 if view.hearts_broken else 0.0)
    names.append("hearts_broken")

    # ---- Points taken by each player (4 features) ----
    # Points relative to the current player: my points first,
    # then left, across, right. This lets the model learn
    # "I have too many points" regardless of which seat we're in.
    me = view.player_index
    for offset, label in enumerate(
        ["my", "left", "across", "right"]
    ):
        idx = (me + offset) % 4
        features.append(float(view.points_taken_by_player[idx]))
        names.append(f"points_{label}")

    # ---- Trick number (1 feature) ----
    # Normalized to 0.0-1.0 (trick 1 = 0.0, trick 13 = 1.0).
    # Early-round strategy differs from late-round strategy.
    if view.cards_played_this_round:
        trick_num = view.cards_played_this_round[-1][0]
        if not view.trick_so_far:
            trick_num += 1
    else:
        trick_num = 1
    features.append((trick_num - 1) / 12.0)
    names.append("trick_number")

    # ---- Cumulative scores (4 features) ----
    # Game-level scores divided by 100 for normalization.
    # Relative to current player: my score first, then left,
    # across, right. Lets the model learn "I'm close to losing"
    # regardless of seat position.
    me = view.player_index
    for offset, label in enumerate(
        ["my", "left", "across", "right"]
    ):
        idx = (me + offset) % 4
        features.append(view.cumulative_scores[idx] / 100.0)
        names.append(f"cum_score_{label}")

    # ---- What BasicFeatureExtractor does NOT capture ----
    #
    # This is your starting point for feature engineering. The basic
    # extractor is missing a lot of useful information:
    #
    # - Which suits you are void in (can't follow suit = can dump)
    # - Whether dangerous cards (Q♠, K♠, A♠) have been played yet
    # - How many hearts are still unplayed
    # - Your position in the current trick (leading vs. last to play)
    # - Which players appear to be void in which suits (based on
    #   them not following suit in earlier tricks)
    # - How many cards remain in each suit in your hand
    # - Whether a shoot-the-moon attempt is in progress

    return features, names


def _extract_basic_from_record(record: dict) -> np.ndarray:
    """Fast path: extract basic features directly from a serialized dict.

    Avoids creating Card objects — works with integer card indices
    already present in the record. Produces the same output as
    _extract_basic_features() but ~10x faster for bulk processing.
    """
    features = np.zeros(223, dtype=np.float32)
    offset = 0

    # Hand (52 features)
    for ci in record["hand"]:
        features[offset + ci] = 1.0
    offset += 52

    # Legal plays (52 features)
    for ci in record["legal_plays"]:
        features[offset + ci] = 1.0
    offset += 52

    # Cards played this round (52 features)
    for trick in record["cards_played_this_round"]:
        for _, ci in trick:
            features[offset + ci] = 1.0
    offset += 52

    # Current trick with positional encoding (52 features)
    for pos, (_, ci) in enumerate(record["trick_so_far"]):
        features[offset + ci] = (pos + 1) * 0.25
    offset += 52

    # Suit counts in hand (4 features)
    for ci in record["hand"]:
        features[offset + ci // 13] += 1.0
    features[offset:offset + 4] /= 13.0
    offset += 4

    # Trick position (1 feature)
    features[offset] = len(record["trick_so_far"]) / 3.0
    offset += 1

    # Hearts broken (1 feature)
    features[offset] = 1.0 if record["hearts_broken"] else 0.0
    offset += 1

    # Points taken by each player (4 features, relative to current)
    me = record["player_index"]
    for off in range(4):
        idx = (me + off) % 4
        features[offset + off] = float(
            record["points_taken_by_player"][idx]
        )
    offset += 4

    # Trick number (1 feature)
    features[offset] = (record["trick_number"] - 1) / 12.0
    offset += 1

    # Cumulative scores (4 features, relative to current)
    for off in range(4):
        idx = (me + off) % 4
        features[offset + off] = (
            record["cumulative_scores"][idx] / 100.0
        )

    return features


class BasicFeatureExtractor(FeatureExtractor):
    """Reference feature extractor with ~166 features.

    This is the baseline. It encodes the essential game state but
    misses many strategic signals. See the comments at the end of
    _extract_basic_features() for what it does NOT capture.
    """

    def __init__(self) -> None:
        self._cached_names: list[str] | None = None

    def extract(self, view: PlayerView) -> np.ndarray:
        features, _ = _extract_basic_features(view)
        return np.array(features, dtype=np.float32)

    def extract_from_record(self, record: dict) -> np.ndarray:
        return _extract_basic_from_record(record)

    def feature_names(self) -> list[str]:
        if self._cached_names is not None:
            return self._cached_names

        # Build names — deterministic and view-independent.
        names: list[str] = []

        for i in range(52):
            suit = Suit(i // 13)
            rank = (i % 13) + 2
            rank_name = {11: "J", 12: "Q", 13: "K", 14: "A"}.get(
                rank, str(rank)
            )
            names.append(f"hand_{rank_name}{suit.name[0]}")
        for i in range(52):
            suit = Suit(i // 13)
            rank = (i % 13) + 2
            rank_name = {11: "J", 12: "Q", 13: "K", 14: "A"}.get(
                rank, str(rank)
            )
            names.append(f"legal_{rank_name}{suit.name[0]}")
        for i in range(52):
            suit = Suit(i // 13)
            rank = (i % 13) + 2
            rank_name = {11: "J", 12: "Q", 13: "K", 14: "A"}.get(
                rank, str(rank)
            )
            names.append(f"played_{rank_name}{suit.name[0]}")
        for i in range(52):
            suit = Suit(i // 13)
            rank = (i % 13) + 2
            rank_name = {11: "J", 12: "Q", 13: "K", 14: "A"}.get(
                rank, str(rank)
            )
            names.append(f"trick_{rank_name}{suit.name[0]}")

        for suit in Suit:
            names.append(f"suit_count_{suit.name}")
        names.append("trick_position")
        names.append("hearts_broken")
        for label in ["my", "left", "across", "right"]:
            names.append(f"points_{label}")
        names.append("trick_number")
        for label in ["my", "left", "across", "right"]:
            names.append(f"cum_score_{label}")

        self._cached_names = names
        return names


class StudentFeatureExtractor(FeatureExtractor):
    """Your feature extractor — modify this class!

    Start by understanding what BasicFeatureExtractor provides (read
    the comments in _extract_basic_features above). Then add your own
    features below the marked section.

    Workflow:
        1. Add features in the YOUR FEATURES BELOW section
        2. Run: python scripts/train_model.py
        3. Run: python scripts/evaluate_model.py
        4. Read the report, iterate
    """

    def __init__(self) -> None:
        self._cached_names: list[str] | None = None

    def extract(self, view: PlayerView) -> np.ndarray:
        features, _ = _extract_basic_features(view)

        # === YOUR FEATURES BELOW ===
        # Add new features by appending to the features list.
        # Each feature should be a single float value.
        #
        # Example: suit void tracking (uncomment and complete)
        # for suit in Suit:
        #     cards_of_suit = [c for c in view.hand if c.suit == suit]
        #     features.append(1.0 if len(cards_of_suit) == 0 else 0.0)

        return np.array(features, dtype=np.float32)

    def feature_names(self) -> list[str]:
        if self._cached_names is not None:
            return self._cached_names

        _, names = _extract_basic_features(
            _dummy_view()
        )

        # === YOUR FEATURE NAMES BELOW ===
        # Add one name per feature you added above, in the same order.
        #
        # Example: suit void tracking (uncomment and complete)
        # for suit in Suit:
        #     names.append(f"void_{suit.name}")

        self._cached_names = names
        return names


def _dummy_view() -> PlayerView:
    """Create a minimal PlayerView for generating feature names."""
    from hearts.state import PassDirection

    return PlayerView(
        hand=set(),
        player_index=0,
        trick_so_far=[],
        lead_suit=None,
        hearts_broken=False,
        is_first_trick=True,
        cards_played_this_round=[],
        tricks_won_by_player=[0, 0, 0, 0],
        points_taken_by_player=[0, 0, 0, 0],
        pass_direction=PassDirection.NONE,
        cards_received_in_pass=[],
        legal_plays=[],
        cumulative_scores=[0, 0, 0, 0],
    )
