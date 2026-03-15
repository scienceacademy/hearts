"""Tests for hearts.ml.features — feature extractors."""

import numpy as np
import pytest

from hearts.ml.features import (
    BasicFeatureExtractor,
    StudentFeatureExtractor,
)
from hearts.state import PassDirection, PlayerView
from hearts.types import Card, Suit


def _make_view(
    trick_num: int = 1,
    hearts_broken: bool = False,
    trick_so_far: list | None = None,
    cards_played: list | None = None,
    hand: set | None = None,
    points: list | None = None,
    cumulative_scores: list | None = None,
) -> PlayerView:
    """Build a PlayerView with sensible defaults for testing."""
    if hand is None:
        hand = {
            Card(Suit.CLUBS, 2),
            Card(Suit.CLUBS, 5),
            Card(Suit.CLUBS, 9),
            Card(Suit.DIAMONDS, 3),
            Card(Suit.DIAMONDS, 10),
            Card(Suit.SPADES, 7),
            Card(Suit.SPADES, 12),
            Card(Suit.HEARTS, 4),
            Card(Suit.HEARTS, 8),
            Card(Suit.HEARTS, 14),
        }
    if trick_so_far is None:
        trick_so_far = []
    if cards_played is None:
        cards_played = []
    if points is None:
        points = [0, 0, 0, 0]
    if cumulative_scores is None:
        cumulative_scores = [0, 0, 0, 0]

    lead_suit = None
    if trick_so_far:
        lead_suit = trick_so_far[0][1].suit

    return PlayerView(
        hand=hand,
        player_index=2,
        trick_so_far=trick_so_far,
        lead_suit=lead_suit,
        hearts_broken=hearts_broken,
        is_first_trick=(trick_num == 1),
        cards_played_this_round=cards_played,
        tricks_won_by_player=[0, 0, 0, 0],
        points_taken_by_player=points,
        pass_direction=PassDirection.LEFT,
        cards_received_in_pass=[
            Card(Suit.CLUBS, 5),
            Card(Suit.DIAMONDS, 3),
            Card(Suit.HEARTS, 14),
        ],
        legal_plays=list(hand)[:3],
        cumulative_scores=cumulative_scores,
    )


class TestBasicFeatureExtractor:
    """BasicFeatureExtractor produces consistent, valid features."""

    def test_output_dimension_matches_names(self):
        ext = BasicFeatureExtractor()
        view = _make_view()
        vec = ext.extract(view)
        names = ext.feature_names()
        assert len(vec) == len(names), (
            f"extract() returned {len(vec)} features but "
            f"feature_names() returned {len(names)} names. "
            f"Each feature needs exactly one name."
        )

    def test_expected_dimension(self):
        ext = BasicFeatureExtractor()
        view = _make_view()
        vec = ext.extract(view)
        # 52 hand + 52 legal + 52 played + 52 trick + 4 suit_count
        # + 1 trick_position + 1 hearts_broken
        # + 4 points + 1 trick_number + 4 cumulative = 223
        assert len(vec) == 223

    def test_returns_float32(self):
        ext = BasicFeatureExtractor()
        view = _make_view()
        vec = ext.extract(view)
        assert vec.dtype == np.float32

    def test_no_nan_or_inf(self):
        ext = BasicFeatureExtractor()
        view = _make_view()
        vec = ext.extract(view)
        assert not np.any(np.isnan(vec))
        assert not np.any(np.isinf(vec))

    def test_hand_features_are_binary(self):
        ext = BasicFeatureExtractor()
        view = _make_view()
        vec = ext.extract(view)
        hand_features = vec[:52]
        assert all(v in (0.0, 1.0) for v in hand_features)

    def test_hand_features_count_matches(self):
        ext = BasicFeatureExtractor()
        view = _make_view()
        vec = ext.extract(view)
        hand_features = vec[:52]
        assert sum(hand_features) == len(view.hand)

    def test_played_features_empty_first_trick(self):
        ext = BasicFeatureExtractor()
        view = _make_view(trick_num=1, cards_played=[])
        vec = ext.extract(view)
        # legal plays at offset 52, played at offset 104
        played_features = vec[104:156]
        assert sum(played_features) == 0.0

    def test_played_features_after_tricks(self):
        cards_played = [
            (1, 0, Card(Suit.CLUBS, 2)),
            (1, 1, Card(Suit.CLUBS, 9)),
            (1, 2, Card(Suit.CLUBS, 11)),
            (1, 3, Card(Suit.CLUBS, 4)),
        ]
        ext = BasicFeatureExtractor()
        view = _make_view(trick_num=2, cards_played=cards_played)
        vec = ext.extract(view)
        played_features = vec[104:156]
        assert sum(played_features) == 4.0

    def test_trick_features_positional_encoding(self):
        trick_so_far = [
            (0, Card(Suit.DIAMONDS, 5)),
            (1, Card(Suit.DIAMONDS, 10)),
        ]
        ext = BasicFeatureExtractor()
        view = _make_view(trick_so_far=trick_so_far)
        vec = ext.extract(view)
        trick_features = vec[156:208]
        # 5♦ = index 16 (DIAMONDS=1, rank 5 → 1*13 + 3 = 16)
        assert trick_features[16] == pytest.approx(0.25)
        # 10♦ = index 21 (DIAMONDS=1, rank 10 → 1*13 + 8 = 21)
        assert trick_features[21] == pytest.approx(0.50)

    def test_hearts_broken_flag(self):
        ext = BasicFeatureExtractor()
        view_no = _make_view(hearts_broken=False)
        view_yes = _make_view(hearts_broken=True)
        vec_no = ext.extract(view_no)
        vec_yes = ext.extract(view_yes)
        # hearts_broken at offset 213
        assert vec_no[213] == 0.0
        assert vec_yes[213] == 1.0

    def test_trick_number_normalization(self):
        ext = BasicFeatureExtractor()
        # Trick 1 → 0.0
        view1 = _make_view(trick_num=1, cards_played=[])
        vec1 = ext.extract(view1)
        # trick_number at offset 218
        assert vec1[218] == pytest.approx(0.0)

        # Trick 13: 12 completed tricks (48 cards) + leading
        cards_played = [
            (t, p, Card(Suit.CLUBS, 2))
            for t in range(1, 13)
            for p in range(4)
        ]
        # Use unique cards to avoid confusion — just need trick nums
        view13 = _make_view(trick_num=13, cards_played=cards_played)
        vec13 = ext.extract(view13)
        assert vec13[218] == pytest.approx(1.0)

    def test_cumulative_scores_normalized(self):
        ext = BasicFeatureExtractor()
        # Scores are relative to player_index=2:
        # my(p2)=75, left(p3)=100, across(p0)=25, right(p1)=50
        view = _make_view(cumulative_scores=[25, 50, 75, 100])
        vec = ext.extract(view)
        # cumulative scores at offset 219-222
        assert vec[219] == pytest.approx(0.75)
        assert vec[220] == pytest.approx(1.00)
        assert vec[221] == pytest.approx(0.25)
        assert vec[222] == pytest.approx(0.50)

    def test_consistent_across_calls(self):
        ext = BasicFeatureExtractor()
        view = _make_view()
        vec1 = ext.extract(view)
        vec2 = ext.extract(view)
        np.testing.assert_array_equal(vec1, vec2)

    def test_different_views_different_features(self):
        ext = BasicFeatureExtractor()
        view1 = _make_view(hearts_broken=False)
        view2 = _make_view(hearts_broken=True)
        vec1 = ext.extract(view1)
        vec2 = ext.extract(view2)
        assert not np.array_equal(vec1, vec2)


class TestStudentFeatureExtractor:
    """StudentFeatureExtractor starts identical to Basic."""

    def test_output_dimension_matches_names(self):
        ext = StudentFeatureExtractor()
        view = _make_view()
        vec = ext.extract(view)
        names = ext.feature_names()
        assert len(vec) == len(names), (
            f"extract() returned {len(vec)} features but "
            f"feature_names() returned {len(names)} names. "
            f"Did you add a feature value without adding a "
            f"corresponding name (or vice versa)?"
        )

    def test_extends_basic(self):
        basic = BasicFeatureExtractor()
        student = StudentFeatureExtractor()
        view = _make_view()
        basic_len = len(basic.extract(view))
        student_len = len(student.extract(view))
        # Student extractor should have at least as many features
        assert student_len >= basic_len
        # The basic portion should match
        np.testing.assert_array_equal(
            basic.extract(view), student.extract(view)[:basic_len]
        )

    def test_no_nan_or_inf(self):
        ext = StudentFeatureExtractor()
        view = _make_view()
        vec = ext.extract(view)
        assert not np.any(np.isnan(vec))
        assert not np.any(np.isinf(vec))

    def test_feature_names_no_duplicates(self):
        ext = StudentFeatureExtractor()
        names = ext.feature_names()
        assert len(names) == len(set(names))


class TestExtractorsWithVariousViews:
    """Both extractors handle diverse game states without errors."""

    @pytest.fixture(params=["basic", "student"])
    def extractor(self, request):
        if request.param == "basic":
            return BasicFeatureExtractor()
        return StudentFeatureExtractor()

    def test_empty_hand(self, extractor):
        view = _make_view(hand=set())
        vec = extractor.extract(view)
        names = extractor.feature_names()
        assert len(vec) == len(names)
        assert not np.any(np.isnan(vec))

    def test_full_hand(self, extractor):
        hand = {
            Card(suit, rank)
            for suit in Suit
            for rank in range(2, 15)
        }
        # 52-card hand (degenerate but shouldn't crash)
        view = _make_view(hand=hand)
        vec = extractor.extract(view)
        assert vec[:52].sum() == 52.0

    def test_mid_game_state(self, extractor):
        cards_played = [
            (1, 0, Card(Suit.CLUBS, 2)),
            (1, 1, Card(Suit.CLUBS, 9)),
            (1, 2, Card(Suit.CLUBS, 11)),
            (1, 3, Card(Suit.CLUBS, 4)),
            (2, 2, Card(Suit.HEARTS, 3)),
            (2, 3, Card(Suit.HEARTS, 6)),
            (2, 0, Card(Suit.HEARTS, 2)),
            (2, 1, Card(Suit.HEARTS, 5)),
        ]
        trick_so_far = [
            (0, Card(Suit.DIAMONDS, 5)),
            (1, Card(Suit.DIAMONDS, 10)),
        ]
        view = _make_view(
            trick_num=3,
            hearts_broken=True,
            cards_played=cards_played,
            trick_so_far=trick_so_far,
            points=[0, 2, 0, 2],
            cumulative_scores=[30, 40, 20, 50],
        )
        vec = extractor.extract(view)
        names = extractor.feature_names()
        assert len(vec) == len(names)
        assert not np.any(np.isnan(vec))
        assert not np.any(np.isinf(vec))

    def test_no_pass_direction(self, extractor):
        view = _make_view()
        view = PlayerView(
            hand=view.hand,
            player_index=view.player_index,
            trick_so_far=view.trick_so_far,
            lead_suit=view.lead_suit,
            hearts_broken=view.hearts_broken,
            is_first_trick=view.is_first_trick,
            cards_played_this_round=view.cards_played_this_round,
            tricks_won_by_player=view.tricks_won_by_player,
            points_taken_by_player=view.points_taken_by_player,
            pass_direction=PassDirection.NONE,
            cards_received_in_pass=[],
            legal_plays=list(view.hand)[:3],
            cumulative_scores=view.cumulative_scores,
        )
        vec = extractor.extract(view)
        assert len(vec) == len(extractor.feature_names())


class TestFastPathEquivalence:
    """extract_from_record must produce identical output to extract."""

    def test_basic_fast_path_matches_normal(self):
        """Run real games and verify fast path matches normal path."""
        from hearts.bots.rule_bot import RuleBot
        from hearts.data.generator import generate_game_data
        from hearts.data.schema import deserialize_player_view

        ext = BasicFeatureExtractor()
        bots = [RuleBot() for _ in range(4)]
        _, play_recs, _ = generate_game_data(bots, seed=99)

        # Check a sample of records across the game
        for rec in play_recs[::10]:
            view = deserialize_player_view(rec)
            normal = ext.extract(view)
            fast = ext.extract_from_record(rec)
            np.testing.assert_array_almost_equal(
                normal, fast,
                err_msg="Fast path output differs from normal path",
            )
