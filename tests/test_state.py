"""Tests for hearts.state — TrickState, RoundState, GameState, etc."""

import pytest

from hearts.types import Card, Suit
from hearts.state import (
    GameState,
    PassDirection,
    PASS_ROTATION,
    RoundState,
    TrickState,
)


class TestTrickState:
    def test_winner_highest_lead_suit(self):
        trick = TrickState(
            cards_played=[
                (0, Card(Suit.CLUBS, 5)),
                (1, Card(Suit.CLUBS, 10)),
                (2, Card(Suit.CLUBS, 3)),
                (3, Card(Suit.CLUBS, 14)),
            ],
            lead_suit=Suit.CLUBS,
        )
        assert trick.winner() == 3

    def test_winner_ignores_off_suit(self):
        trick = TrickState(
            cards_played=[
                (0, Card(Suit.CLUBS, 5)),
                (1, Card(Suit.HEARTS, 14)),
                (2, Card(Suit.CLUBS, 10)),
                (3, Card(Suit.SPADES, 13)),
            ],
            lead_suit=Suit.CLUBS,
        )
        assert trick.winner() == 2

    def test_winner_only_one_of_lead_suit(self):
        trick = TrickState(
            cards_played=[
                (0, Card(Suit.DIAMONDS, 2)),
                (1, Card(Suit.HEARTS, 14)),
                (2, Card(Suit.SPADES, 14)),
                (3, Card(Suit.CLUBS, 14)),
            ],
            lead_suit=Suit.DIAMONDS,
        )
        assert trick.winner() == 0

    def test_winner_empty_trick_raises(self):
        trick = TrickState()
        with pytest.raises(ValueError):
            trick.winner()

    def test_winner_no_lead_suit_raises(self):
        trick = TrickState(
            cards_played=[(0, Card(Suit.CLUBS, 5))],
            lead_suit=None,
        )
        with pytest.raises(ValueError):
            trick.winner()


class TestRoundState:
    def test_scores_no_points(self):
        state = RoundState(hands=[set() for _ in range(4)])
        state.cards_taken[0] = [Card(Suit.CLUBS, 5)]
        assert state.scores == [0, 0, 0, 0]

    def test_scores_hearts(self):
        state = RoundState(hands=[set() for _ in range(4)])
        state.cards_taken[0] = [
            Card(Suit.HEARTS, 2),
            Card(Suit.HEARTS, 3),
        ]
        state.cards_taken[1] = [Card(Suit.HEARTS, 5)]
        assert state.scores == [2, 1, 0, 0]

    def test_scores_queen_of_spades(self):
        state = RoundState(hands=[set() for _ in range(4)])
        state.cards_taken[2] = [Card(Suit.SPADES, 12)]
        assert state.scores == [0, 0, 13, 0]

    def test_scores_shoot_the_moon(self):
        state = RoundState(hands=[set() for _ in range(4)])
        # Player 1 takes all hearts and Q♠
        all_hearts = [Card(Suit.HEARTS, r) for r in range(2, 15)]
        state.cards_taken[1] = all_hearts + [Card(Suit.SPADES, 12)]
        scores = state.scores
        assert scores[1] == 0
        assert scores[0] == 26
        assert scores[2] == 26
        assert scores[3] == 26

    def test_scores_partial_points(self):
        state = RoundState(hands=[set() for _ in range(4)])
        # Player 0 takes some hearts, player 2 takes Q♠
        state.cards_taken[0] = [
            Card(Suit.HEARTS, 2),
            Card(Suit.HEARTS, 3),
            Card(Suit.HEARTS, 4),
        ]
        state.cards_taken[2] = [Card(Suit.SPADES, 12)]
        assert state.scores == [3, 0, 13, 0]

    def test_hearts_broken_default_false(self):
        state = RoundState(hands=[set() for _ in range(4)])
        assert state.hearts_broken is False


class TestGameState:
    def test_game_not_over_initially(self):
        gs = GameState()
        assert gs.game_over is False

    def test_game_over_at_100(self):
        gs = GameState(
            cumulative_scores=[99, 50, 30, 100]
        )
        assert gs.game_over is True

    def test_game_not_over_at_99(self):
        gs = GameState(cumulative_scores=[99, 50, 30, 20])
        assert gs.game_over is False

    def test_pass_direction_rotation(self):
        gs = GameState()
        assert gs.pass_direction == PassDirection.LEFT

        gs.round_history.append(None)  # type: ignore
        assert gs.pass_direction == PassDirection.RIGHT

        gs.round_history.append(None)  # type: ignore
        assert gs.pass_direction == PassDirection.ACROSS

        gs.round_history.append(None)  # type: ignore
        assert gs.pass_direction == PassDirection.NONE

        gs.round_history.append(None)  # type: ignore
        assert gs.pass_direction == PassDirection.LEFT

    def test_pass_rotation_constant(self):
        assert PASS_ROTATION == [
            PassDirection.LEFT,
            PassDirection.RIGHT,
            PassDirection.ACROSS,
            PassDirection.NONE,
        ]


class TestPassDirection:
    def test_all_directions(self):
        assert len(PassDirection) == 4
        assert PassDirection.LEFT.value == "left"
        assert PassDirection.RIGHT.value == "right"
        assert PassDirection.ACROSS.value == "across"
        assert PassDirection.NONE.value == "none"
