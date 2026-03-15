"""Tests for hearts.data.schema — card index mapping and serialization."""

import json

import pytest

from hearts.data.schema import (
    card_to_index,
    deserialize_pass_view,
    deserialize_player_view,
    index_to_card,
    serialize_pass_view,
    serialize_player_view,
)
from hearts.state import PassDirection, PassView, PlayerView
from hearts.types import Card, Suit


class TestCardIndexMapping:
    """Card index mapping: index = suit_ordinal * 13 + (rank - 2)."""

    def test_known_values(self):
        assert card_to_index(Card(Suit.CLUBS, 2)) == 0
        assert card_to_index(Card(Suit.CLUBS, 14)) == 12
        assert card_to_index(Card(Suit.DIAMONDS, 2)) == 13
        assert card_to_index(Card(Suit.SPADES, 12)) == 36
        assert card_to_index(Card(Suit.HEARTS, 14)) == 51

    def test_queen_of_spades(self):
        qs = Card(Suit.SPADES, 12)
        # Suit.SPADES = 2, rank 12 → 2*13 + (12-2) = 36
        assert card_to_index(qs) == 36

    def test_round_trip_all_cards(self):
        for suit in Suit:
            for rank in range(2, 15):
                card = Card(suit, rank)
                assert index_to_card(card_to_index(card)) == card

    def test_all_indices_unique(self):
        indices = set()
        for suit in Suit:
            for rank in range(2, 15):
                idx = card_to_index(Card(suit, rank))
                assert idx not in indices
                indices.add(idx)
        assert indices == set(range(52))

    def test_index_to_card_round_trip(self):
        for i in range(52):
            assert card_to_index(index_to_card(i)) == i


class TestSerializePlayerView:
    """Serialization and deserialization of PlayerView."""

    def _make_view(self, trick_num=3) -> PlayerView:
        """Create a representative PlayerView for testing."""
        hand = {
            Card(Suit.CLUBS, 2),
            Card(Suit.CLUBS, 5),
            Card(Suit.DIAMONDS, 10),
            Card(Suit.HEARTS, 14),
            Card(Suit.SPADES, 7),
        }
        trick_so_far = [
            (0, Card(Suit.DIAMONDS, 3)),
            (1, Card(Suit.DIAMONDS, 8)),
        ]
        cards_played = [
            # Trick 1
            (1, 0, Card(Suit.CLUBS, 2)),
            (1, 1, Card(Suit.CLUBS, 9)),
            (1, 2, Card(Suit.CLUBS, 11)),
            (1, 3, Card(Suit.CLUBS, 4)),
            # Trick 2
            (2, 2, Card(Suit.HEARTS, 3)),
            (2, 3, Card(Suit.HEARTS, 6)),
            (2, 0, Card(Suit.HEARTS, 2)),
            (2, 1, Card(Suit.HEARTS, 5)),
        ]
        return PlayerView(
            hand=hand,
            player_index=2,
            trick_so_far=trick_so_far,
            lead_suit=Suit.DIAMONDS,
            hearts_broken=True,
            is_first_trick=False,
            cards_played_this_round=cards_played,
            tricks_won_by_player=[1, 1, 0, 0],
            points_taken_by_player=[0, 0, 2, 2],
            pass_direction=PassDirection.LEFT,
            cards_received_in_pass=[
                Card(Suit.CLUBS, 5),
                Card(Suit.DIAMONDS, 10),
                Card(Suit.HEARTS, 14),
            ],
            legal_plays=[Card(Suit.DIAMONDS, 10)],
            cumulative_scores=[22, 45, 18, 31],
        )

    def test_round_trip(self):
        view = self._make_view()
        data = serialize_player_view(view)
        # Verify JSON-serializable
        json_str = json.dumps(data)
        data2 = json.loads(json_str)
        restored = deserialize_player_view(data2)

        assert restored.hand == view.hand
        assert restored.player_index == view.player_index
        assert restored.trick_so_far == view.trick_so_far
        assert restored.lead_suit == view.lead_suit
        assert restored.hearts_broken == view.hearts_broken
        assert restored.is_first_trick == view.is_first_trick
        assert (
            restored.cards_played_this_round
            == view.cards_played_this_round
        )
        assert (
            restored.tricks_won_by_player == view.tricks_won_by_player
        )
        assert (
            restored.points_taken_by_player
            == view.points_taken_by_player
        )
        assert restored.pass_direction == view.pass_direction
        assert (
            restored.cards_received_in_pass
            == view.cards_received_in_pass
        )
        assert restored.legal_plays == view.legal_plays
        assert restored.cumulative_scores == view.cumulative_scores

    def test_cards_played_nested_format(self):
        view = self._make_view()
        data = serialize_player_view(view)
        # Should be list of tricks, each a list of [player, card_idx]
        cpr = data["cards_played_this_round"]
        assert len(cpr) == 2  # 2 completed tricks
        assert len(cpr[0]) == 4  # 4 plays per trick
        assert len(cpr[1]) == 4
        # Each play is [player_index, card_index]
        assert len(cpr[0][0]) == 2

    def test_null_lead_suit(self):
        view = self._make_view()
        # When leading, lead_suit is None
        view = PlayerView(
            hand=view.hand,
            player_index=view.player_index,
            trick_so_far=[],
            lead_suit=None,
            hearts_broken=view.hearts_broken,
            is_first_trick=view.is_first_trick,
            cards_played_this_round=view.cards_played_this_round,
            tricks_won_by_player=view.tricks_won_by_player,
            points_taken_by_player=view.points_taken_by_player,
            pass_direction=view.pass_direction,
            cards_received_in_pass=view.cards_received_in_pass,
            legal_plays=list(view.hand),
            cumulative_scores=view.cumulative_scores,
        )
        data = serialize_player_view(view)
        assert data["lead_suit"] is None
        restored = deserialize_player_view(data)
        assert restored.lead_suit is None

    def test_empty_cards_played(self):
        """First play of the round — no cards played yet."""
        view = PlayerView(
            hand={Card(Suit.CLUBS, 2)},
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
            legal_plays=[Card(Suit.CLUBS, 2)],
            cumulative_scores=[0, 0, 0, 0],
        )
        data = serialize_player_view(view)
        assert data["cards_played_this_round"] == []
        restored = deserialize_player_view(data)
        assert restored.cards_played_this_round == []


class TestSerializePassView:
    """Serialization and deserialization of PassView."""

    def test_round_trip(self):
        hand = set(
            Card(suit, rank)
            for suit in Suit
            for rank in range(2, 5)
        )
        hand.add(Card(Suit.HEARTS, 14))
        view = PassView(
            hand=hand,
            player_index=1,
            pass_direction=PassDirection.RIGHT,
            cumulative_scores=[10, 20, 30, 40],
            round_number=3,
        )
        data = serialize_pass_view(view)
        json_str = json.dumps(data)
        data2 = json.loads(json_str)
        restored = deserialize_pass_view(data2)

        assert restored.hand == view.hand
        assert restored.player_index == view.player_index
        assert restored.pass_direction == view.pass_direction
        assert restored.cumulative_scores == view.cumulative_scores
        assert restored.round_number == view.round_number

    def test_hand_as_sorted_indices(self):
        hand = {Card(Suit.HEARTS, 14), Card(Suit.CLUBS, 2)}
        view = PassView(
            hand=hand,
            player_index=0,
            pass_direction=PassDirection.LEFT,
            cumulative_scores=[0, 0, 0, 0],
            round_number=0,
        )
        data = serialize_pass_view(view)
        assert data["hand"] == sorted(data["hand"])
