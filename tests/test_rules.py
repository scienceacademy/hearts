"""Tests for hearts.rules — legal move validation."""

from hearts.types import Card, Suit
from hearts.rules import (
    TWO_OF_CLUBS,
    get_legal_plays,
    validate_pass,
    validate_play,
)


def _hand(*cards: Card) -> set[Card]:
    return set(cards)


C2 = Card(Suit.CLUBS, 2)
C5 = Card(Suit.CLUBS, 5)
C10 = Card(Suit.CLUBS, 10)
CA = Card(Suit.CLUBS, 14)
D3 = Card(Suit.DIAMONDS, 3)
D7 = Card(Suit.DIAMONDS, 7)
DK = Card(Suit.DIAMONDS, 13)
S5 = Card(Suit.SPADES, 5)
SQ = Card(Suit.SPADES, 12)
SK = Card(Suit.SPADES, 13)
SA = Card(Suit.SPADES, 14)
H2 = Card(Suit.HEARTS, 2)
H3 = Card(Suit.HEARTS, 3)
H5 = Card(Suit.HEARTS, 5)
H9 = Card(Suit.HEARTS, 9)
HA = Card(Suit.HEARTS, 14)


class TestFirstTrickLead:
    def test_must_lead_two_of_clubs(self):
        hand = _hand(C2, C5, D3, H5)
        legal = get_legal_plays(
            hand, [], None,
            hearts_broken=False, is_first_trick=True,
        )
        assert legal == [C2]

    def test_two_of_clubs_constant(self):
        assert TWO_OF_CLUBS == Card(Suit.CLUBS, 2)


class TestFollowSuit:
    def test_must_follow_lead_suit(self):
        hand = _hand(C5, C10, D3, H5)
        legal = get_legal_plays(
            hand, [(0, C2)], Suit.CLUBS,
            hearts_broken=False, is_first_trick=False,
        )
        assert set(legal) == {C5, C10}

    def test_cannot_follow_plays_anything(self):
        hand = _hand(D3, D7, H5, S5)
        legal = get_legal_plays(
            hand, [(0, C2)], Suit.CLUBS,
            hearts_broken=False, is_first_trick=False,
        )
        assert set(legal) == {D3, D7, H5, S5}


class TestFirstTrickRestrictions:
    def test_no_points_when_cant_follow_first_trick(self):
        hand = _hand(D3, H5, SQ)
        legal = get_legal_plays(
            hand, [(0, C2)], Suit.CLUBS,
            hearts_broken=False, is_first_trick=True,
        )
        assert legal == [D3]

    def test_all_point_cards_allowed_first_trick(self):
        """If hand is only point cards and can't follow, must play them."""
        hand = _hand(H2, H5, SQ)
        legal = get_legal_plays(
            hand, [(0, C2)], Suit.CLUBS,
            hearts_broken=False, is_first_trick=True,
        )
        assert set(legal) == {H2, H5, SQ}

    def test_follow_suit_first_trick_overrides_point_restriction(self):
        """If you can follow suit with a point card, you must."""
        hand = _hand(H2, H5, SA)
        legal = get_legal_plays(
            hand, [(0, Card(Suit.HEARTS, 7))], Suit.HEARTS,
            hearts_broken=False, is_first_trick=True,
        )
        # Must follow hearts
        assert set(legal) == {H2, H5}


class TestLeadingHearts:
    def test_cannot_lead_hearts_before_broken(self):
        hand = _hand(C5, D3, H5, H9)
        legal = get_legal_plays(
            hand, [], None,
            hearts_broken=False, is_first_trick=False,
        )
        assert H5 not in legal
        assert H9 not in legal
        assert set(legal) == {C5, D3}

    def test_can_lead_hearts_after_broken(self):
        hand = _hand(C5, D3, H5, H9)
        legal = get_legal_plays(
            hand, [], None,
            hearts_broken=True, is_first_trick=False,
        )
        assert set(legal) == {C5, D3, H5, H9}

    def test_only_hearts_can_lead_hearts(self):
        """If only hearts remain, can lead them even if not broken."""
        hand = _hand(H2, H5, H9)
        legal = get_legal_plays(
            hand, [], None,
            hearts_broken=False, is_first_trick=False,
        )
        assert set(legal) == {H2, H5, H9}


class TestValidatePlay:
    def test_legal_play(self):
        hand = _hand(C2, C5, D3)
        assert validate_play(
            C2, hand, [], None,
            hearts_broken=False, is_first_trick=True,
        )

    def test_card_not_in_hand(self):
        hand = _hand(C5, D3)
        assert not validate_play(
            C2, hand, [], None,
            hearts_broken=False, is_first_trick=True,
        )

    def test_illegal_play(self):
        hand = _hand(C2, C5, D3, H5)
        # Must play 2♣ on first trick lead
        assert not validate_play(
            C5, hand, [], None,
            hearts_broken=False, is_first_trick=True,
        )


class TestValidatePass:
    def test_valid_pass(self):
        hand = _hand(C2, C5, D3, H5, S5)
        assert validate_pass([C2, C5, D3], hand)

    def test_wrong_count(self):
        hand = _hand(C2, C5, D3, H5)
        assert not validate_pass([C2, C5], hand)
        assert not validate_pass([C2, C5, D3, H5], hand)

    def test_duplicates(self):
        hand = _hand(C2, C5, D3)
        assert not validate_pass([C2, C2, C5], hand)

    def test_card_not_in_hand(self):
        hand = _hand(C2, C5, D3)
        assert not validate_pass([C2, C5, H5], hand)

    def test_empty_pass(self):
        hand = _hand(C2, C5, D3)
        assert not validate_pass([], hand)


class TestEdgeCases:
    def test_empty_hand(self):
        legal = get_legal_plays(
            set(), [], None,
            hearts_broken=False, is_first_trick=False,
        )
        assert legal == []

    def test_single_card_hand(self):
        hand = _hand(H5)
        legal = get_legal_plays(
            hand, [], None,
            hearts_broken=False, is_first_trick=False,
        )
        # Only hearts, must lead it
        assert legal == [H5]

    def test_queen_of_spades_blocked_first_trick(self):
        """Q♠ is a point card, blocked on first trick if non-points exist."""
        hand = _hand(D3, SQ, H5)
        legal = get_legal_plays(
            hand, [(0, C2)], Suit.CLUBS,
            hearts_broken=False, is_first_trick=True,
        )
        assert legal == [D3]

    def test_results_are_sorted(self):
        hand = _hand(CA, C5, C2, C10)
        legal = get_legal_plays(
            hand, [(1, D3)], Suit.DIAMONDS,
            hearts_broken=False, is_first_trick=False,
        )
        # Can't follow diamonds, play anything — should be sorted
        assert legal == sorted(legal)
