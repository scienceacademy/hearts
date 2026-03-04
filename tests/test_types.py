"""Tests for hearts.types — Card, Suit, Deck, and helper functions."""

import random

from hearts.types import Card, Deck, Suit, card_points, is_heart, is_queen_of_spades


class TestSuit:
    def test_ordering(self):
        assert Suit.CLUBS < Suit.DIAMONDS < Suit.SPADES < Suit.HEARTS

    def test_symbols(self):
        assert Suit.CLUBS.symbol == "\u2663"
        assert Suit.DIAMONDS.symbol == "\u2666"
        assert Suit.SPADES.symbol == "\u2660"
        assert Suit.HEARTS.symbol == "\u2665"


class TestCard:
    def test_creation(self):
        card = Card(Suit.HEARTS, 14)
        assert card.suit == Suit.HEARTS
        assert card.rank == 14

    def test_invalid_rank_low(self):
        try:
            Card(Suit.HEARTS, 1)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_invalid_rank_high(self):
        try:
            Card(Suit.HEARTS, 15)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_str_number_cards(self):
        assert str(Card(Suit.HEARTS, 2)) == "2\u2665"
        assert str(Card(Suit.CLUBS, 10)) == "10\u2663"

    def test_str_face_cards(self):
        assert str(Card(Suit.SPADES, 11)) == "J\u2660"
        assert str(Card(Suit.SPADES, 12)) == "Q\u2660"
        assert str(Card(Suit.DIAMONDS, 13)) == "K\u2666"
        assert str(Card(Suit.HEARTS, 14)) == "A\u2665"

    def test_hashable(self):
        card = Card(Suit.HEARTS, 14)
        s = {card, card}
        assert len(s) == 1

    def test_equality(self):
        assert Card(Suit.HEARTS, 14) == Card(Suit.HEARTS, 14)
        assert Card(Suit.HEARTS, 14) != Card(Suit.SPADES, 14)

    def test_comparable(self):
        # Cards are ordered by (suit, rank) due to frozen dataclass with order=True
        assert Card(Suit.CLUBS, 2) < Card(Suit.CLUBS, 3)
        assert Card(Suit.CLUBS, 14) < Card(Suit.DIAMONDS, 2)

    def test_frozen(self):
        card = Card(Suit.HEARTS, 14)
        try:
            card.rank = 13  # type: ignore
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass


class TestCardPoints:
    def test_hearts_worth_one(self):
        for rank in range(2, 15):
            assert card_points(Card(Suit.HEARTS, rank)) == 1

    def test_queen_of_spades_worth_thirteen(self):
        assert card_points(Card(Suit.SPADES, 12)) == 13

    def test_other_spades_worth_zero(self):
        for rank in range(2, 15):
            if rank != 12:
                assert card_points(Card(Suit.SPADES, rank)) == 0

    def test_clubs_and_diamonds_worth_zero(self):
        for suit in [Suit.CLUBS, Suit.DIAMONDS]:
            for rank in range(2, 15):
                assert card_points(Card(suit, rank)) == 0


class TestHelpers:
    def test_is_heart(self):
        assert is_heart(Card(Suit.HEARTS, 5))
        assert not is_heart(Card(Suit.SPADES, 5))

    def test_is_queen_of_spades(self):
        assert is_queen_of_spades(Card(Suit.SPADES, 12))
        assert not is_queen_of_spades(Card(Suit.HEARTS, 12))
        assert not is_queen_of_spades(Card(Suit.SPADES, 13))


class TestDeck:
    def test_deck_has_52_cards(self):
        deck = Deck()
        assert len(deck.cards) == 52

    def test_all_cards_unique(self):
        deck = Deck()
        assert len(set(deck.cards)) == 52

    def test_all_suits_and_ranks(self):
        deck = Deck()
        cards = deck.cards
        for suit in Suit:
            suit_cards = [c for c in cards if c.suit == suit]
            assert len(suit_cards) == 13
            ranks = sorted(c.rank for c in suit_cards)
            assert ranks == list(range(2, 15))

    def test_deal_four_hands_of_thirteen(self):
        deck = Deck(rng=random.Random(42))
        hands = deck.deal()
        assert len(hands) == 4
        for hand in hands:
            assert len(hand) == 13

    def test_deal_no_duplicates(self):
        deck = Deck(rng=random.Random(42))
        hands = deck.deal()
        all_cards = set()
        for hand in hands:
            all_cards |= hand
        assert len(all_cards) == 52

    def test_deterministic_with_seed(self):
        hands1 = Deck(rng=random.Random(123)).deal()
        hands2 = Deck(rng=random.Random(123)).deal()
        assert hands1 == hands2

    def test_shuffle_changes_order(self):
        deck1 = Deck(rng=random.Random(1))
        deck2 = Deck(rng=random.Random(2))
        deck1.shuffle()
        deck2.shuffle()
        # Extremely unlikely to be the same with different seeds
        assert deck1.cards != deck2.cards

    def test_cards_property_returns_copy(self):
        deck = Deck()
        cards = deck.cards
        cards.pop()
        assert len(deck.cards) == 52
