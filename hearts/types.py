"""Core data types for the Hearts card game."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class Suit(IntEnum):
    """Card suits, ordered for display purposes."""

    CLUBS = 0
    DIAMONDS = 1
    SPADES = 2
    HEARTS = 3

    @property
    def symbol(self) -> str:
        return _SUIT_SYMBOLS[self]


_SUIT_SYMBOLS = {
    Suit.CLUBS: "\u2663",
    Suit.DIAMONDS: "\u2666",
    Suit.SPADES: "\u2660",
    Suit.HEARTS: "\u2665",
}

_RANK_NAMES = {11: "J", 12: "Q", 13: "K", 14: "A"}


@dataclass(frozen=True, order=True)
class Card:
    """A playing card with suit and rank.

    Ranks: 2-14, where 11=J, 12=Q, 13=K, 14=A.
    """

    suit: Suit
    rank: int

    def __post_init__(self) -> None:
        if not (2 <= self.rank <= 14):
            raise ValueError(f"Rank must be 2-14, got {self.rank}")

    def __str__(self) -> str:
        rank_str = _RANK_NAMES.get(self.rank, str(self.rank))
        return f"{rank_str}{self.suit.symbol}"

    def __repr__(self) -> str:
        return f"Card({self.suit.name}, {self.rank})"

    def __hash__(self) -> int:
        return hash((self.suit, self.rank))


def card_points(card: Card) -> int:
    """Return the point value of a card (hearts=1, Q♠=13, else 0)."""
    if card.suit == Suit.HEARTS:
        return 1
    if card.suit == Suit.SPADES and card.rank == 12:
        return 13
    return 0


def is_heart(card: Card) -> bool:
    """Return True if the card is a heart."""
    return card.suit == Suit.HEARTS


def is_queen_of_spades(card: Card) -> bool:
    """Return True if the card is the Queen of Spades."""
    return card.suit == Suit.SPADES and card.rank == 12


class Deck:
    """A standard 52-card deck that can be shuffled and dealt."""

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self._cards = [
            Card(suit, rank)
            for suit in Suit
            for rank in range(2, 15)
        ]
        self._rng = rng

    @property
    def cards(self) -> list[Card]:
        return list(self._cards)

    def shuffle(self) -> None:
        """Shuffle the deck in place."""
        if self._rng is not None:
            self._rng.shuffle(self._cards)
        else:
            random.shuffle(self._cards)

    def deal(self) -> list[set[Card]]:
        """Deal all 52 cards into 4 hands of 13. Shuffles first."""
        self.shuffle()
        return [
            set(self._cards[i * 13:(i + 1) * 13])
            for i in range(4)
        ]
