"""Game state types for Hearts: tricks, rounds, and full games."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from hearts.types import Card, Suit, card_points


class PassDirection(Enum):
    """Direction cards are passed at the start of a round."""

    LEFT = "left"
    RIGHT = "right"
    ACROSS = "across"
    NONE = "none"


PASS_ROTATION = [
    PassDirection.LEFT,
    PassDirection.RIGHT,
    PassDirection.ACROSS,
    PassDirection.NONE,
]


@dataclass
class TrickState:
    """State of a trick in progress."""

    cards_played: list[tuple[int, Card]] = field(
        default_factory=list
    )
    lead_suit: Suit | None = None

    def winner(self) -> int:
        """Return the player index who won this trick.

        The winner is the player who played the highest card
        of the lead suit.
        """
        if not self.cards_played or self.lead_suit is None:
            raise ValueError("Cannot determine winner of empty trick")
        best_player = -1
        best_rank = -1
        for player, card in self.cards_played:
            if card.suit == self.lead_suit and card.rank > best_rank:
                best_rank = card.rank
                best_player = player
        return best_player


@dataclass
class TrickResult:
    """Summary of a completed trick."""

    cards_played: list[tuple[int, Card]]
    lead_suit: Suit
    winner: int
    points: int
    trick_number: int


@dataclass
class RoundState:
    """State of a single 13-trick round."""

    hands: list[set[Card]]
    current_trick: TrickState = field(default_factory=TrickState)
    cards_taken: list[list[Card]] = field(
        default_factory=lambda: [[] for _ in range(4)]
    )
    hearts_broken: bool = False
    lead_player: int = 0

    @property
    def scores(self) -> list[int]:
        """Per-round point totals computed from cards taken."""
        raw = [
            sum(card_points(c) for c in taken)
            for taken in self.cards_taken
        ]
        # Shoot the moon: one player took all 26 points
        for i, s in enumerate(raw):
            if s == 26:
                return [
                    0 if j == i else 26
                    for j in range(4)
                ]
        return raw


@dataclass
class RoundResult:
    """Summary of a completed round."""

    scores: list[int]
    tricks: list[TrickResult]
    shoot_the_moon: bool
    moon_shooter: int | None
    pass_direction: PassDirection
    hands_dealt: list[set[Card]]


@dataclass
class PassView:
    """What bots see during the passing phase."""

    hand: set[Card]
    player_index: int
    pass_direction: PassDirection
    cumulative_scores: list[int]
    round_number: int


@dataclass
class PlayerView:
    """What bots see when choosing a card to play."""

    hand: set[Card]
    player_index: int
    trick_so_far: list[tuple[int, Card]]
    lead_suit: Suit | None
    hearts_broken: bool
    is_first_trick: bool
    cards_played_this_round: list[tuple[int, int, Card]]
    tricks_won_by_player: list[int]
    points_taken_by_player: list[int]
    pass_direction: PassDirection
    cards_received_in_pass: list[Card]
    legal_plays: list[Card]


@dataclass
class GameState:
    """State of a multi-round game."""

    cumulative_scores: list[int] = field(
        default_factory=lambda: [0, 0, 0, 0]
    )
    round_history: list[RoundResult] = field(
        default_factory=list
    )
    current_round: RoundState | None = None

    @property
    def game_over(self) -> bool:
        """Game ends when any player reaches 100 points."""
        return any(s >= 100 for s in self.cumulative_scores)

    @property
    def pass_direction(self) -> PassDirection:
        """Pass direction for the next/current round."""
        idx = len(self.round_history) % len(PASS_ROTATION)
        return PASS_ROTATION[idx]


@dataclass
class GameResult:
    """Final result of a completed game."""

    final_scores: list[int]
    winner: int
    rounds: list[RoundResult]
    num_rounds: int
