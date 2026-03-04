"""Abstract base class for Hearts bots."""

from __future__ import annotations

from abc import ABC, abstractmethod

from hearts.state import PassView, PlayerView, RoundResult, TrickResult
from hearts.types import Card


class Bot(ABC):
    """Base class for Hearts bots."""

    @abstractmethod
    def pass_cards(self, view: PassView) -> list[Card]:
        """Select 3 cards to pass."""

    @abstractmethod
    def play_card(self, view: PlayerView) -> Card:
        """Select a card to play."""

    def on_trick_complete(self, result: TrickResult) -> None:
        """Optional callback — observe completed trick."""
        _ = result

    def on_round_complete(self, result: RoundResult) -> None:
        """Optional callback — observe completed round."""
        _ = result
