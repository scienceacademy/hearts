"""A bot that plays randomly — baseline opponent."""

from __future__ import annotations

import random
from typing import Optional

from hearts.bot import Bot
from hearts.state import PassView, PlayerView
from hearts.types import Card


class RandomBot(Bot):
    """Plays a random legal card and passes 3 random cards.

    Args:
        rng: optional Random instance for reproducibility.
    """

    def __init__(
        self, rng: Optional[random.Random] = None
    ) -> None:
        self._rng = rng or random.Random()

    def pass_cards(self, view: PassView) -> list[Card]:
        """Pass 3 random cards from hand."""
        hand = sorted(view.hand)
        return self._rng.sample(hand, 3)

    def play_card(self, view: PlayerView) -> Card:
        """Play a random legal card."""
        return self._rng.choice(view.legal_plays)
