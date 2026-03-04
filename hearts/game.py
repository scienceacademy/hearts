"""Game engine: runs multi-round Hearts games to 100 points."""

from __future__ import annotations

import random
from typing import Optional

from hearts.bot import Bot
from hearts.round import Round
from hearts.state import GameResult, GameState
from hearts.types import Deck


class Game:
    """Runs a full Hearts game (multiple rounds until score >= 100).

    Args:
        bots: 4 bot instances (one per seat)
        rng: optional Random instance for reproducibility
    """

    def __init__(
        self,
        bots: list[Bot],
        rng: Optional[random.Random] = None,
    ) -> None:
        if len(bots) != 4:
            raise ValueError("Exactly 4 bots required")
        self._bots = bots
        self._rng = rng
        self._state = GameState()

    def play(self) -> GameResult:
        """Play rounds until a player reaches 100 points."""
        while not self._state.game_over:
            deck = Deck(rng=self._rng)
            hands = deck.deal()

            round_num = len(self._state.round_history)
            direction = self._state.pass_direction

            rnd = Round(
                bots=self._bots,
                hands=hands,
                pass_direction=direction,
                cumulative_scores=list(
                    self._state.cumulative_scores
                ),
                round_number=round_num,
            )
            result = rnd.play()

            self._state.round_history.append(result)
            for i in range(4):
                self._state.cumulative_scores[i] += result.scores[i]

        scores = self._state.cumulative_scores
        winner = scores.index(min(scores))

        return GameResult(
            final_scores=list(scores),
            winner=winner,
            rounds=list(self._state.round_history),
            num_rounds=len(self._state.round_history),
        )
