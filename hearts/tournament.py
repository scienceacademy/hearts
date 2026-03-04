"""Tournament runner: evaluates bots across many games."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from hearts.bot import Bot
from hearts.game import Game


@dataclass
class BotStats:
    """Accumulated stats for one bot type."""

    name: str
    wins: int = 0
    games: int = 0
    total_score: int = 0
    scores: list[int] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def avg_score(self) -> float:
        return self.total_score / self.games if self.games else 0.0


@dataclass
class TournamentResult:
    """Results of a tournament."""

    stats: dict[str, BotStats]
    num_games: int

    def summary(self) -> str:
        """Return a formatted summary table."""
        lines = [
            f"{'Bot':<20} {'Wins':>6} {'Games':>6} "
            f"{'Win%':>7} {'AvgScore':>9}",
            "-" * 52,
        ]
        for s in sorted(
            self.stats.values(),
            key=lambda s: s.win_rate,
            reverse=True,
        ):
            lines.append(
                f"{s.name:<20} {s.wins:>6} {s.games:>6} "
                f"{s.win_rate:>6.1%} {s.avg_score:>9.1f}"
            )
        return "\n".join(lines)


class Tournament:
    """Runs games between bot factories, rotating seats.

    Args:
        bot_factories: dict mapping bot name to a callable
            that creates a Bot instance.
        num_games: number of games to play.
        rng: optional Random instance for reproducibility.
    """

    def __init__(
        self,
        bot_factories: dict[str, Callable[[], Bot]],
        num_games: int = 100,
        rng: Optional[random.Random] = None,
    ) -> None:
        if len(bot_factories) < 2:
            raise ValueError("Need at least 2 bot types")
        self._factories = bot_factories
        self._num_games = num_games
        self._rng = rng or random.Random()

    def run(self) -> TournamentResult:
        """Run the tournament and return results."""
        names = list(self._factories.keys())
        stats = {
            name: BotStats(name=name) for name in names
        }

        for game_idx in range(self._num_games):
            # Build a 4-player roster by cycling through bots
            # and rotating seats each game
            seat_names = [
                names[i % len(names)] for i in range(4)
            ]
            # Rotate based on game index
            rotation = game_idx % 4
            seat_names = (
                seat_names[rotation:]
                + seat_names[:rotation]
            )

            bots: list[Bot] = [
                self._factories[n]() for n in seat_names
            ]

            game_rng = random.Random(
                self._rng.randint(0, 2**63)
            )
            result = Game(bots, rng=game_rng).play()

            winner_name = seat_names[result.winner]
            for seat, name in enumerate(seat_names):
                stats[name].games += 1
                score = result.final_scores[seat]
                stats[name].total_score += score
                stats[name].scores.append(score)
            stats[winner_name].wins += 1

        return TournamentResult(
            stats=stats, num_games=self._num_games
        )
