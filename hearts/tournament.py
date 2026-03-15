"""Tournament runner: evaluates bots across many games."""

from __future__ import annotations

import multiprocessing
import pickle
import random
from concurrent.futures import ProcessPoolExecutor
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

    def _build_seat_names(
        self, game_idx: int
    ) -> list[str]:
        """Build rotated seat assignment for a game."""
        names = list(self._factories.keys())
        seat_names = [
            names[i % len(names)] for i in range(4)
        ]
        rotation = game_idx % 4
        return seat_names[rotation:] + seat_names[:rotation]

    def _play_one_game(
        self, game_idx: int, seed: int
    ) -> tuple[int, list[str], list[int], int]:
        """Play a single game and return results.

        Returns:
            Tuple of (game_idx, seat_names, final_scores, winner).
        """
        seat_names = self._build_seat_names(game_idx)
        bots: list[Bot] = [
            self._factories[n]() for n in seat_names
        ]
        game_rng = random.Random(seed)
        result = Game(bots, rng=game_rng).play()
        return (
            game_idx,
            seat_names,
            result.final_scores,
            result.winner,
        )

    def run(
        self,
        progress: Optional[Callable[[int, int], None]] = None,
        parallel: bool = False,
    ) -> TournamentResult:
        """Run the tournament and return results.

        Args:
            progress: optional callback(current, total) for progress
            parallel: if True, run games in parallel using
                multiple processes (default False)
        """
        names = list(self._factories.keys())
        stats = {
            name: BotStats(name=name) for name in names
        }

        # Pre-generate deterministic seeds for each game
        seeds = [
            self._rng.randint(0, 2**63)
            for _ in range(self._num_games)
        ]

        if parallel:
            try:
                self._run_parallel(stats, seeds, progress)
            except (pickle.PicklingError, AttributeError, TypeError):
                self._run_sequential(stats, seeds, progress)
        else:
            self._run_sequential(stats, seeds, progress)

        return TournamentResult(
            stats=stats, num_games=self._num_games
        )

    def _run_sequential(
        self,
        stats: dict[str, BotStats],
        seeds: list[int],
        progress: Optional[Callable[[int, int], None]],
    ) -> None:
        """Run games sequentially."""
        for game_idx in range(self._num_games):
            _, seat_names, final_scores, winner = (
                self._play_one_game(game_idx, seeds[game_idx])
            )
            self._record_result(
                stats, seat_names, final_scores, winner
            )
            if progress is not None and (
                (game_idx + 1) % 10 == 0
                or game_idx + 1 == self._num_games
            ):
                progress(game_idx + 1, self._num_games)

    def _run_parallel(
        self,
        stats: dict[str, BotStats],
        seeds: list[int],
        progress: Optional[Callable[[int, int], None]],
    ) -> None:
        """Run games in parallel using ProcessPoolExecutor."""
        completed = 0
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(mp_context=ctx) as executor:
            futures = [
                executor.submit(
                    self._play_one_game, game_idx, seeds[game_idx]
                )
                for game_idx in range(self._num_games)
            ]
            for future in futures:
                _, seat_names, final_scores, winner = (
                    future.result()
                )
                completed += 1
                self._record_result(
                    stats, seat_names, final_scores, winner
                )
                if progress is not None and (
                    completed % 10 == 0
                    or completed == self._num_games
                ):
                    progress(completed, self._num_games)

    @staticmethod
    def _record_result(
        stats: dict[str, BotStats],
        seat_names: list[str],
        final_scores: list[int],
        winner: int,
    ) -> None:
        """Record a single game result into stats."""
        winner_name = seat_names[winner]
        for seat, name in enumerate(seat_names):
            stats[name].games += 1
            stats[name].total_score += final_scores[seat]
            stats[name].scores.append(final_scores[seat])
        stats[winner_name].wins += 1
