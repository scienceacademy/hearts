"""Game data generation for ML training.

Provides RecordingBot (a wrapper that records decisions) and functions
to generate annotated datasets of play and pass decisions.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Optional

from hearts.bot import Bot
from hearts.data.schema import (
    card_to_index,
    serialize_pass_view,
    serialize_player_view,
)
from hearts.game import Game
from hearts.state import (
    GameResult,
    PassView,
    PlayerView,
    RoundResult,
    TrickResult,
)
from hearts.types import Card, card_points


class RecordingBot(Bot):
    """Wraps a Bot to record its pass and play decisions.

    Records the serialized view and chosen action for each decision,
    storing them in memory for later annotation with outcome data.

    Args:
        inner: the bot instance to delegate decisions to
        player_index: which seat this bot occupies (set by generator)
    """

    def __init__(self, inner: Bot, player_index: int = 0) -> None:
        self._inner = inner
        self.player_index = player_index
        self.play_records: list[dict] = []
        self.pass_records: list[dict] = []
        self._current_round_play_records: list[dict] = []
        self._current_round_number: int = 0

    def pass_cards(self, view: PassView) -> list[Card]:
        """Record the pass view and chosen cards, then delegate."""
        cards = self._inner.pass_cards(view)
        record = serialize_pass_view(view)
        record["cards_passed"] = [card_to_index(c) for c in cards]
        self.pass_records.append(record)
        self._current_round_number = view.round_number
        return cards

    def play_card(self, view: PlayerView) -> Card:
        """Record the play view and chosen card, then delegate."""
        card = self._inner.play_card(view)
        record = serialize_player_view(view)
        record["round_number"] = self._current_round_number
        record["card_played"] = card_to_index(card)
        self.play_records.append(record)
        self._current_round_play_records.append(record)
        return card

    def on_trick_complete(self, result: TrickResult) -> None:
        """Forward to inner bot."""
        self._inner.on_trick_complete(result)

    def on_round_complete(self, result: RoundResult) -> None:
        """Annotate play records for this round with round scores."""
        self._inner.on_round_complete(result)

        # Annotate each play record with points earned in its trick
        trick_points: dict[int, int] = {}
        for trick_result in result.tricks:
            trick_points[trick_result.trick_number] = (
                trick_result.points
                if trick_result.winner == self.player_index
                else 0
            )

        for record in self._current_round_play_records:
            trick_num = record["trick_number"]
            record["points_this_trick"] = trick_points.get(
                trick_num, 0
            )
            record["round_score"] = result.scores[self.player_index]

        # Annotate pass records for this round
        for rec in self.pass_records:
            if (
                rec.get("round_score") is None
                and rec["round_number"] == self._current_round_number
            ):
                rec["round_score"] = result.scores[self.player_index]

        self._current_round_play_records = []


def generate_game_data(
    bots: list[Bot],
    seed: Optional[int] = None,
) -> tuple[GameResult, list[dict], list[dict]]:
    """Run one game with recording bots and return annotated records.

    Args:
        bots: 4 bot instances to play the game
        seed: optional seed for reproducibility

    Returns:
        Tuple of (GameResult, play_records, pass_records) where records
        are annotated with outcome data (points_this_trick, round_score,
        game_rank).
    """
    rng = random.Random(seed) if seed is not None else None

    recorders = [
        RecordingBot(bot, player_index=i)
        for i, bot in enumerate(bots)
    ]

    game = Game(recorders, rng=rng)
    result = game.play()

    # Compute game rank for each player (1 = best/lowest score)
    ranks = _compute_ranks(result.final_scores)

    # Collect and annotate all records
    play_records: list[dict] = []
    pass_records: list[dict] = []

    for i, recorder in enumerate(recorders):
        for record in recorder.play_records:
            record["game_rank"] = ranks[i]
            play_records.append(record)

        for record in recorder.pass_records:
            record["game_rank"] = ranks[i]
            pass_records.append(record)

    return result, play_records, pass_records


def _compute_ranks(scores: list[int]) -> list[int]:
    """Compute game ranks (1=best) from final scores.

    Lower score is better. Ties share the same rank.
    """
    sorted_scores = sorted(set(scores))
    score_to_rank = {s: i + 1 for i, s in enumerate(sorted_scores)}
    return [score_to_rank[s] for s in scores]


def generate_dataset(
    n_games: int,
    bot_factory: Callable[[], list[Bot]],
    output_dir: str | Path,
    seed: Optional[int] = None,
) -> tuple[int, int]:
    """Generate a dataset of game decisions and write to JSONL files.

    Args:
        n_games: number of games to simulate
        bot_factory: callable returning a list of 4 bots
        output_dir: directory to write output files
        seed: optional seed for reproducibility

    Returns:
        Tuple of (total_play_records, total_pass_records).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    play_file = output_path / "play_decisions.jsonl"
    pass_file = output_path / "pass_decisions.jsonl"

    total_play = 0
    total_pass = 0

    with (
        open(play_file, "w") as pf,
        open(pass_file, "w") as sf,
    ):
        for game_num in range(n_games):
            game_seed = seed + game_num if seed is not None else None
            bots = bot_factory()

            _, play_records, pass_records = generate_game_data(
                bots, seed=game_seed
            )

            for record in play_records:
                pf.write(json.dumps(record) + "\n")
            for record in pass_records:
                sf.write(json.dumps(record) + "\n")

            total_play += len(play_records)
            total_pass += len(pass_records)

            if (game_num + 1) % 1000 == 0 or game_num + 1 == n_games:
                print(
                    f"  Games: {game_num + 1}/{n_games} "
                    f"({total_play} play, {total_pass} pass records)"
                )

    return total_play, total_pass
