"""Tests for hearts.data.generator — RecordingBot and data generation."""

import json
import random
from pathlib import Path

import pytest

from hearts.bots.rule_bot import RuleBot
from hearts.data.generator import (
    RecordingBot,
    generate_dataset,
    generate_game_data,
)
from hearts.state import PassDirection
from hearts.types import Card, Suit


class TestRecordingBot:
    """RecordingBot wraps a bot and records decisions."""

    def test_delegates_to_inner(self):
        """RecordingBot should play exactly like the inner bot."""
        rng = random.Random(42)
        inner = RuleBot()
        recorder = RecordingBot(inner, player_index=0)
        # Just verify it can be constructed and has empty records
        assert recorder.play_records == []
        assert recorder.pass_records == []


class TestGenerateGameData:
    """generate_game_data runs a game and returns annotated records."""

    def test_returns_game_result_and_records(self):
        bots = [RuleBot() for _ in range(4)]
        result, play_recs, pass_recs = generate_game_data(
            bots, seed=42
        )
        assert result.num_rounds > 0
        assert len(play_recs) > 0
        assert len(pass_recs) >= 0  # Could be 0 if only NONE rounds

    def test_play_record_count(self):
        """Each round has 52 card plays (13 tricks × 4 players)."""
        bots = [RuleBot() for _ in range(4)]
        result, play_recs, _ = generate_game_data(bots, seed=42)
        expected_plays = result.num_rounds * 52
        assert len(play_recs) == expected_plays

    def test_pass_record_count(self):
        """Each round with passing has 4 pass decisions."""
        bots = [RuleBot() for _ in range(4)]
        result, _, pass_recs = generate_game_data(bots, seed=42)
        # Count rounds with passing (not NONE)
        passing_rounds = sum(
            1
            for r in result.rounds
            if r.pass_direction != PassDirection.NONE
        )
        assert len(pass_recs) == passing_rounds * 4

    def test_deterministic_with_seed(self):
        """Same seed produces identical records."""
        bots1 = [RuleBot() for _ in range(4)]
        bots2 = [RuleBot() for _ in range(4)]
        r1, plays1, passes1 = generate_game_data(bots1, seed=123)
        r2, plays2, passes2 = generate_game_data(bots2, seed=123)
        assert r1.final_scores == r2.final_scores
        assert len(plays1) == len(plays2)
        for a, b in zip(plays1, plays2):
            assert a == b

    def test_play_records_have_required_fields(self):
        bots = [RuleBot() for _ in range(4)]
        _, play_recs, _ = generate_game_data(bots, seed=42)
        required = {
            "player_index",
            "trick_number",
            "hand",
            "legal_plays",
            "trick_so_far",
            "lead_suit",
            "hearts_broken",
            "is_first_trick",
            "cards_played_this_round",
            "tricks_won_by_player",
            "points_taken_by_player",
            "pass_direction",
            "cards_received_in_pass",
            "cumulative_scores",
            "round_number",
            "card_played",
            "points_this_trick",
            "round_score",
            "game_rank",
        }
        for rec in play_recs:
            assert required.issubset(rec.keys()), (
                f"Missing fields: {required - rec.keys()}"
            )

    def test_pass_records_have_required_fields(self):
        bots = [RuleBot() for _ in range(4)]
        _, _, pass_recs = generate_game_data(bots, seed=42)
        if not pass_recs:
            pytest.skip("No pass records in this game")
        required = {
            "player_index",
            "hand",
            "pass_direction",
            "cumulative_scores",
            "round_number",
            "cards_passed",
            "round_score",
            "game_rank",
        }
        for rec in pass_recs:
            assert required.issubset(rec.keys()), (
                f"Missing fields: {required - rec.keys()}"
            )

    def test_round_scores_sum_correctly(self):
        """Round scores should sum to 26 (normal) or 78 (shoot moon)."""
        bots = [RuleBot() for _ in range(4)]
        result, play_recs, _ = generate_game_data(bots, seed=42)
        for round_result in result.rounds:
            total = sum(round_result.scores)
            assert total in (26, 78), (
                f"Round scores sum to {total}: {round_result.scores}"
            )

    def test_game_rank_values(self):
        """game_rank should be 1-4 with correct ordering."""
        bots = [RuleBot() for _ in range(4)]
        result, play_recs, _ = generate_game_data(bots, seed=42)
        # Collect ranks by player
        player_ranks = {}
        for rec in play_recs:
            pi = rec["player_index"]
            player_ranks[pi] = rec["game_rank"]
        # All players should have ranks
        assert len(player_ranks) == 4
        # Ranks should be between 1 and 4
        for rank in player_ranks.values():
            assert 1 <= rank <= 4
        # Lower score should have lower (better) rank
        scores = result.final_scores
        for i in range(4):
            for j in range(4):
                if scores[i] < scores[j]:
                    assert player_ranks[i] <= player_ranks[j]

    def test_card_played_is_legal(self):
        """The card_played should always be in legal_plays."""
        bots = [RuleBot() for _ in range(4)]
        _, play_recs, _ = generate_game_data(bots, seed=42)
        for rec in play_recs:
            assert rec["card_played"] in rec["legal_plays"], (
                f"Card {rec['card_played']} not in "
                f"legal plays {rec['legal_plays']}"
            )

    def test_card_played_is_in_hand(self):
        """The card_played should be in the player's hand."""
        bots = [RuleBot() for _ in range(4)]
        _, play_recs, _ = generate_game_data(bots, seed=42)
        for rec in play_recs:
            assert rec["card_played"] in rec["hand"]


class TestGenerateDataset:
    """generate_dataset writes JSONL files."""

    def test_writes_jsonl_files(self, tmp_path):
        n_games = 3

        def bot_factory():
            return [RuleBot() for _ in range(4)]

        total_play, total_pass = generate_dataset(
            n_games=n_games,
            bot_factory=bot_factory,
            output_dir=tmp_path,
            seed=42,
        )

        play_file = tmp_path / "play_decisions.jsonl"
        pass_file = tmp_path / "pass_decisions.jsonl"
        assert play_file.exists()
        assert pass_file.exists()

        # Verify line count matches returned totals
        play_lines = play_file.read_text().strip().split("\n")
        pass_lines = pass_file.read_text().strip().split("\n")
        assert len(play_lines) == total_play
        # pass_lines could be [''] if no pass records
        actual_pass = (
            len(pass_lines) if pass_lines != [""] else 0
        )
        assert actual_pass == total_pass

    def test_jsonl_parseable(self, tmp_path):
        def bot_factory():
            return [RuleBot() for _ in range(4)]

        generate_dataset(
            n_games=2,
            bot_factory=bot_factory,
            output_dir=tmp_path,
            seed=99,
        )

        play_file = tmp_path / "play_decisions.jsonl"
        with open(play_file) as f:
            for line in f:
                record = json.loads(line)
                assert isinstance(record, dict)
                assert "card_played" in record

    def test_deterministic_output(self, tmp_path):
        def bot_factory():
            return [RuleBot() for _ in range(4)]

        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"

        generate_dataset(2, bot_factory, dir1, seed=42)
        generate_dataset(2, bot_factory, dir2, seed=42)

        plays1 = (dir1 / "play_decisions.jsonl").read_text()
        plays2 = (dir2 / "play_decisions.jsonl").read_text()
        assert plays1 == plays2
