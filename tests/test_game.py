"""Tests for hearts.game — multi-round game engine."""

import random

from hearts.bot import Bot
from hearts.game import Game
from hearts.state import PassDirection, PassView, PlayerView
from hearts.types import Card


class LowestCardBot(Bot):
    """Always plays the lowest legal card. Passes lowest 3."""

    def pass_cards(self, view: PassView) -> list[Card]:
        return sorted(view.hand)[:3]

    def play_card(self, view: PlayerView) -> Card:
        return sorted(view.legal_plays)[0]


class TestGameCompletion:
    def test_game_completes(self):
        rng = random.Random(42)
        bots = [LowestCardBot() for _ in range(4)]
        result = Game(bots, rng=rng).play()
        assert result.num_rounds > 0
        assert any(s >= 100 for s in result.final_scores)

    def test_winner_has_lowest_score(self):
        rng = random.Random(42)
        bots = [LowestCardBot() for _ in range(4)]
        result = Game(bots, rng=rng).play()
        assert result.final_scores[result.winner] == min(
            result.final_scores
        )

    def test_deterministic(self):
        results = []
        for _ in range(2):
            rng = random.Random(123)
            bots = [LowestCardBot() for _ in range(4)]
            results.append(Game(bots, rng=rng).play())
        assert results[0].final_scores == results[1].final_scores
        assert results[0].num_rounds == results[1].num_rounds


class TestPassDirectionRotation:
    def test_pass_direction_cycles(self):
        rng = random.Random(42)
        bots = [LowestCardBot() for _ in range(4)]
        result = Game(bots, rng=rng).play()

        expected = [
            PassDirection.LEFT,
            PassDirection.RIGHT,
            PassDirection.ACROSS,
            PassDirection.NONE,
        ]
        for i, rnd in enumerate(result.rounds):
            assert rnd.pass_direction == expected[i % 4], (
                f"Round {i}: expected {expected[i % 4]}, "
                f"got {rnd.pass_direction}"
            )


class TestGameScoring:
    def test_scores_accumulate(self):
        rng = random.Random(42)
        bots = [LowestCardBot() for _ in range(4)]
        result = Game(bots, rng=rng).play()

        # Verify final scores equal sum of round scores
        totals = [0, 0, 0, 0]
        for rnd in result.rounds:
            for i in range(4):
                totals[i] += rnd.scores[i]
        assert totals == result.final_scores

    def test_each_round_scores_sum_to_26_or_78(self):
        rng = random.Random(42)
        bots = [LowestCardBot() for _ in range(4)]
        result = Game(bots, rng=rng).play()
        for rnd in result.rounds:
            assert sum(rnd.scores) in (26, 78)

    def test_rounds_list_matches_count(self):
        rng = random.Random(42)
        bots = [LowestCardBot() for _ in range(4)]
        result = Game(bots, rng=rng).play()
        assert len(result.rounds) == result.num_rounds


class TestGameEdgeCases:
    def test_requires_four_bots(self):
        try:
            Game([LowestCardBot()] * 3)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_multiple_games_different_seeds(self):
        """Different seeds should generally produce different games."""
        results = []
        for seed in range(5):
            rng = random.Random(seed)
            bots = [LowestCardBot() for _ in range(4)]
            results.append(Game(bots, rng=rng).play())
        # At least some should differ in round count or scores
        scores = [tuple(r.final_scores) for r in results]
        assert len(set(scores)) > 1


class TestTieBreaking:
    def test_winner_with_tie_is_first_occurrence(self):
        """When tied, winner should be the lowest index player."""
        # Run many games to find a tie
        for seed in range(500):
            rng = random.Random(seed)
            bots = [LowestCardBot() for _ in range(4)]
            result = Game(bots, rng=rng).play()
            min_score = min(result.final_scores)
            tied = [
                i for i, s in enumerate(result.final_scores)
                if s == min_score
            ]
            if len(tied) > 1:
                # Winner should be the first tied player
                assert result.winner == tied[0]
                return
        # If no tie found in 500 games, that's fine
