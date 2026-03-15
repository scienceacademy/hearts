"""Tests for hearts.tournament — tournament runner."""

import random

from hearts.bots.random_bot import RandomBot
from hearts.bots.rule_bot import RuleBot
from hearts.tournament import Tournament


class TestTournamentBasic:
    def test_runs_correct_number_of_games(self):
        t = Tournament(
            bot_factories={
                "Rule": RuleBot,
                "Random": lambda: RandomBot(
                    rng=random.Random()
                ),
            },
            num_games=10,
            rng=random.Random(42),
        )
        result = t.run(parallel=False)
        assert result.num_games == 10

    def test_all_bots_have_stats(self):
        t = Tournament(
            bot_factories={
                "Rule": RuleBot,
                "Random": lambda: RandomBot(
                    rng=random.Random()
                ),
            },
            num_games=10,
            rng=random.Random(42),
        )
        result = t.run(parallel=False)
        assert "Rule" in result.stats
        assert "Random" in result.stats

    def test_total_wins_equal_games(self):
        t = Tournament(
            bot_factories={
                "Rule": RuleBot,
                "Random": lambda: RandomBot(
                    rng=random.Random()
                ),
            },
            num_games=20,
            rng=random.Random(42),
        )
        result = t.run(parallel=False)
        total_wins = sum(s.wins for s in result.stats.values())
        assert total_wins == 20

    def test_games_per_bot_consistent(self):
        t = Tournament(
            bot_factories={
                "Rule": RuleBot,
                "Random": lambda: RandomBot(
                    rng=random.Random()
                ),
            },
            num_games=20,
            rng=random.Random(42),
        )
        result = t.run(parallel=False)
        # Each bot plays in every game, occupying 2 seats
        for s in result.stats.values():
            assert s.games == 40  # 20 games * 2 seats each


class TestTournamentDeterministic:
    def test_same_seed_same_results(self):
        results = []
        for _ in range(2):
            t = Tournament(
                bot_factories={
                    "Rule": RuleBot,
                    "Random": lambda: RandomBot(
                        rng=random.Random(0)
                    ),
                },
                num_games=10,
                rng=random.Random(42),
            )
            results.append(t.run(parallel=False))
        for name in results[0].stats:
            assert (
                results[0].stats[name].wins
                == results[1].stats[name].wins
            )


class TestTournamentRotation:
    def test_seat_rotation(self):
        """Bots should occupy different seats across games."""
        # With 2 bot types and 4 seats, rotation shifts
        # the seating each game
        t = Tournament(
            bot_factories={
                "A": RuleBot,
                "B": lambda: RandomBot(
                    rng=random.Random()
                ),
            },
            num_games=4,
            rng=random.Random(42),
        )
        result = t.run(parallel=False)
        # Both bots should have played games
        for s in result.stats.values():
            assert s.games > 0


class TestTournamentSummary:
    def test_summary_output(self):
        t = Tournament(
            bot_factories={
                "Rule": RuleBot,
                "Random": lambda: RandomBot(
                    rng=random.Random()
                ),
            },
            num_games=5,
            rng=random.Random(42),
        )
        result = t.run(parallel=False)
        summary = result.summary()
        assert "Rule" in summary
        assert "Random" in summary
        assert "Win%" in summary


class TestTournamentStats:
    def test_win_rate(self):
        t = Tournament(
            bot_factories={
                "Rule": RuleBot,
                "Random": lambda: RandomBot(
                    rng=random.Random()
                ),
            },
            num_games=10,
            rng=random.Random(42),
        )
        result = t.run(parallel=False)
        for s in result.stats.values():
            assert 0.0 <= s.win_rate <= 1.0

    def test_avg_score(self):
        t = Tournament(
            bot_factories={
                "Rule": RuleBot,
                "Random": lambda: RandomBot(
                    rng=random.Random()
                ),
            },
            num_games=10,
            rng=random.Random(42),
        )
        result = t.run(parallel=False)
        for s in result.stats.values():
            assert s.avg_score > 0

    def test_scores_list_populated(self):
        t = Tournament(
            bot_factories={
                "Rule": RuleBot,
                "Random": lambda: RandomBot(
                    rng=random.Random()
                ),
            },
            num_games=10,
            rng=random.Random(42),
        )
        result = t.run(parallel=False)
        for s in result.stats.values():
            assert len(s.scores) == s.games


class TestTournamentEdgeCases:
    def test_requires_at_least_two_bot_types(self):
        try:
            Tournament(
                bot_factories={"Solo": RuleBot},
                num_games=1,
            )
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_rule_bot_beats_random_in_tournament(self):
        """RuleBot should have a higher win rate than RandomBot."""
        t = Tournament(
            bot_factories={
                "Rule": RuleBot,
                "Random": lambda: RandomBot(
                    rng=random.Random()
                ),
            },
            num_games=100,
            rng=random.Random(42),
        )
        result = t.run(parallel=False)
        assert (
            result.stats["Rule"].win_rate
            > result.stats["Random"].win_rate
        )
