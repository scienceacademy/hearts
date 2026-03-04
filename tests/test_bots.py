"""Tests for RandomBot and RuleBot — legality and relative strength."""

import random

from hearts.bot import Bot
from hearts.bots.random_bot import RandomBot
from hearts.bots.rule_bot import RuleBot
from hearts.game import Game
from hearts.round import Round
from hearts.state import PassDirection
from hearts.types import Deck


class TestRandomBot:
    def test_always_plays_legal(self):
        """RandomBot plays legal moves across many games."""
        for seed in range(50):
            rng = random.Random(seed)
            bots: list[Bot] = [
                RandomBot(rng=random.Random(seed + i))
                for i in range(4)
            ]
            game = Game(bots, rng=rng)
            game.play()  # Would raise if illegal move

    def test_passes_three_cards(self):
        """RandomBot passes exactly 3 cards from hand."""
        rng = random.Random(42)
        hands = Deck(rng=rng).deal()
        bot = RandomBot(rng=random.Random(42))
        from hearts.state import PassView
        view = PassView(
            hand=set(hands[0]),
            player_index=0,
            pass_direction=PassDirection.LEFT,
            cumulative_scores=[0, 0, 0, 0],
            round_number=0,
        )
        passed = bot.pass_cards(view)
        assert len(passed) == 3
        assert len(set(passed)) == 3
        assert all(c in hands[0] for c in passed)

    def test_deterministic_with_seed(self):
        results = []
        for _ in range(2):
            rng = random.Random(99)
            bots: list[Bot] = [
                RandomBot(rng=random.Random(99 + i))
                for i in range(4)
            ]
            results.append(Game(bots, rng=rng).play())
        assert results[0].final_scores == results[1].final_scores


class TestRuleBot:
    def test_always_plays_legal(self):
        """RuleBot plays legal moves across many games."""
        for seed in range(50):
            rng = random.Random(seed)
            bots: list[Bot] = [RuleBot() for _ in range(4)]
            game = Game(bots, rng=rng)
            game.play()

    def test_passes_three_cards(self):
        rng = random.Random(42)
        hands = Deck(rng=rng).deal()
        bot = RuleBot()
        from hearts.state import PassView
        view = PassView(
            hand=set(hands[0]),
            player_index=0,
            pass_direction=PassDirection.LEFT,
            cumulative_scores=[0, 0, 0, 0],
            round_number=0,
        )
        passed = bot.pass_cards(view)
        assert len(passed) == 3
        assert len(set(passed)) == 3
        assert all(c in hands[0] for c in passed)

    def test_passes_queen_of_spades_when_held(self):
        """RuleBot should prioritise passing Q♠."""
        from hearts.types import Card, Suit
        from hearts.state import PassView
        # Construct a hand containing Q♠
        hand = {
            Card(Suit.CLUBS, r) for r in range(2, 12)
        }
        hand.add(Card(Suit.SPADES, 12))  # Q♠
        hand.add(Card(Suit.DIAMONDS, 2))
        hand.add(Card(Suit.DIAMONDS, 3))
        bot = RuleBot()
        view = PassView(
            hand=hand,
            player_index=0,
            pass_direction=PassDirection.LEFT,
            cumulative_scores=[0, 0, 0, 0],
            round_number=0,
        )
        passed = bot.pass_cards(view)
        assert Card(Suit.SPADES, 12) in passed


class TestRuleBotBeatsRandom:
    def test_rule_bot_wins_more(self):
        """RuleBot should beat RandomBot in a majority of games."""
        rule_wins = 0
        random_wins = 0
        num_games = 100

        for seed in range(num_games):
            rng = random.Random(seed)
            bots: list[Bot] = [
                RuleBot(),
                RandomBot(rng=random.Random(seed + 100)),
                RuleBot(),
                RandomBot(rng=random.Random(seed + 200)),
            ]
            result = Game(bots, rng=rng).play()
            # Check who won — seats 0,2 are RuleBot, 1,3 are Random
            winner = result.winner
            if winner in (0, 2):
                rule_wins += 1
            else:
                random_wins += 1

        # RuleBot should win significantly more often
        assert rule_wins > random_wins, (
            f"RuleBot won {rule_wins}, RandomBot won {random_wins}"
        )
