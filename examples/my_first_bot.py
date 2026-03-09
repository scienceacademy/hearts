"""Example: a minimal custom Hearts bot.

This bot uses a simple strategy:
- Pass the three highest-ranked cards
- Always play the lowest legal card

To test it, run a tournament against the built-in bots:

    python examples/my_first_bot.py
"""

import os
import random
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from hearts.bot import Bot  # noqa: E402
from hearts.bots.random_bot import RandomBot
from hearts.bots.rule_bot import RuleBot
from hearts.state import PassView, PlayerView
from hearts.tournament import Tournament
from hearts.types import Card


class MyFirstBot(Bot):
    """A simple bot that always plays low and passes high."""

    def pass_cards(self, view: PassView) -> list[Card]:
        # Pass the three highest cards
        return sorted(view.hand, key=lambda c: c.rank)[-3:]

    def play_card(self, view: PlayerView) -> Card:
        # Play the lowest legal card
        return min(view.legal_plays, key=lambda c: c.rank)


def main() -> None:
    t = Tournament(
        bot_factories={
            "MyFirstBot": MyFirstBot,
            "RuleBot": RuleBot,
            "RandomBot": lambda: RandomBot(
                rng=random.Random()
            ),
        },
        num_games=100,
        rng=random.Random(42),
    )
    result = t.run()
    print(result.summary())


if __name__ == "__main__":
    main()
