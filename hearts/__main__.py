"""CLI entry point: python -m hearts."""

import random

from hearts.bots.random_bot import RandomBot
from hearts.bots.rule_bot import RuleBot
from hearts.tournament import Tournament


def main() -> None:
    """Run a default tournament: RuleBot vs RandomBot."""
    rng = random.Random(42)
    t = Tournament(
        bot_factories={
            "RuleBot": RuleBot,
            "RandomBot": lambda: RandomBot(rng=random.Random()),
        },
        num_games=100,
        rng=rng,
    )
    result = t.run()
    print(result.summary())


if __name__ == "__main__":
    main()
