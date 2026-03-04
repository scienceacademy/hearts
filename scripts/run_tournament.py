"""Run a tournament between RuleBot and RandomBot."""

import argparse
import random

from hearts.bots.random_bot import RandomBot
from hearts.bots.rule_bot import RuleBot
from hearts.tournament import Tournament


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Hearts tournament"
    )
    parser.add_argument(
        "--games", type=int, default=100,
        help="Number of games to play (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    t = Tournament(
        bot_factories={
            "RuleBot": RuleBot,
            "RandomBot": lambda: RandomBot(
                rng=random.Random()
            ),
        },
        num_games=args.games,
        rng=rng,
    )
    result = t.run()
    print(result.summary())


if __name__ == "__main__":
    main()
