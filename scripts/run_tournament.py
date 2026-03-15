"""Run a tournament between RuleBot and RandomBot."""

import argparse
import os
import random
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

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
    parser.add_argument(
        "--parallel", action="store_true", default=False,
        help="Run games in parallel (default: sequential)",
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
    mode = "parallel" if args.parallel else "sequential"
    print(
        f"Running {args.games} games ({mode} mode)..."
    )
    result = t.run(parallel=args.parallel)
    print(result.summary())


if __name__ == "__main__":
    main()
