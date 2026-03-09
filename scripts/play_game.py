"""Run a single Hearts game with four specified bots.

Usage:
    python scripts/play_game.py RuleBot RandomBot RuleBot RandomBot
    python scripts/play_game.py --seed 42 RuleBot RuleBot RuleBot RuleBot
    python scripts/play_game.py --verbose RuleBot RandomBot RuleBot RandomBot
    python scripts/play_game.py --list

Bot names can be repeated. Any Bot subclass in hearts/bots/ is
automatically discovered.
"""

import argparse
import importlib
import inspect
import os
import pkgutil
import random
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import hearts.bots  # noqa: E402
from hearts.bot import Bot
from hearts.game import Game


def discover_bots() -> dict[str, type[Bot]]:
    """Find all Bot subclasses in the hearts.bots package."""
    registry: dict[str, type[Bot]] = {}
    for info in pkgutil.iter_modules(hearts.bots.__path__):
        module = importlib.import_module(
            f"hearts.bots.{info.name}"
        )
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Bot) and obj is not Bot:
                registry[obj.__name__.lower()] = obj
    return registry


def make_bot(
    name: str,
    registry: dict[str, type[Bot]],
    rng: random.Random,
) -> Bot:
    """Create a bot instance by name (case-insensitive)."""
    key = name.lower()
    if key not in registry:
        available = ", ".join(
            cls.__name__ for cls in registry.values()
        )
        print(f"Unknown bot: {name}")
        print(f"Available bots: {available}")
        sys.exit(1)
    cls = registry[key]
    # Pass rng if the constructor accepts one
    sig = inspect.signature(cls.__init__)
    if "rng" in sig.parameters:
        return cls(rng=random.Random(rng.randint(0, 2**63)))
    return cls()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single Hearts game with four bots",
    )
    parser.add_argument(
        "bots", nargs="*", metavar="BOT",
        help="Bot name (case-insensitive)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show per-round details",
    )
    parser.add_argument(
        "--list", "-l", action="store_true",
        dest="list_bots",
        help="List available bots and exit",
    )
    args = parser.parse_args()

    registry = discover_bots()

    if args.list_bots:
        print("Available bots:")
        for cls in registry.values():
            doc = (cls.__doc__ or "").strip().split("\n")[0]
            print(f"  {cls.__name__:<16} {doc}")
        return

    if len(args.bots) != 4:
        parser.error("exactly 4 bot names are required")

    rng = random.Random(args.seed)
    bot_names = args.bots
    bots = [make_bot(name, registry, rng) for name in bot_names]

    result = Game(bots, rng=rng).play()

    if args.verbose:
        for i, rnd in enumerate(result.rounds):
            direction = rnd.pass_direction.value
            scores_str = "  ".join(
                f"P{j}:{s:>3}"
                for j, s in enumerate(rnd.scores)
            )
            moon = (
                " (shoot the moon!)"
                if rnd.shoot_the_moon else ""
            )
            print(
                f"Round {i + 1:>2} ({direction:<6}): "
                f"{scores_str}{moon}"
            )
        print()

    print(f"{'Seat':<6} {'Bot':<16} {'Score':>5}")
    print("-" * 30)
    for i in range(4):
        marker = " *" if i == result.winner else ""
        print(
            f"  {i:<4} {bot_names[i]:<16} "
            f"{result.final_scores[i]:>5}{marker}"
        )
    print()
    print(
        f"Winner: Seat {result.winner} "
        f"({bot_names[result.winner]}) "
        f"with {result.final_scores[result.winner]} points "
        f"after {result.num_rounds} rounds"
    )


if __name__ == "__main__":
    main()
