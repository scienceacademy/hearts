"""Play a Hearts game as a human against 3 bots.

Usage:
    python scripts/play_interactive.py
    python scripts/play_interactive.py RuleBot RuleBot RuleBot
    python scripts/play_interactive.py --seat 2 RuleBot RandomBot RuleBot
    python scripts/play_interactive.py MLBot MLBot MLBot --model models/model.pkl
    python scripts/play_interactive.py --list

Specify 3 bot names for the other seats. Defaults to 3 RuleBots.
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
from hearts.bot import Bot  # noqa: E402
from hearts.bots.human_bot import HumanBot  # noqa: E402
from hearts.bots.ml_bot import MLBot, load_ml_bot  # noqa: E402
from hearts.game import Game  # noqa: E402

RESET = '\033[0m'
GREY = '\033[90m'
CYAN = '\033[96m'
BOLD = '\033[1m'

def discover_bots() -> dict[str, type[Bot]]:
    """Find all non-human Bot subclasses in hearts.bots."""
    registry: dict[str, type[Bot]] = {}
    for info in pkgutil.iter_modules(hearts.bots.__path__):
        module = importlib.import_module(
            f"hearts.bots.{info.name}"
        )
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, Bot)
                and obj is not Bot
                and obj is not HumanBot
            ):
                registry[obj.__name__.lower()] = obj
    return registry


def make_bot(
    name: str,
    registry: dict[str, type[Bot]],
    rng: random.Random,
    model_path: str | None = None,
) -> Bot:
    """Create a bot instance by name."""
    key = name.lower()
    if key not in registry:
        available = ", ".join(
            cls.__name__ for cls in registry.values()
        )
        print(f"Unknown bot: {name}")
        print(f"Available bots: {available}")
        sys.exit(1)
    cls = registry[key]
    if issubclass(cls, MLBot):
        if model_path is None:
            print("MLBot requires --model path to a trained model file.")
            sys.exit(1)
        return load_ml_bot(model_path, rng=random.Random(rng.randint(0, 2**63)))
    sig = inspect.signature(cls.__init__)
    if "rng" in sig.parameters:
        return cls(rng=random.Random(rng.randint(0, 2**63)))
    return cls()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play Hearts against 3 bots",
    )
    parser.add_argument(
        "bots", nargs="*", metavar="BOT",
        help="3 bot names for opponents (default: RuleBot)",
    )
    parser.add_argument(
        "--seat", type=int, default=0, choices=[0, 1, 2, 3],
        help="Your seat position 0-3 (default: 0)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to trained model file (required for MLBot)",
    )
    parser.add_argument(
        "--list", "-l", action="store_true",
        dest="list_bots",
        help="List available bots and exit",
    )
    args = parser.parse_args()

    registry = discover_bots()

    os.system("cls" if os.name == "nt" else "clear")
    if args.list_bots:
        print("Available bots:")
        for cls in registry.values():
            doc = (cls.__doc__ or "").strip().split("\n")[0]
            print(f"  {cls.__name__:<16} {doc}")
        return

    bot_names = args.bots or ["RuleBot", "RuleBot", "RuleBot"]
    if len(bot_names) != 3:
        parser.error("exactly 3 bot names are required")

    rng = random.Random(args.seed)
    seat = args.seat
    human = HumanBot(seat)

    opponents = [make_bot(n, registry, rng, args.model) for n in bot_names]
    bots: list[Bot] = []
    names: list[str] = []
    opp_idx = 0
    for i in range(4):
        if i == seat:
            bots.append(human)
            names.append("You")
        else:
            bots.append(opponents[opp_idx])
            names.append(bot_names[opp_idx])
            opp_idx += 1

    print(f"{BOLD}=== Hearts ==={RESET}")
    print(f"Seats: {', '.join(f'{n} ({CYAN if i == seat else GREY}P{i}{RESET})' for i, n in enumerate(names))}\n")

    result = Game(bots, rng=rng).play()

    print()
    print(f"{'Seat':<6} {'Player':<16} {'Score':>5}")
    print("-" * 30)
    for i in range(4):
        marker = " *" if i == result.winner else ""
        print(
            f"  {i:<4} {names[i]:<16} "
            f"{result.final_scores[i]:>5}{marker}"
        )
    print()
    if result.winner == seat:
        print("Congratulations, you win!")
    else:
        print(
            f"P{result.winner} ({names[result.winner]}) wins "
            f"with {result.final_scores[result.winner]} points "
            f"after {result.num_rounds} rounds."
        )


if __name__ == "__main__":
    main()
