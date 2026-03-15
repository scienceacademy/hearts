"""Generate training data for Hearts ML models.

Runs many games and records every play and pass decision along with
outcome annotations (round score, game rank, etc.).

Usage:
    python scripts/generate_data.py --games 10000 --seed 42
    python scripts/generate_data.py RuleBot RandomBot RuleBot RandomBot
    python scripts/generate_data.py RandomBot --games 5000
    python scripts/generate_data.py --list

Bot names are positional and case-insensitive. Any Bot subclass in
hearts/bots/ is automatically discovered. If fewer than 4 bots are
specified, the last one fills the remaining seats. Defaults to 4x
RuleBot if none are specified.
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
from hearts.data.generator import generate_dataset  # noqa: E402


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
    cls: type[Bot],
    rng: random.Random | None,
) -> Bot:
    """Create a bot instance, passing rng if the constructor accepts one."""
    sig = inspect.signature(cls.__init__)
    if "rng" in sig.parameters and rng is not None:
        return cls(rng=random.Random(rng.randint(0, 2**63)))
    return cls()


def resolve_bot_names(
    names: list[str],
    registry: dict[str, type[Bot]],
) -> list[type[Bot]]:
    """Resolve bot names to classes, padding to 4 with the last entry."""
    if not names:
        names = ["RuleBot"]
    # Pad to 4 by repeating the last name
    while len(names) < 4:
        names.append(names[-1])
    if len(names) != 4:
        print("Error: at most 4 bot names allowed")
        sys.exit(1)

    classes = []
    for name in names:
        key = name.lower()
        if key not in registry:
            available = ", ".join(
                cls.__name__ for cls in registry.values()
            )
            print(f"Unknown bot: {name}")
            print(f"Available bots: {available}")
            sys.exit(1)
        classes.append(registry[key])
    return classes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Hearts ML training data"
    )
    parser.add_argument(
        "bots",
        nargs="*",
        metavar="BOT",
        help=(
            "Bot names for seats 0-3 (case-insensitive). "
            "If fewer than 4, the last one fills remaining seats. "
            "Defaults to 4x RuleBot."
        ),
    )
    parser.add_argument(
        "--games",
        type=int,
        default=10000,
        help="Number of games to simulate (default: 10000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/output",
        help="Output directory (default: data/output)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
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

    bot_classes = resolve_bot_names(args.bots, registry)
    bot_names = [cls.__name__ for cls in bot_classes]
    rng = random.Random(args.seed) if args.seed is not None else None

    def bot_factory():
        return [make_bot(cls, rng) for cls in bot_classes]

    import time

    bots_display = ", ".join(bot_names)
    print(f"Generating data from {args.games} games...")
    print(f"Bots: {bots_display}")
    print(f"Output directory: {args.output_dir}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print()

    t0 = time.monotonic()
    total_play, total_pass = generate_dataset(
        n_games=args.games,
        bot_factory=bot_factory,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    elapsed = time.monotonic() - t0

    print()
    print("=== Summary ===")
    print(f"Games played:      {args.games}")
    print(f"Bots:              {bots_display}")
    print(f"Play decisions:    {total_play}")
    print(f"Pass decisions:    {total_pass}")
    print(f"Time:              {elapsed:.0f}s")
    print(f"Play decisions file: {args.output_dir}/play_decisions.jsonl")
    print(f"Pass decisions file: {args.output_dir}/pass_decisions.jsonl")


if __name__ == "__main__":
    main()
