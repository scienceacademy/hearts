"""Evaluate a trained Hearts ML model against RuleBot.

Loads a saved model, runs a tournament (1x MLBot vs 3x RuleBot),
and prints a unified report with training metrics, feature
importance, and tournament results.

Usage:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --model models/model.pkl --games 200
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from hearts.ml.evaluate import (  # noqa: E402
    print_report,
    run_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a Hearts ML model vs RuleBot"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/model.pkl",
        help="Path to saved model (default: models/model.pkl)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=200,
        help="Number of tournament games (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Run tournament games in parallel (default: sequential)",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: {model_path} not found.")
        print()
        print("Train a model first:")
        print("  python scripts/train_model.py")
        sys.exit(1)

    def tournament_progress(current, total):
        print(
            f"\r  Games: {current}/{total}", end="", flush=True
        )

    mode = "parallel" if args.parallel else "sequential"
    print(f"Loading model from {args.model}...")
    print(
        f"Running tournament: MLBot vs 3x RuleBot "
        f"({args.games} games, {mode} mode)..."
    )

    result = run_evaluation(
        model_path=args.model,
        n_games=args.games,
        seed=args.seed,
        progress=tournament_progress,
        parallel=args.parallel,
    )
    print()
    print()

    print_report(result)


if __name__ == "__main__":
    main()
