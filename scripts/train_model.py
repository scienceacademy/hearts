"""Train a Hearts ML model from generated game data.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --features BasicFeatureExtractor
    python scripts/train_model.py --data-dir data/output --seed 42
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from hearts.ml.features import (  # noqa: E402
    BasicFeatureExtractor,
    StudentFeatureExtractor,
)
from hearts.ml.train import (  # noqa: E402
    build_dataset,
    load_data,
    save_model,
    train_model,
)

EXTRACTORS = {
    "BasicFeatureExtractor": BasicFeatureExtractor,
    "StudentFeatureExtractor": StudentFeatureExtractor,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Hearts ML model"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/output",
        help="Directory with play_decisions.jsonl (default: data/output)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/model.pkl",
        help="Output model path (default: models/model.pkl)",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="StudentFeatureExtractor",
        choices=list(EXTRACTORS.keys()),
        help="Feature extractor class (default: StudentFeatureExtractor)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    extractor = EXTRACTORS[args.features]()
    extractor_name = args.features
    pipeline_start = time.monotonic()

    data_file = Path(args.data_dir) / "play_decisions.jsonl"
    if not data_file.exists():
        print(f"Error: {data_file} not found.")
        print()
        print("Generate training data first:")
        print("  python scripts/generate_data.py --games 10000 --seed 42")
        sys.exit(1)

    print(f"Loading data from {args.data_dir}...")
    t0 = time.monotonic()
    records = load_data(args.data_dir)
    t_load = time.monotonic() - t0
    print(f"  {len(records)} play decisions loaded ({t_load:.1f}s)")
    print()

    def build_progress(current, total):
        elapsed = time.monotonic() - t_build
        rate = current / elapsed if elapsed > 0 else 0
        remaining = (total - current) / rate if rate > 0 else 0
        print(
            f"\r  Extracting features: {current}/{total} "
            f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]",
            end="", flush=True,
        )
        if current == total:
            print()

    print(f"Building features with {extractor_name}...")
    t_build = time.monotonic()
    X, y = build_dataset(records, extractor, progress=build_progress)
    t_features = time.monotonic() - t_build
    print(f"  Feature vector size: {X.shape[1]} features")
    print(f"  Samples: {X.shape[0]}")
    print()

    print("Training RandomForestClassifier...")
    t0 = time.monotonic()
    model, metrics = train_model(
        X, y, extractor.feature_names(), seed=args.seed
    )
    t_train = time.monotonic() - t0
    print(
        f"  Done ({t_train:.1f}s, "
        f"{metrics['n_jobs']} parallel workers)"
    )

    print()
    print("=== Training Results ===")
    print(f"Feature vector size:  {metrics['n_features']} features")
    print(f"Training samples:     {metrics['n_train']}")
    print(f"Test samples:         {metrics['n_test']}")
    print(f"Train accuracy:       {metrics['train_accuracy']:.1%}")
    print(f"Test accuracy:        {metrics['test_accuracy']:.1%}")
    print()

    print("=== Feature Importance (top 10) ===")
    for i, (name, importance) in enumerate(
        metrics["feature_importances"][:10]
    ):
        print(f"  {i + 1:>2}. {name:<24} {importance:.4f}")
    print()

    save_model(model, extractor_name, metrics, args.output)
    total_time = time.monotonic() - pipeline_start
    print(f"Model saved to {args.output}")
    print(
        f"Total time: {total_time:.0f}s "
        f"(load {t_load:.0f}s + features {t_features:.0f}s "
        f"+ train {t_train:.0f}s)"
    )


if __name__ == "__main__":
    main()
