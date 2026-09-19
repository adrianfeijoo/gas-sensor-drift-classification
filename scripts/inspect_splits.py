"""Inspect the evaluation splits used in the project."""

import pandas as pd

from src.config import PROCESSED_DATA_PATH
from src.evaluation import (
    expanding_window_splits,
    random_stratified_splits,
    split_development_holdout,
)


def describe_split(split, df: pd.DataFrame) -> None:
    train = df.iloc[split.train_indices]
    validation = df.iloc[split.validation_indices]

    print(f"\n{split.name}")
    print("-" * len(split.name))

    print(
        f"Train: {len(train):5d} samples | "
        f"batches: {split.train_batches}"
    )
    print(
        f"Validation: {len(validation):5d} samples | "
        f"batches: {split.validation_batches}"
    )

    print(
        "Validation classes:",
        sorted(validation["label"].unique()),
    )


def main() -> None:
    df = pd.read_csv(PROCESSED_DATA_PATH)

    development, holdout = split_development_holdout(df)

    print("Dataset")
    print(f"Development samples: {len(development)}")
    print(f"Development batches: {sorted(development['batch'].unique())}")
    print(f"Final holdout samples: {len(holdout)}")
    print(f"Final holdout batches: {sorted(holdout['batch'].unique())}")

    print("\n\nRandom stratified CV")

    for split in random_stratified_splits(development):
        describe_split(split, development)

    print("\n\nExpanding-window validation")

    for split in expanding_window_splits(development):
        describe_split(split, development)


if __name__ == "__main__":
    main()