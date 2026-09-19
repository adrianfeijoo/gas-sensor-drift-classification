"""Prepare the Gas Sensor Array Drift dataset.

Parses the original UCI batch files into a single tabular CSV file.
No model-dependent preprocessing (scaling, feature selection, etc.) is
performed here.
"""

from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_DIR, PROCESSED_DATA_PATH


N_FEATURES = 128
EXPECTED_BATCHES = 10


def parse_line(line: str, batch: int) -> dict:
    """Parse one LIBSVM-like row from the original dataset."""
    tokens = line.split()

    if len(tokens) != N_FEATURES + 1:
        raise ValueError(
            f"Expected {N_FEATURES} features, found {len(tokens) - 1} "
            f"in batch {batch}."
        )

    row: dict[str, int | float] = {
        "label": int(tokens[0]),
        "batch": batch,
    }

    for token in tokens[1:]:
        index, value = token.split(":", maxsplit=1)
        row[f"feature_{int(index)}"] = float(value)

    return row


def parse_batch(path: Path, batch: int) -> list[dict]:
    """Parse all observations from a batch file."""
    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(parse_line(line, batch))
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"Failed to parse {path.name}, line {line_number}."
                ) from exc

    return rows


def prepare_dataset() -> pd.DataFrame:
    """Load and combine all raw acquisition batches."""
    rows = []

    for batch in range(1, EXPECTED_BATCHES + 1):
        path = RAW_DATA_DIR / f"batch{batch}.dat"

        if not path.exists():
            raise FileNotFoundError(f"Missing raw data file: {path}")

        rows.extend(parse_batch(path, batch))

    df = pd.DataFrame(rows)

    feature_columns = [f"feature_{i}" for i in range(1, N_FEATURES + 1)]
    expected_columns = ["label", "batch", *feature_columns]

    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {sorted(missing_columns)}"
        )

    # Enforce a stable column order.
    df = df[expected_columns]

    if df.isna().any().any():
        raise ValueError("Processed dataset contains missing values.")

    return df


def main() -> None:
    df = prepare_dataset()

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Saved processed dataset to: {PROCESSED_DATA_PATH}")
    print(f"Samples: {len(df):,}")
    print(f"Features: {N_FEATURES}")
    print(f"Classes: {df['label'].nunique()}")
    print(f"Batches: {df['batch'].nunique()}")


if __name__ == "__main__":
    main()
