"""Evaluation protocols for gas classification under sensor drift."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold


DEFAULT_HOLDOUT_BATCH = 10


@dataclass(frozen=True)
class EvaluationSplit:
    """Indices and metadata describing one evaluation split."""

    name: str
    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_batches: tuple[int, ...]
    validation_batches: tuple[int, ...]


def split_development_holdout(
    df: pd.DataFrame,
    holdout_batch: int = DEFAULT_HOLDOUT_BATCH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate development data from the final temporal holdout."""

    development = df[df["batch"] != holdout_batch].copy()
    holdout = df[df["batch"] == holdout_batch].copy()

    if development.empty:
        raise ValueError("Development set is empty.")

    if holdout.empty:
        raise ValueError(
            f"No observations found for holdout batch {holdout_batch}."
        )

    return (
        development.reset_index(drop=True),
        holdout.reset_index(drop=True),
    )


def random_stratified_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[EvaluationSplit]:
    """Create random stratified cross-validation splits.

    Batch membership is deliberately ignored when generating these splits.
    This provides a conventional IID-like reference evaluation.
    """

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    y = df["label"].to_numpy()
    splits = []

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(df, y),
        start=1,
    ):
        train_batches = tuple(
            sorted(df.iloc[train_idx]["batch"].unique())
        )
        validation_batches = tuple(
            sorted(df.iloc[val_idx]["batch"].unique())
        )

        splits.append(
            EvaluationSplit(
                name=f"random_fold_{fold}",
                train_indices=train_idx,
                validation_indices=val_idx,
                train_batches=train_batches,
                validation_batches=validation_batches,
            )
        )

    return splits


def expanding_window_splits(
    df: pd.DataFrame,
) -> list[EvaluationSplit]:
    """Create chronological next-batch expanding-window splits.

    For each split, all previous batches are used for training and the
    immediately following batch is used for validation.
    """

    batches = sorted(df["batch"].unique())

    if len(batches) < 2:
        raise ValueError(
            "At least two batches are required for expanding-window evaluation."
        )

    splits = []

    for position in range(1, len(batches)):
        train_batches = tuple(batches[:position])
        validation_batch = batches[position]

        train_idx = np.flatnonzero(
            df["batch"].isin(train_batches).to_numpy()
        )
        val_idx = np.flatnonzero(
            (df["batch"] == validation_batch).to_numpy()
        )

        splits.append(
            EvaluationSplit(
                name=f"temporal_batch_{validation_batch}",
                train_indices=train_idx,
                validation_indices=val_idx,
                train_batches=train_batches,
                validation_batches=(validation_batch,),
            )
        )

    return splits


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute primary and secondary classification metrics.

    Macro-F1 is computed over classes present in the evaluation set.
    """

    labels_present = np.unique(y_true)

    return {
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels_present,
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def per_class_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[int, float]:
    """Compute F1 independently for every class present in the test set."""

    labels = np.unique(y_true)

    scores = np.asarray(
        f1_score(
            y_true,
            y_pred,
            labels=labels,
            average=None,
            zero_division=0,
        ),
        dtype=float,
    )

    return {
        int(label): float(score)
        for label, score in zip(labels, scores)
    }
