from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from src.evaluation import (
    EvaluationSplit,
    classification_metrics,
    per_class_f1,
)


def evaluate_split(
    model_name: str,
    protocol: str,
    model: BaseEstimator,
    df: pd.DataFrame,
    split: EvaluationSplit,
    feature_columns: list[str],
    class_labels: tuple[int, ...],
) -> dict:
    """Evaluate one model on one predefined split."""
    train = df.iloc[split.train_indices]
    validation = df.iloc[split.validation_indices]

    fitted_model = clone(model)

    fitted_model.fit(
        train[feature_columns],
        train["label"],
    )

    predictions = fitted_model.predict(
        validation[feature_columns],
    )

    y_true = validation["label"].to_numpy()

    result = {
        "model": model_name,
        "protocol": protocol,
        "split": split.name,
        "train_batches": split.train_batches,
        "validation_batches": split.validation_batches,
        "validation_batch": (
            split.validation_batches[0]
            if len(split.validation_batches) == 1
            else np.nan
        ),
        "n_train": len(train),
        "n_validation": len(validation),
        **classification_metrics(y_true, predictions),
    }

    class_scores = per_class_f1(
        y_true,
        predictions,
    )

    for label in class_labels:
        result[f"f1_class_{label}"] = class_scores.get(
            label,
            np.nan,
        )

    return result


def evaluate_model(
    model_name: str,
    protocol: str,
    model: BaseEstimator,
    df: pd.DataFrame,
    splits: list[EvaluationSplit],
    feature_columns: list[str],
    class_labels: tuple[int, ...],
    progress_callback: Callable[[int], object] | None = None,
) -> pd.DataFrame:
    """Evaluate one model across a collection of predefined splits.

    ``progress_callback`` is called after every completed split, so callers
    can show aggregate progress without coupling evaluation to a UI.
    """
    records = []

    for split in splits:
        records.append(
            evaluate_split(
                model_name=model_name,
                protocol=protocol,
                model=model,
                df=df,
                split=split,
                feature_columns=feature_columns,
                class_labels=class_labels,
            )
        )

        if progress_callback is not None:
            progress_callback(1)

    return pd.DataFrame.from_records(records)
