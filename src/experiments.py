from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.preprocessing import LabelEncoder

from src.evaluation import (
    EvaluationSplit,
    classification_metrics,
    per_class_f1,
)


@dataclass(frozen=True)
class Experiment:
    """Configuration required to run one experiment."""

    data: pd.DataFrame
    models: dict[str, BaseEstimator]
    protocols: dict[str, list[EvaluationSplit]]
    output_path: Path


@dataclass(frozen=True)
class ExperimentOutput:
    """Metrics and prediction records produced by an experiment run."""

    metrics: pd.DataFrame
    predictions: pd.DataFrame


def get_feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    """Return sensor feature columns ordered by feature index."""
    feature_columns = [
        column
        for column in df.columns
        if column.startswith("feature_")
    ]

    return sorted(
        feature_columns,
        key=lambda column: int(
            column.split("_", maxsplit=1)[1]
        ),
    )


def get_class_labels(
    df: pd.DataFrame,
) -> tuple[int, ...]:
    """Return all target classes present in the experiment data."""
    return tuple(
        sorted(
            int(label)
            for label in df["label"].unique()
        )
    )


def evaluate_split(
    model_name: str,
    protocol: str,
    model: BaseEstimator,
    df: pd.DataFrame,
    split: EvaluationSplit,
    feature_columns: list[str],
    class_labels: tuple[int, ...],
) -> tuple[dict, pd.DataFrame]:
    """Evaluate one model and retain its validation predictions."""
    train = df.iloc[split.train_indices]
    validation = df.iloc[split.validation_indices]

    fitted_model = clone(model)

    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(
        train["label"],
    )

    fitted_model.fit(
        train[feature_columns],
        y_train,
    )

    encoded_predictions = fitted_model.predict(
        validation[feature_columns],
    )

    predictions = label_encoder.inverse_transform(
        encoded_predictions.astype(int),
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
        **classification_metrics(
            y_true,
            predictions,
        ),
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

    prediction_records = pd.DataFrame(
        {
            "model": model_name,
            "protocol": protocol,
            "split": split.name,
            "validation_batch": validation["batch"].to_numpy(),
            "actual": y_true,
            "predicted": predictions,
        }
    )

    return result, prediction_records


def evaluate_model(
    model_name: str,
    protocol: str,
    model: BaseEstimator,
    df: pd.DataFrame,
    splits: list[EvaluationSplit],
    feature_columns: list[str],
    class_labels: tuple[int, ...],
    progress_callback: Callable[[int], object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one model across splits and retain every prediction."""
    records = []
    prediction_frames = []

    for split in splits:
        metrics, predictions = evaluate_split(
            model_name=model_name,
            protocol=protocol,
            model=model,
            df=df,
            split=split,
            feature_columns=feature_columns,
            class_labels=class_labels,
        )
        records.append(metrics)
        prediction_frames.append(predictions)

        if progress_callback is not None:
            progress_callback(1)

    return (
        pd.DataFrame.from_records(records),
        pd.concat(prediction_frames, ignore_index=True),
    )


def run_experiment(
    experiment: Experiment,
    progress_callback: Callable[[int], object] | None = None,
) -> ExperimentOutput:
    """Evaluate every model and return metrics plus predictions."""
    feature_columns = get_feature_columns(
        experiment.data
    )
    class_labels = get_class_labels(
        experiment.data
    )

    experiment_results = []
    experiment_predictions = []

    for model_name, model in experiment.models.items():
        for protocol_name, splits in experiment.protocols.items():
            metrics, predictions = evaluate_model(
                model_name=model_name,
                protocol=protocol_name,
                model=model,
                df=experiment.data,
                splits=splits,
                feature_columns=feature_columns,
                class_labels=class_labels,
                progress_callback=progress_callback,
            )

            experiment_results.append(metrics)
            experiment_predictions.append(predictions)

    return ExperimentOutput(
        metrics=pd.concat(
            experiment_results,
            ignore_index=True,
        ),
        predictions=pd.concat(
            experiment_predictions,
            ignore_index=True,
        ),
    )
