import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator
from tqdm import tqdm

from src.config import (
    PROCESSED_DATA_PATH,
    RESULTS_DIR,
)
from src.evaluation import (
    EvaluationSplit,
    expanding_window_splits,
    random_stratified_splits,
    split_development_holdout,
)
from src.experiments import evaluate_model
from src.models import build_logistic_regression


@dataclass(frozen=True)
class Experiment:
    """Configuration required to run one experiment."""

    data: pd.DataFrame
    models: dict[str, BaseEstimator]
    protocols: dict[str, list[EvaluationSplit]]
    output_path: Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run model evaluation experiments.",
    )

    parser.add_argument(
        "--experiment",
        required=True,
        choices=["logreg_tuning"],
        help="Experiment to run.",
    )

    return parser.parse_args()


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


def build_logreg_tuning_experiment(
    df: pd.DataFrame,
) -> Experiment:
    """Build the logistic-regression regularization experiment."""
    development, _ = split_development_holdout(df)

    models = {
        "logreg_C0.01": build_logistic_regression(C=0.01),
        "logreg_C0.1": build_logistic_regression(C=0.1),
        "logreg_C1": build_logistic_regression(C=1.0),
        "logreg_C10": build_logistic_regression(C=10.0),
    }

    protocols = {
        "random_stratified": random_stratified_splits(
            development
        ),
        "expanding_window": expanding_window_splits(
            development
        ),
    }

    return Experiment(
        data=development,
        models=models,
        protocols=protocols,
        output_path=RESULTS_DIR / "logreg_tuning.csv",
    )


def build_experiment(
    name: str,
    df: pd.DataFrame,
) -> Experiment:
    """Build an experiment from its command-line name."""
    if name == "logreg_tuning":
        return build_logreg_tuning_experiment(df)

    raise ValueError(f"Unknown experiment: {name}")


def run_experiment(
    experiment: Experiment,
) -> pd.DataFrame:
    """Evaluate all models under all protocols in an experiment."""
    feature_columns = get_feature_columns(experiment.data)
    class_labels = get_class_labels(experiment.data)

    total_splits = len(experiment.models) * sum(
        len(splits)
        for splits in experiment.protocols.values()
    )

    experiment_results = []

    with tqdm(
        total=total_splits,
        desc="Running experiment",
        unit="split",
    ) as progress_bar:
        for model_name, model in experiment.models.items():
            for protocol_name, splits in experiment.protocols.items():
                results = evaluate_model(
                    model_name=model_name,
                    protocol=protocol_name,
                    model=model,
                    df=experiment.data,
                    splits=splits,
                    feature_columns=feature_columns,
                    class_labels=class_labels,
                    progress_callback=progress_bar.update,
                )

                experiment_results.append(results)

    return pd.concat(
        experiment_results,
        ignore_index=True,
    )


def save_results(
    results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save experiment results to disk."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print(f"Results saved to {output_path}")


def main() -> None:
    args = parse_args()

    df = pd.read_csv(PROCESSED_DATA_PATH)

    experiment = build_experiment(
        name=args.experiment,
        df=df,
    )

    results = run_experiment(experiment)

    save_results(
        results=results,
        output_path=experiment.output_path,
    )


if __name__ == "__main__":
    main()
    