import argparse
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
    final_holdout_split,
    random_stratified_splits,
    split_development_holdout,
)
from src.experiments import (
    Experiment,
    run_experiment,
)
from src.models import (
    build_logistic_regression,
    build_xgboost,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run model evaluation experiments.",
    )

    parser.add_argument(
        "--experiment",
        required=True,
        choices=[
            "logreg_tuning",
            "xgboost_tuning",
            "final_evaluation",
        ],
        help="Experiment to run.",
    )

    return parser.parse_args()


def build_development_protocols(
    development: pd.DataFrame,
) -> dict[str, list[EvaluationSplit]]:
    """Build the common development evaluation protocols."""
    return {
        "random_stratified": random_stratified_splits(
            development
        ),
        "expanding_window": expanding_window_splits(
            development
        ),
    }


def build_logreg_tuning_experiment(
    df: pd.DataFrame,
) -> Experiment:
    """Build the Logistic Regression regularization experiment."""
    development, _ = split_development_holdout(df)

    models: dict[str, BaseEstimator] = {
        "logreg_C0.01": build_logistic_regression(
            C=0.01
        ),
        "logreg_C0.1": build_logistic_regression(
            C=0.1
        ),
        "logreg_C1": build_logistic_regression(
            C=1.0
        ),
        "logreg_C10": build_logistic_regression(
            C=10.0
        ),
    }

    return Experiment(
        data=development,
        models=models,
        protocols=build_development_protocols(
            development
        ),
        output_path=RESULTS_DIR / "logreg_tuning.csv",
    )


def build_xgboost_tuning_experiment(
    df: pd.DataFrame,
) -> Experiment:
    """Build the XGBoost tree-depth experiment."""
    development, _ = split_development_holdout(df)

    models: dict[str, BaseEstimator] = {
        "xgboost_depth2": build_xgboost(
            max_depth=2
        ),
        "xgboost_depth4": build_xgboost(
            max_depth=4
        ),
        "xgboost_depth6": build_xgboost(
            max_depth=6
        ),
    }

    return Experiment(
        data=development,
        models=models,
        protocols=build_development_protocols(
            development
        ),
        output_path=RESULTS_DIR / "xgboost_tuning.csv",
    )


def build_final_evaluation_experiment(
    df: pd.DataFrame,
) -> Experiment:
    """Build the final holdout evaluation experiment."""
    models: dict[str, BaseEstimator] = {
        "logreg_C10": build_logistic_regression(
            C=10.0
        ),
    }

    protocols = {
        "final_holdout": final_holdout_split(
            df
        ),
    }

    return Experiment(
        data=df,
        models=models,
        protocols=protocols,
        output_path=RESULTS_DIR / "final_evaluation.csv",
    )


def build_experiment(
    name: str,
    df: pd.DataFrame,
) -> Experiment:
    """Build an experiment from its command-line name."""
    if name == "logreg_tuning":
        return build_logreg_tuning_experiment(df)

    if name == "xgboost_tuning":
        return build_xgboost_tuning_experiment(df)

    if name == "final_evaluation":
        return build_final_evaluation_experiment(df)

    raise ValueError(f"Unknown experiment: {name}")


def get_total_splits(
    experiment: Experiment,
) -> int:
    """Return the total number of model/split evaluations."""
    return len(experiment.models) * sum(
        len(splits)
        for splits in experiment.protocols.values()
    )


def save_results(
    results: pd.DataFrame,
    output_path: Path,
    artifact_name: str = "Results",
) -> None:
    """Save one experiment artifact to disk."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print(f"{artifact_name} saved to {output_path}")


def prediction_output_path(
    metrics_output_path: Path,
) -> Path:
    """Return the prediction artifact path for an experiment."""
    return metrics_output_path.with_name(
        f"{metrics_output_path.stem}_predictions.csv"
    )


def main() -> None:
    args = parse_args()

    df = pd.read_csv(PROCESSED_DATA_PATH)

    experiment = build_experiment(
        name=args.experiment,
        df=df,
    )

    with tqdm(
        total=get_total_splits(experiment),
        desc="Running experiment",
        unit="split",
    ) as progress_bar:
        experiment_output = run_experiment(
            experiment,
            progress_callback=progress_bar.update,
        )

    save_results(
        results=experiment_output.metrics,
        output_path=experiment.output_path,
        artifact_name="Metrics",
    )
    save_results(
        results=experiment_output.predictions,
        output_path=prediction_output_path(
            experiment.output_path
        ),
        artifact_name="Predictions",
    )


if __name__ == "__main__":
    main()
