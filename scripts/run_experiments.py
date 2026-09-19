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


def load_development_data() -> pd.DataFrame:
    """Load processed data and exclude the final temporal holdout."""
    df = pd.read_csv(PROCESSED_DATA_PATH)
    development, _ = split_development_holdout(df)

    return development


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
    """Return all target classes present in development data."""
    return tuple(
        sorted(
            int(label)
            for label in df["label"].unique()
        )
    )


def build_models() -> dict[str, BaseEstimator]:
    """Define models included in the development experiments."""
    return {
        "logistic_regression": build_logistic_regression(),
    }

def build_protocols(
    development: pd.DataFrame,
) -> dict[str, list[EvaluationSplit]]:
    """Define evaluation protocols used during model development."""
    return {
        "random_stratified": random_stratified_splits(
            development
        ),
        "expanding_window": expanding_window_splits(
            development
        ),
    }


def run_experiments(
    development: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate all configured models under all development protocols."""
    feature_columns = get_feature_columns(development)
    class_labels = get_class_labels(development)

    models = build_models()
    protocols = build_protocols(development)
    total_splits = len(models) * sum(
        len(splits) for splits in protocols.values()
    )

    experiment_results = []

    with tqdm(
        total=total_splits,
        desc="Running experiments",
        unit="split",
    ) as progress_bar:
        for model_name, model in models.items():
            for protocol_name, splits in protocols.items():
                results = evaluate_model(
                    model_name=model_name,
                    protocol=protocol_name,
                    model=model,
                    df=development,
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
) -> None:
    """Save development experiment results."""
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS_DIR
        / "development_results.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print(f"Results saved to {output_path}")


def main() -> None:
    development = load_development_data()
    results = run_experiments(development)
    save_results(results)


if __name__ == "__main__":
    main()
