"""Model definitions used in the experiments."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_logistic_regression(
    C: float = 1.0,
) -> Pipeline:
    """Build the scaled logistic regression baseline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    max_iter=2000,
                ),
            ),
        ]
    )