"""Model definitions used in the experiments."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


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


def build_xgboost(
    max_depth: int = 4,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
) -> XGBClassifier:
    """Build an XGBoost classifier."""
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )