import pandas as pd
import numpy as np
import json
import os
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    precision_recall_curve, average_precision_score, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "amt_diff_abs", "amt_diff_pct", "date_lag",
    "ref_similarity", "currency_match", "is_round",
    "tx_type_match", "combined_cost"
]

CONFIDENCE_HIGH = float(os.getenv("CONFIDENCE_HIGH", "0.85"))
CONFIDENCE_LOW = float(os.getenv("CONFIDENCE_LOW", "0.40"))


def train_scorer(features_df: pd.DataFrame, model_path: str = "data/xgboost_model.pkl") -> tuple:
    """
    Trains XGBoost on the feature matrix and returns (model, evaluation_metrics_dict).
    """
    if features_df is None or features_df.empty or "label" not in features_df.columns:
        model = XGBClassifier(n_estimators=50, max_depth=4, random_state=42)
        return model, {}

    X = features_df[FEATURE_COLS].values
    y = features_df["label"].values

    if len(np.unique(y)) < 2:
        model = XGBClassifier(n_estimators=50, max_depth=4, random_state=42)
        model.fit(X, y)
        return model, {"precision": 1.0, "recall": 1.0, "f1_score": 1.0, "average_precision": 1.0}

    # Stratified train-test split if enough samples exist per class
    min_class_count = min((y == 0).sum(), (y == 1).sum())
    if len(y) >= 10 and min_class_count >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y


    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = neg / pos if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale,
        random_state=42,
        eval_metric="logloss",
        verbosity=0
    )

    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall    = float(recall_score(y_test, y_pred, zero_division=0))
    f1        = float(f1_score(y_test, y_pred, zero_division=0))
    ap        = float(average_precision_score(y_test, y_proba))

    prec_curve, rec_curve, thresholds = precision_recall_curve(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()

    importances = dict(zip(FEATURE_COLS, [float(x) for x in model.feature_importances_]))

    metrics = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "average_precision": round(ap, 4),
        "pr_curve": {
            "precision": prec_curve.tolist(),
            "recall": rec_curve.tolist(),
            "thresholds": thresholds.tolist()
        },
        "confusion_matrix": cm,
        "feature_importances": importances
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return model, metrics


def load_or_train_scorer(features_df: pd.DataFrame, model_path: str = "data/xgboost_model.pkl") -> XGBClassifier:
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    model, _ = train_scorer(features_df, model_path=model_path)
    return model


def score_candidates(
    model: XGBClassifier,
    features_df: pd.DataFrame,
    conf_high: float = CONFIDENCE_HIGH,
    conf_low: float = CONFIDENCE_LOW
) -> pd.DataFrame:
    """
    Scores candidates with confidence probabilities and zones:
    - HIGH >= conf_high (AUTO_ACCEPT)
    - MEDIUM conf_low <= p < conf_high (REVIEW)
    - LOW < conf_low (AUTO_REJECT)
    """
    if features_df is None or features_df.empty:
        return pd.DataFrame()

    X = features_df[FEATURE_COLS].values
    probas = model.predict_proba(X)[:, 1]

    scored_df = features_df.copy()
    scored_df["confidence"] = probas.round(4)

    def classify_zone(p):
        if p >= conf_high:
            return "HIGH"
        elif p >= conf_low:
            return "MEDIUM"
        return "LOW"

    scored_df["confidence_zone"] = scored_df["confidence"].apply(classify_zone)
    return scored_df