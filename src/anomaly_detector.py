"""
anomaly_detector.py
--------------------
Importable module: fits an unsupervised Isolation Forest on
data/metrics_with_incident.csv's four numeric metric columns and scores it
against the file's own is_anomaly ground-truth labels.

The label boundary is structural, not just a convention: load_metrics() and
detect() never reference "is_anomaly" at all -- only evaluate() does, and
only after detect() has already produced predictions. That ordering is what
makes this a genuine unsupervised-then-scored evaluation instead of the
model accidentally leaking the answer it's being graded against.

Every function takes its DataFrame as an explicit parameter rather than
reading a module-level global, so importing this file has no side effects
(no CSV read, no model fit) and each function can be tested independently.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

# The ONLY columns the model is ever allowed to see. Keeping this as one
# named constant (rather than repeating the column list in every function)
# makes it easy to audit that is_anomaly never sneaks into the feature set.
FEATURE_COLUMNS = ["cpu_pct", "mem_pct", "req_per_sec", "error_rate"]

DEFAULT_METRICS_PATH = "data/metrics_with_incident.csv"
RANDOM_STATE = 42  # fixed seed for IsolationForest -> reproducible fits; a modeling choice, not a data threshold

# The three contamination settings the assignment asks sweep_contamination() to compare.
DEFAULT_CONTAMINATION_VALUES = (0.01, 0.04, 0.10)


def load_metrics(path: str = DEFAULT_METRICS_PATH) -> pd.DataFrame:
    """Load the metrics CSV (with parsed timestamps) as a DataFrame."""
    # parse_dates turns the timestamp column into real Timestamp objects,
    # matching the convention used by the other scripts in this repo.
    return pd.read_csv(path, parse_dates=["timestamp"])


def detect(df: pd.DataFrame, contamination: float) -> pd.DataFrame:
    """Fit an IsolationForest on the 4 numeric metric columns and return df + a predicted_anomaly column.

    contamination is IsolationForest's expected proportion of outliers in
    the data -- it directly controls how aggressive the decision threshold
    is, so a fresh model is fit here every call rather than reused/cached.
    """
    # Isolation Forest is unsupervised -- it must never see the ground-truth label,
    # so select ONLY the four numeric metric columns before fitting
    X = df[FEATURE_COLUMNS]
    model = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
    raw_predictions = model.fit_predict(X)  # sklearn convention: -1 = outlier, 1 = inlier
    out = df.copy()  # never mutate the caller's DataFrame in place
    # Remap sklearn's -1/1 convention onto is_anomaly's own 1=anomaly/0=normal convention.
    out["predicted_anomaly"] = np.where(raw_predictions == -1, 1, 0)
    return out


def evaluate(df: pd.DataFrame) -> dict:
    """Score predicted_anomaly against is_anomaly and return precision/recall/F1.

    This is the ONLY function in the module that reads is_anomaly -- and it
    only ever does so for scoring, never to influence detect()'s predictions.
    """
    y_true = df["is_anomaly"]
    y_pred = df["predicted_anomaly"]
    # zero_division=0 keeps a contamination setting that predicts zero
    # positives from raising a warning/crashing instead of just scoring 0.
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {"precision": precision, "recall": recall, "f1": f1}


def sweep_contamination(df: pd.DataFrame,
                         contamination_values: tuple[float, ...] = DEFAULT_CONTAMINATION_VALUES) -> pd.DataFrame:
    """Run detect() + evaluate() once per contamination value and return a comparison table."""
    rows = []
    for contamination in contamination_values:
        predicted = detect(df, contamination)          # fit+predict blind to is_anomaly
        scores = evaluate(predicted)                    # score against is_anomaly only now
        rows.append({"contamination": contamination, **scores})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Load once, sweep every requested contamination setting, and print the comparison table.
    metrics_df = load_metrics()
    results = sweep_contamination(metrics_df)
    print(results.to_string(index=False))
