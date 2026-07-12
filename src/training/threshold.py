"""
src/training/threshold.py

Standalone threshold optimization module.
Tests thresholds from THRESHOLD_MIN to THRESHOLD_MAX and selects the
threshold that maximizes F1-score on out-of-fold predictions.

Selection tiebreaking:
  1. Highest F1
  2. If tied: highest recall
  3. If still tied: closest to 0.50
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from config import THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP, REPORTS_DIR


def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_report: bool = True,
) -> float:
    """
    Sweep decision thresholds and return the optimal value.

    Parameters
    ----------
    y_true : array of int
        Ground-truth binary labels (0 or 1).
    y_prob : array of float
        Out-of-fold predicted probabilities for the positive class.
    save_report : bool
        If True, writes reports/threshold_analysis.csv.

    Returns
    -------
    float
        The selected decision threshold.
    """
    thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP / 2, THRESHOLD_STEP)
    records = []

    for thresh in thresholds:
        preds = (y_prob >= thresh).astype(int)

        prec = precision_score(y_true, preds, zero_division=0)
        rec  = recall_score(y_true, preds, zero_division=0)
        f1   = f1_score(y_true, preds, zero_division=0)
        acc  = accuracy_score(y_true, preds)

        tn = int(np.sum((y_true == 0) & (preds == 0)))
        fp = int(np.sum((y_true == 0) & (preds == 1)))
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        records.append({
            "threshold": round(float(thresh), 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "specificity": round(spec, 4),
            "accuracy": round(acc, 4),
        })

    df = pd.DataFrame(records)

    if save_report:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(REPORTS_DIR / "threshold_analysis.csv", index=False)

    # Selection logic
    best_f1 = df["f1"].max()
    candidates = df[df["f1"] == best_f1]

    best_recall = candidates["recall"].max()
    candidates = candidates[candidates["recall"] == best_recall]

    # Closest to 0.50
    candidates = candidates.copy()
    candidates["dist_to_half"] = (candidates["threshold"] - 0.5).abs()
    selected_row = candidates.sort_values("dist_to_half").iloc[0]

    return float(selected_row["threshold"])
