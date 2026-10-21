"""Точний пошук порога для C(t) = 10 * FN(t) + FP(t)."""
from __future__ import annotations

import numpy as np
import pandas as pd


COST_FN = 10
COST_FP = 1
COLUMNS = ["threshold", "TP", "FP", "FN", "TN", "C"]


def validate_inputs(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel()
    if scores.size == 0 or labels.size != scores.size:
        raise ValueError("Оцінки та мітки мають бути непорожні та однакової довжини")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Оцінки повинні бути скінченними числами в межах [0, 1]")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("Мітки повинні бути двійковими: 0 чи 1")
    return scores, labels.astype(np.int64, copy=False)


def count_at_threshold(scores: np.ndarray, labels: np.ndarray, t: float) -> dict:
    """Прямий розрахунок матриці помилок для ОДНОГО порога."""
    scores, labels = validate_inputs(scores, labels)
    positive = scores >= t
    tp = int(np.count_nonzero(positive & (labels == 1)))
    fp = int(np.count_nonzero(positive & (labels == 0)))
    fn = int(np.count_nonzero(~positive & (labels == 1)))
    tn = int(np.count_nonzero(~positive & (labels == 0)))
    return {"threshold": float(t), "TP": tp, "FP": fp, "FN": fn,
            "TN": tn, "C": COST_FN * fn + COST_FP * fp}


def all_thresholds_fast(scores: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    
    scores, labels = validate_inputs(scores, labels)
    n = scores.size
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    
    ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    cum_pos = np.cumsum(sorted_labels, dtype=np.int64)
    tp = cum_pos[ends]
    predicted_pos = ends + 1
    fp = predicted_pos - tp
    p = int(labels.sum())
    fn = p - tp
    tn = (n - p) - fp

    thresholds = np.r_[np.inf, sorted_scores[ends]]
    tp = np.r_[0, tp]
    fp = np.r_[0, fp]
    fn = np.r_[p, fn]
    tn = np.r_[n - p, tn]
    costs = COST_FN * fn + COST_FP * fp

    df = pd.DataFrame({"threshold": thresholds, "TP": tp, "FP": fp,
                       "FN": fn, "TN": tn, "C": costs})
    for col in COLUMNS[1:]:
        df[col] = df[col].astype(np.int64)
    assert df.shape[0] == len(ends) + 1
    assert (df["TP"] + df["FN"] == p).all()
    assert (df["FP"] + df["TN"] == n-p).all()
    return df[COLUMNS]


def select_optimal_threshold(table: pd.DataFrame) -> dict:
    
    
    min_cost = int(table["C"].min())
    best = table.loc[table["C"] == min_cost].sort_values(
        "threshold", ascending=False, kind="stable"
    ).iloc[0]
    return {k: (float(best[k]) if k == "threshold" else int(best[k])) for k in COLUMNS}


def display_threshold(t: float) -> str:
    return "+∞" if np.isposinf(t) else format(float(t), ".9g")
