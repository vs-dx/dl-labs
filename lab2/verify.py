"""Незалежні перевірки порогового алгоритму на наборах A, B, C."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from threshold import all_thresholds_fast, select_optimal_threshold, COLUMNS, display_threshold


CASES = {
    "A": (np.array([0.9, 0.7, 0.7, 0.4, 0.2, 0.1], dtype=np.float64),
          np.array([0, 1, 0, 1, 0, 0], dtype=np.int64)),
    "B": (np.array([0.9, 0.6, 0.3], dtype=np.float64),
          np.array([0, 0, 0], dtype=np.int64)),
    "C": (np.array([0.8] + [0.5]*11 + [0.2], dtype=np.float64),
          np.array([1, 1] + [0]*10 + [0], dtype=np.int64)),
}


def independent_slow_table(scores: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    
    candidates = [float("inf")] + sorted({float(s) for s in scores}, reverse=True)
    rows = []
    for threshold in candidates:
        tp = fp = fn = tn = 0
        for s, y in zip(scores, labels):
            prediction = int(s >= threshold)
            if prediction == 1 and y == 1:
                tp += 1
            elif prediction == 1 and y == 0:
                fp += 1
            elif prediction == 0 and y == 1:
                fn += 1
            else:
                tn += 1
        rows.append({"threshold": threshold, "TP": tp, "FP": fp,
                     "FN": fn, "TN": tn, "C": 10 * fn + fp})
    return pd.DataFrame(rows)[COLUMNS]


def run_checks(output: Path | None = None) -> pd.DataFrame:
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, (scores, labels) in CASES.items():
        fast = all_thresholds_fast(scores, labels)
        slow = independent_slow_table(scores, labels)
        
        pd.testing.assert_frame_equal(fast, slow, check_exact=True)
        fast_opt = select_optimal_threshold(fast)
        slow_opt = select_optimal_threshold(slow)
        assert fast_opt == slow_opt
        expected = {"A": 0.4, "B": float("inf"), "C": 0.8}[name]
        assert fast_opt["threshold"] == expected, f"Невірний поріг для {name}"
        if name == "C":
            tie = fast.loc[fast.C == fast.C.min(), "threshold"].tolist()
            assert tie == [0.8, 0.5], f"C має дві рівні вартості: {tie}"
        if output is not None:
            fast.to_csv(output / f"check_{name}.csv", index=False, float_format="%.17g")
        summary.append({"case": name, "candidates": len(fast),
                        "threshold": fast_opt["threshold"], "TP": fast_opt["TP"],
                        "FP": fast_opt["FP"], "FN": fast_opt["FN"],
                        "TN": fast_opt["TN"], "C": fast_opt["C"],
                        "exact_match": True})
    result = pd.DataFrame(summary)
    if output is not None:
        result.to_csv(output / "verification_summary.csv", index=False, float_format="%.17g")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results") / "verification")
    args = parser.parse_args()
    summary = run_checks(args.output)
    print("Усі три незалежні перевірки пройдено (точний збіг):")
    for row in summary.itertuples():
        print(f"{row.case}: t*={display_threshold(row.threshold)}; TP={row.TP}; FP={row.FP}; "
              f"FN={row.FN}; TN={row.TN}; C={row.C}")


if __name__ == "__main__":
    main()
