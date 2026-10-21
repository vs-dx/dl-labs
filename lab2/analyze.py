"""ЛР2, пункти 2--4: аналіз ЛИШЕ зі збережених прогнозів, без навчання."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from threshold import all_thresholds_fast, count_at_threshold, display_threshold, select_optimal_threshold
from verify import run_checks


START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


def metric_or_none(num: int, den: int) -> float | None:
    return num / den if den else None


def metric_text(value: float | None) -> str:
    return "не визначено" if value is None else f"{value:.6f}"


def json_threshold(value: float) -> str | float:
    return "+inf" if np.isposinf(value) else float(value)


def enhanced_counts(row: dict) -> dict:
    row = dict(row)
    row["precision"] = metric_or_none(row["TP"], row["TP"] + row["FP"])
    row["recall"] = metric_or_none(row["TP"], row["TP"] + row["FN"])
    return row


def format_summary_table(summary: pd.DataFrame) -> str:
    lines = ["| Правило | Поріг | TP | FP | FN | TN | C |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary.to_dict(orient="records"):
        lines.append(f"| {row['rule']} | {display_threshold(row['threshold'])} | "
                     f"{row['TP']} | {row['FP']} | {row['FN']} | {row['TN']} | {row['C']} |")
    return "\n".join(lines)


def save_curves(labels: np.ndarray, scores: np.ndarray, output: Path) -> float:
    precision, recall, _ = precision_recall_curve(labels, scores)
    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.step(recall, precision, where="post", linewidth=1.8, label="PR")
    ax.axhline(labels.mean(), linestyle="--", linewidth=1,
               label=f"Рівень частоти шахрайства ({labels.mean():.4%})")
    ax.set(xlabel="Recall / Повнота", ylabel="Precision / Точність", title="Validation: precision–recall")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output / "validation_pr.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(fpr, tpr, linewidth=1.8, label=f"ROC-AUC = {auc:.6f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Випадкова модель")
    ax.set(xlabel="False positive rate (FPR)", ylabel="True positive rate (TPR)",
           title="Validation: ROC")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "validation_roc.png", dpi=180)
    plt.close(fig)
    return auc


def extract_error_rows(scores, labels, rows, raw_amount, threshold: float) -> pd.DataFrame:
    predicted = scores >= threshold
    
    records = []
    for error_kind, mask in (
        ("FP", predicted & (labels == 0)),
        ("FN", ~predicted & (labels == 1)),
    ):
        matches = np.flatnonzero(mask)
        first_by_csv = sorted(matches, key=lambda i: int(rows[i]))[:5]
        for i in first_by_csv:
            s = float(scores[i])
            records.append({"csv_row_0_based": int(rows[i]), "error": error_kind,
                            "label": int(labels[i]), "score": s,
                            "Amount": float(raw_amount[i]),
                            "distance_to_threshold": (abs(s - threshold)
                                                      if np.isfinite(threshold) else np.nan)})
    return pd.DataFrame(records, columns=["csv_row_0_based", "error", "label", "score",
                                           "Amount", "distance_to_threshold"])


def table_errors_md(errors: pd.DataFrame) -> str:
    lines = ["| Рядок CSV (від 0) | Помилка | Мітка | Оцінка s | Сирий Amount | Відстань до t* |",
             "|---:|---|---:|---:|---:|---:|"]
    for r in errors.to_dict(orient="records"):
        dist = "—" if pd.isna(r["distance_to_threshold"]) else f"{r['distance_to_threshold']:.9g}"
        lines.append(f"| {r['csv_row_0_based']} | {r['error']} | {r['label']} | "
                     f"{r['score']:.9g} | {r['Amount']:.9g} | {dist} |")
    if len(errors) == 0:
        lines.append("| — | Немає помилок | — | — | — | — |")
    return "\n".join(lines)


def build_results_block(meta: dict, val_summary: pd.DataFrame, auc: float,
                        checks: pd.DataFrame, test: dict, errors: pd.DataFrame,
                        baseline_test: dict) -> str:
    stats = meta["splits"]
    lines = [
        "### Фактичні результати експерименту",
        "",
        "#### Розміри підвибірок",
        "",
        "| Вибірка | N | Шахрайських | Частка |",
        "|---|---:|---:|---:|",
    ]
    for name in ("train", "validation", "test"):
        item = stats[name]
        lines.append(f"| {name} | {item['n']} | {item['fraud']} | {item['fraud_rate']:.6%} |")
    lines += [
        "", f"Фінальна середня train-BCE після 12 епох: **{meta['final_train_bce']:.9g}**.",
        "", "#### Validation: якість ранжування та пороги", "",
        f"ROC-AUC на validation: **{auc:.8f}**.", "",
        "![PR-крива](results/validation_pr.png)", "",
        "![ROC-крива](results/validation_roc.png)", "",
        format_summary_table(val_summary), "",
        "Повну таблицю всіх кандидатів див. у `results/threshold_candidates_validation.csv`.",
        "", "#### Перевірка трьох контрольних наборів", "",
        "| Набір | t* | TP | FP | FN | TN | C | Точний збіг |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in checks.to_dict(orient="records"):
        lines.append(f"| {r['case']} | {display_threshold(r['threshold'])} | {r['TP']} | "
                     f"{r['FP']} | {r['FN']} | {r['TN']} | {r['C']} | так |")
    lines += [
        "", "Набір **A** перевіряє групування рівних оцінок 0,7; ",
        "набір **B** — випадок без позитивних міток, коли оптимальним є +∞; ",
        "набір **C** — нічию C(0,8)=C(0,5)=10, після якої обираємо найбільший поріг 0,8.",
        "", "Повні таблиці перевірки: `results/verification/check_A.csv`, "
            "`check_B.csv`, `check_C.csv`.",
        "", "#### Test: фінальна оцінка (без підбору порога)", "",
        f"Фіксований на validation поріг **t* = {display_threshold(test['threshold'])}**.",
        "", "| Поріг | TP | FP | FN | TN | C | Precision | Recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| 0,5 (для порівняння) | {baseline_test['TP']} | {baseline_test['FP']} | "
        f"{baseline_test['FN']} | {baseline_test['TN']} | {baseline_test['C']} | "
        f"{metric_text(baseline_test['precision'])} | {metric_text(baseline_test['recall'])} |",
        f"| t* = {display_threshold(test['threshold'])} | {test['TP']} | {test['FP']} | "
        f"{test['FN']} | {test['TN']} | {test['C']} | "
        f"{metric_text(test['precision'])} | {metric_text(test['recall'])} |",
        "", "Матриця плутанини (рядки — справжні 0,1; стовпці — передбачені 0,1):", "",
        "```text", f"             pred=0   pred=1", f"true=0       {test['TN']:6d}   {test['FP']:6d}",
        f"true=1       {test['FN']:6d}   {test['TP']:6d}", "```", "",
        "#### Перші п’ять FP і FN за номером оригінального рядка CSV", "",
        table_errors_md(errors), "",
    ]
    
    for kind in ("FP", "FN"):
        sub = errors[errors["error"] == kind]
        if len(sub):
            amount_min = float(sub["Amount"].min())
            amount_med = float(sub["Amount"].median())
            amount_max = float(sub["Amount"].max())
            valid_dist = sub["distance_to_threshold"].dropna()
            dist_text = (f"медіанна |s−t*|={valid_dist.median():.9g}"
                         if len(valid_dist) else "відстань до +∞ не визначена")
            lines.append(f"Для {kind} (показані {len(sub)} прикладів): сирий Amount у діапазоні "
                         f"[{amount_min:.9g}; {amount_max:.9g}], медіана {amount_med:.9g}; {dist_text}.")
        else:
            lines.append(f"Для {kind} немає прикладів, отже порівнювати Amount та відстані неможливо.")
        lines.append("")
    lines += [
        "**Інтерпретація:** вибір t* змінює лише правило прийняття рішення, "
        "не ваги нейромережі й не її оцінки s. На test різниця між t* і 0,5 становить "
        f"ΔFP = {test['FP']-baseline_test['FP']:+d}, "
        f"ΔFN = {test['FN']-baseline_test['FN']:+d}, "
        f"ΔC = {test['C']-baseline_test['C']:+d} одиниць.",
        "", "Мінімум C на validation не гарантує такого самого виграшу на test: "
        "це різні скінченні вибірки, і поріг було підібрано за випадковими особливостями validation.",
        "", "Десять (або менше) прикладів FP/FN недостатні, щоб встановити причину помилок: "
        "це мала невипадкова підбірка, а V1–V28 анонімізовані; з Amount і відстані до порога "
        "причинність не випливає.",
        "", "Якщо TP+FP=0, precision не визначена, бо модель нічого не позначила як шахрайство; "
        "якщо TP+FN=0, recall не визначена, бо у вибірці не було жодного реального шахрайства.",
    ]
    return "\n".join(lines)


def update_readme(path: Path, block: str):
    if not path.exists():
        raise FileNotFoundError(f"Немає README-шаблона: {path}")
    src = path.read_text(encoding="utf-8")
    if src.count(START) != 1 or src.count(END) != 1:
        raise ValueError("README має містити одну пару маркерів RESULTS")
    front, rest = src.split(START, 1)
    _, tail = rest.split(END, 1)
    path.write_text(front + START + "\n" + block + "\n" + END + tail,
                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("results"),
                        help="Каталог зі збереженим predictions.npz")
    args = parser.parse_args()
    folder = args.artifacts
    pred_file = folder / "predictions.npz"
    if not pred_file.exists():
        raise FileNotFoundError(f"Не знайдено {pred_file}. Спочатку виконайте train.py з локальним CSV.")
    predictions = np.load(pred_file, allow_pickle=False)
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    val_s = predictions["validation_scores"]
    val_y = predictions["validation_labels"]
    test_s = predictions["test_scores"]
    test_y = predictions["test_labels"]
    test_rows = predictions["test_rows"]
    test_amount = predictions["test_amount"]
    if len(val_s) != metadata["splits"]["validation"]["n"] or len(test_s) != metadata["splits"]["test"]["n"]:
        raise ValueError("Розмір збережених прогнозів не узгоджено з метаданими")
    if (np.diff(test_rows) <= 0).any():
        raise ValueError("Рядки test не відсортовані за вихідним CSV")

    # Пункт 2. Не повторюємо навчання. Використовуємо ТІЛЬКИ validation.
    candidates = all_thresholds_fast(val_s, val_y)
    candidates.to_csv(folder / "threshold_candidates_validation.csv",
                      index=False, float_format="%.17g")
    opt = select_optimal_threshold(candidates)
    tstar = opt["threshold"]
    standard = count_at_threshold(val_s, val_y, 0.5)
    negative = count_at_threshold(val_s, val_y, float("inf"))
    summary_val = pd.DataFrame([
        {"rule": "Стандартний 0,5", **standard},
        {"rule": "Мінімум вартості", **opt},
        {"rule": "Усі негативні", **negative},
    ])
    summary_val.to_csv(folder / "validation_summary.csv", index=False, float_format="%.17g")
    (folder / "selected_threshold.json").write_text(json.dumps({
        "threshold": json_threshold(tstar), "min_validation_cost": opt["C"],
        "tie_break": "largest_threshold",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    auc = save_curves(val_y, val_s, folder)

    # Пункт 3. Тільки після проходження ВСІХ незалежних перевірок допускаємо test.
    checks = run_checks(folder / "verification")
    if len(checks) != 3 or not checks["exact_match"].all():
        raise AssertionError("Перевірки A/B/C мають бути успішними до test")

    # Пункт 4. t* зафіксований, не змінюється за test.
    test_stats = enhanced_counts(count_at_threshold(test_s, test_y, tstar))
    baseline_test = enhanced_counts(count_at_threshold(test_s, test_y, 0.5))
    errors = extract_error_rows(test_s, test_y, test_rows, test_amount, tstar)
    errors.to_csv(folder / "test_errors.csv", index=False, float_format="%.17g")
    (folder / "test_evaluation.json").write_text(json.dumps({
        "tstar": {"threshold": json_threshold(tstar), **{k: v for k,v in test_stats.items() if k != "threshold"}},
        "default_0_5": {k: v for k,v in baseline_test.items()},
        "validation_roc_auc": auc,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_test = pd.DataFrame([
        {"rule": "Стандартний 0,5", **baseline_test},
        {"rule": "Фіксований t*", **test_stats},
    ])
    summary_test.to_csv(folder / "test_summary.csv", index=False, float_format="%.17g")

    readme = Path(__file__).resolve().parent / "README.md"
    update_readme(readme, build_results_block(metadata, summary_val, auc, checks,
                                              test_stats, errors, baseline_test))
    print(f"VALIDATION: ROC-AUC={auc:.8f}; t*={display_threshold(tstar)}; C*={opt['C']}")
    print("Перевірки A/B/C пройдено. TEST:")
    print(f"TP={test_stats['TP']}, FP={test_stats['FP']}, FN={test_stats['FN']}, "
          f"TN={test_stats['TN']}, C={test_stats['C']}; "
          f"precision={metric_text(test_stats['precision'])}, recall={metric_text(test_stats['recall'])}")
    print(f"Графіки, CSV, таблиця помилок та оновлений README збережено у {folder}.")


if __name__ == "__main__":
    main()
