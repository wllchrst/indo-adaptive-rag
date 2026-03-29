#!/usr/bin/env python3
"""Generate summary statistics from human evaluation labeling results."""

import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent

LABEL_ORDER = ["A", "B", "C"]

COLORS = {
    "header": "\033[1;36m",
    "subheader": "\033[1;33m",
    "correct": "\033[32m",
    "incorrect": "\033[31m",
    "reset": "\033[0m",
    "bold": "\033[1m",
}


def c(text, color_key):
    return f"{COLORS[color_key]}{text}{COLORS['reset']}"


def load_harry():
    df = pd.read_csv(BASE_DIR / "harry.csv", dtype=str)
    df = df.dropna(subset=["ID"]).copy()

    df["question_id"] = df["ID"].astype(str)
    df["auto_label"] = df["Label"].str.strip().str.upper()
    df["human_label"] = df["Your Label"].str.strip().str.upper()
    df["justification"] = df["Justification"].fillna("").astype(str)

    def derive_correctness(row):
        if row["human_label"] == "?":
            return "ambiguous"
        if row["auto_label"] == row["human_label"]:
            return "correct"
        return "incorrect"

    df["correctness"] = df.apply(derive_correctness, axis=1)
    return df[["question_id", "auto_label", "human_label", "correctness", "justification"]].copy()


def load_dave():
    df = pd.read_excel(BASE_DIR / "dave.xlsx", dtype=str)
    df = df.dropna(subset=["Question ID"]).copy()

    df["question_id"] = df["Question ID"].astype(str)
    df["auto_label"] = df["Automated Label"].str.strip().str.upper()
    df["human_label"] = df["My Label"].str.strip().str.upper()
    df["justification"] = df["Justification"].fillna("").astype(str)
    df["correctness"] = df["Label Correctness"].str.strip().str.lower()
    df.loc[df["correctness"] == "ambiguous", "human_label"] = "AMBIGUOUS"
    return df[["question_id", "auto_label", "human_label", "correctness", "justification"]].copy()


def load_garent():
    df = pd.read_excel(
        BASE_DIR / "garent.xlsx", sheet_name="Evaluation Garent", dtype=str
    )
    df = df.dropna(subset=["Question Id"]).copy()

    df["question_id"] = df["Question Id"].astype(str)
    df["auto_label"] = df["Automated Label"].str.strip().str.upper()
    df["human_label"] = df["My Label"].str.strip().str.upper()
    df["justification"] = df["Justification"].fillna("").astype(str)
    df["correctness"] = df["Label Correctness"].str.strip().str.lower()
    return df[["question_id", "auto_label", "human_label", "correctness", "justification"]].copy()


def print_section(title):
    print()
    print(c("=" * 70, "header"))
    print(c(f"  {title}", "header"))
    print(c("=" * 70, "header"))


def print_subsection(title):
    print()
    print(c(f"--- {title} ---", "subheader"))


def compute_per_evaluator_stats(name, df):
    print_section(f"Evaluator: {name}")

    total = len(df)
    non_ambiguous = df[df["correctness"] != "ambiguous"]
    ambiguous_count = len(df) - len(non_ambiguous)
    correct = len(non_ambiguous[non_ambiguous["correctness"] == "correct"])
    incorrect = len(non_ambiguous[non_ambiguous["correctness"] == "incorrect"])
    accuracy = correct / len(non_ambiguous) * 100 if len(non_ambiguous) > 0 else 0

    print(f"\n  Total evaluated:  {total}")
    print(f"  Correct:          {c(str(correct), 'correct')}")
    print(f"  Incorrect:        {c(str(incorrect), 'incorrect')}")
    print(f"  Ambiguous:        {ambiguous_count}")
    print(f"  Accuracy:         {accuracy:.1f}%")

    print_subsection("Label Distribution")
    auto_dist = df["auto_label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    human_dist = (
        df[df["human_label"].isin(LABEL_ORDER)]["human_label"]
        .value_counts()
        .reindex(LABEL_ORDER, fill_value=0)
    )

    print(f"\n  {'Label':<10} {'Automated':>12} {'Human':>12} {'Shift':>12}")
    print(f"  {'-'*46}")
    for label in LABEL_ORDER:
        shift = human_dist[label] - auto_dist[label]
        shift_str = f"{shift:+d}" if shift != 0 else "0"
        print(f"  {label:<10} {auto_dist[label]:>12} {human_dist[label]:>12} {shift_str:>12}")

    eval_df = df[df["human_label"].isin(LABEL_ORDER) & df["auto_label"].isin(LABEL_ORDER)]
    if len(eval_df) == 0:
        return

    print_subsection("Confusion Matrix (rows=Automated, cols=Human)")
    y_true = eval_df["auto_label"].values
    y_pred = eval_df["human_label"].values
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)

    header = f"  {'':>10}"
    for label in LABEL_ORDER:
        header += f" {label:>8}"
    print(header)
    for i, label in enumerate(LABEL_ORDER):
        row = f"  {label:>10}"
        for j in range(len(LABEL_ORDER)):
            val = cm[i][j]
            if i == j:
                row += f" {c(str(val), 'correct'):>8}"
            else:
                row += f" {c(str(val), 'incorrect'):>8}"
        print(row)

    print_subsection("Misclassification Patterns (Automated -> Human)")
    misclassified = eval_df[eval_df["auto_label"] != eval_df["human_label"]]
    if len(misclassified) > 0:
        patterns = (
            misclassified.groupby(["auto_label", "human_label"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        for _, row in patterns.iterrows():
            print(f"  {row['auto_label']} -> {row['human_label']}: {row['count']}")
    else:
        print("  None")

    print_subsection("Per-Class Accuracy (Human as ground truth)")
    print()
    report = classification_report(
        y_true, y_pred, labels=LABEL_ORDER, zero_division=0
    )
    for line in report.strip().split("\n"):
        print(f"  {line}")

    return accuracy, non_ambiguous, correct, incorrect, ambiguous_count


def compute_cross_evaluator_agreement(evaluators):
    print_section("Cross-Evaluator Agreement")

    eval_names = list(evaluators.keys())
    eval_dfs = {
        name: df.set_index("question_id")["human_label"]
        for name, df in evaluators.items()
    }

    pairs = []
    for i in range(len(eval_names)):
        for j in range(i + 1, len(eval_names)):
            pairs.append((eval_names[i], eval_names[j]))

    for name1, name2 in pairs:
        common_ids = eval_dfs[name1].index.intersection(eval_dfs[name2].index)
        valid_mask_1 = eval_dfs[name1].loc[common_ids].isin(LABEL_ORDER)
        valid_mask_2 = eval_dfs[name2].loc[common_ids].isin(LABEL_ORDER)
        valid_mask = valid_mask_1 & valid_mask_2
        common_ids = common_ids[valid_mask]

        if len(common_ids) == 0:
            print(f"\n  {name1} vs {name2}: No overlapping questions")
            continue

        labels1 = eval_dfs[name1].loc[common_ids].values
        labels2 = eval_dfs[name2].loc[common_ids].values

        agreement = np.mean(labels1 == labels2) * 100
        kappa = cohen_kappa_score(labels1, labels2, labels=LABEL_ORDER)

        print(f"\n  {name1} vs {name2}:")
        print(f"    Overlapping questions: {len(common_ids)}")
        print(f"    Agreement:             {agreement:.1f}%")
        print(f"    Cohen's Kappa:         {kappa:.3f}")

        cm = confusion_matrix(labels1, labels2, labels=LABEL_ORDER)
        header = f"    {'':>8}"
        for label in LABEL_ORDER:
            header += f" {label:>6}({name2})"
        print(header)
        for i, label in enumerate(LABEL_ORDER):
            row = f"    {label:>6}({name1})"
            for j in range(len(LABEL_ORDER)):
                row += f" {cm[i][j]:>13}"
            print(row)


def compute_overall_summary(stats, evaluators):
    print_section("Overall Combined Summary")

    accuracies = [s[0] for s in stats.values()]
    avg_accuracy = np.mean(accuracies)

    print(f"\n  Evaluators:           {len(stats)}")
    print(f"  Questions/evaluator:  {len(list(stats.values())[0][1])}")
    print(f"  Average accuracy:     {avg_accuracy:.1f}%")
    print(f"  Accuracy range:       {min(accuracies):.1f}% - {max(accuracies):.1f}%")

    print_subsection("Combined Label Distribution (All Evaluators)")

    all_auto = []
    all_human = []
    for name, df in evaluators.items():
        valid = df[df["human_label"].isin(LABEL_ORDER) & df["auto_label"].isin(LABEL_ORDER)]
        all_auto.extend(valid["auto_label"].tolist())
        all_human.extend(valid["human_label"].tolist())

    all_auto_series = pd.Series(all_auto).value_counts().reindex(LABEL_ORDER, fill_value=0)
    all_human_series = pd.Series(all_human).value_counts().reindex(LABEL_ORDER, fill_value=0)

    print(f"\n  {'Label':<10} {'Automated':>12} {'Human':>12} {'Shift':>12}")
    print(f"  {'-'*46}")
    for label in LABEL_ORDER:
        shift = all_human_series[label] - all_auto_series[label]
        shift_str = f"{shift:+d}" if shift != 0 else "0"
        print(f"  {label:<10} {all_auto_series[label]:>12} {all_human_series[label]:>12} {shift_str:>12}")

    print_subsection("Combined Confusion Matrix (rows=Automated, cols=Human)")
    cm = confusion_matrix(all_auto, all_human, labels=LABEL_ORDER)
    header = f"  {'':>10}"
    for label in LABEL_ORDER:
        header += f" {label:>8}"
    print(header)
    for i, label in enumerate(LABEL_ORDER):
        row = f"  {label:>10}"
        for j in range(len(LABEL_ORDER)):
            val = cm[i][j]
            if i == j:
                row += f" {c(str(val), 'correct'):>8}"
            else:
                row += f" {c(str(val), 'incorrect'):>8}"
        print(row)

    print_subsection("Top Misclassification Patterns (Combined)")
    misclassified_data = []
    for name, df in evaluators.items():
        valid = df[df["human_label"].isin(LABEL_ORDER) & df["auto_label"].isin(LABEL_ORDER)]
        mis = valid[valid["auto_label"] != valid["human_label"]]
        misclassified_data.append(mis)
    all_mis = pd.concat(misclassified_data)

    if len(all_mis) > 0:
        patterns = (
            all_mis.groupby(["auto_label", "human_label"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        for _, row in patterns.iterrows():
            print(f"  {row['auto_label']} -> {row['human_label']}: {row['count']}")


def save_summary_csv(evaluators, stats, output_path):
    rows = []
    for name, df in evaluators.items():
        valid = df[df["human_label"].isin(LABEL_ORDER) & df["auto_label"].isin(LABEL_ORDER)]
        non_amb = df[df["correctness"] != "ambiguous"]
        correct = len(non_amb[non_amb["correctness"] == "correct"])
        incorrect = len(non_amb[non_amb["correctness"] == "incorrect"])
        ambiguous = len(df) - len(non_amb)
        accuracy = correct / len(non_amb) * 100 if len(non_amb) > 0 else 0

        rows.append({
            "evaluator": name,
            "total_questions": len(df),
            "correct": correct,
            "incorrect": incorrect,
            "ambiguous": ambiguous,
            "accuracy_pct": round(accuracy, 1),
        })

    summary_df = pd.DataFrame(rows)
    avg_row = {
        "evaluator": "AVERAGE",
        "total_questions": summary_df["total_questions"].sum(),
        "correct": summary_df["correct"].sum(),
        "incorrect": summary_df["incorrect"].sum(),
        "ambiguous": summary_df["ambiguous"].sum(),
        "accuracy_pct": round(summary_df["accuracy_pct"].mean(), 1),
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([avg_row])], ignore_index=True)
    summary_df.to_csv(output_path, index=False)
    print(f"\n  Summary saved to: {output_path}")


if __name__ == "__main__":
    print(c("\n  Evaluation Labeling Summary Report", "header"))
    print(c("  " + "=" * 40 + "\n", "header"))

    harry = load_harry()
    dave = load_dave()
    garent = load_garent()

    evaluators = {"Harry": harry, "Dave": dave, "Garent": garent}

    stats = {}
    for name, df in evaluators.items():
        result = compute_per_evaluator_stats(name, df)
        if result:
            stats[name] = result

    compute_cross_evaluator_agreement(evaluators)
    compute_overall_summary(stats, evaluators)

    csv_path = BASE_DIR / "summary.csv"
    save_summary_csv(evaluators, stats, csv_path)

    print()
    print(c("=" * 70, "header"))
    print(c("  Done.", "header"))
    print(c("=" * 70, "header"))
    print()
