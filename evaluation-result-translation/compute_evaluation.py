import pandas as pd
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVALUATORS = ["CO", "EZ", "LY"]
METRICS = [
    "q_semantic_score",
    "q_entity_translation_score",
    "a_semantic_score",
    "a_entity_translation_score",
]
METRIC_LABELS = {
    "q_semantic_score": "Question Semantic",
    "q_entity_translation_score": "Question Entity Translation",
    "a_semantic_score": "Answer Semantic",
    "a_entity_translation_score": "Answer Entity Translation",
}

dfs = []
for name in EVALUATORS:
    path = os.path.join(SCRIPT_DIR, f"{name}.xlsx")
    df = pd.read_excel(path)
    df["evaluator"] = name
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
combined[METRICS] = combined[METRICS].astype(int)

combined.to_csv(os.path.join(SCRIPT_DIR, "combined_evaluation.csv"), index=False)

stats_funcs = ["count", "mean", "std", "median", "min", "max"]
overall_stats = combined[METRICS].agg(stats_funcs)
evaluator_stats = combined.groupby("evaluator")[METRICS].agg(stats_funcs)

all_stats = pd.concat([overall_stats, evaluator_stats])
all_stats.index.name = "evaluator"
all_stats.to_csv(os.path.join(SCRIPT_DIR, "evaluation_summary.csv"))

print("=" * 70)
print("HUMAN EVALUATION RESULTS — Hotpot Translation Quality")
print("=" * 70)
print(f"Total evaluations: {len(combined)}")
print(f"Evaluators: {EVALUATORS}")
print(f"Samples per evaluator: {len(combined) // len(EVALUATORS)}")
print()

print("-" * 70)
print("OVERALL DESCRIPTIVE STATISTICS")
print("-" * 70)
print(overall_stats.to_string(float_format="%.2f"))
print()

print("-" * 70)
print("PER-EVALUATOR DESCRIPTIVE STATISTICS")
print("-" * 70)
for name in EVALUATORS:
    print(f"\n  [{name}]")
    ev = combined[combined["evaluator"] == name][METRICS]
    print(ev.describe().loc[["count", "mean", "std", "50%", "min", "max"]].to_string(float_format="%.2f"))

print()
print("-" * 70)
print("SCORE DISTRIBUTION (Count | Percentage)")
print("-" * 70)
for metric in METRICS:
    print(f"\n  {METRIC_LABELS[metric]}")
    dist = combined[metric].value_counts().sort_index()
    for score in range(1, 6):
        count = dist.get(score, 0)
        pct = count / len(combined) * 100
        print(f"    Score {score}: {count:>4d} ({pct:5.1f}%)")

print()
print("-" * 70)
print("FILES SAVED")
print("-" * 70)
print(f"  combined_evaluation.csv  — {len(combined)} rows")
print(f"  evaluation_summary.csv   — overall + per-evaluator stats")
