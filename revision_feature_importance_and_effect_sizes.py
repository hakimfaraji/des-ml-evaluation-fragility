"""Revision addendum v1.1 for the DES ML evaluation study.

This script documents the additional analyses added during peer-review revision:
permutation feature importance and paired ranking effect sizes. It is primarily intended
as an executable reference. The notebook Code_S1.ipynb contains the full pipeline.
"""


# ============================================================
# Revision addendum v1.1
# Additional analyses for revised manuscript/SI
# ============================================================

import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from scipy.stats import binomtest

REVISION_OUTPUT_DIR = os.environ.get("REVISION_OUTPUT_DIR", "revision_outputs")
os.makedirs(REVISION_OUTPUT_DIR, exist_ok=True)

# ---------- Helpers ----------
def assign_feature_family(feature):
    if feature.startswith("analyte_"):
        return "Analyte descriptor"
    if feature.startswith("hba_"):
        return "HBA descriptor"
    if feature.startswith("hbd_"):
        return "HBD descriptor"
    if feature.startswith("des_"):
        return "DES-level descriptor"
    if feature.startswith("pair_"):
        return "DES-analyte interaction"
    if feature.startswith("ratio_"):
        return "Composition / ratio"
    return "Other"


def bootstrap_mean_ci(values, n_boot=10000, random_state=42):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(random_state)
    boot = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    return np.quantile(boot, 0.025), np.quantile(boot, 0.975)


def compute_group_metrics(rank_df, rank_col):
    rows = []
    for (doi, analyte), g in rank_df.groupby(["doi", "analyte"], dropna=False):
        g = g.copy()
        n = len(g)
        opt_ranks = g.loc[g["is_optimal"].astype(int) == 1, rank_col].tolist()
        if len(opt_ranks) == 0:
            continue
        best_rank = min(opt_ranks)
        rows.append({
            "doi": doi,
            "analyte": analyte,
            "group_size": n,
            "Hit@1": int(best_rank == 1),
            "Hit@3": int(best_rank <= 3),
            "MRR": 1.0 / best_rank,
        })
    return pd.DataFrame(rows)


def summarize_scope(df_scope, scope_name):
    delta_hit1 = df_scope["Hit@1_hybrid"].values - df_scope["Hit@1_baseline"].values
    delta_mrr = df_scope["MRR_hybrid"].values - df_scope["MRR_baseline"].values
    better = int((delta_mrr > 0).sum())
    worse = int((delta_mrr < 0).sum())
    equal = int((delta_mrr == 0).sum())
    n_non_tied = better + worse
    p_value = binomtest(better, n=n_non_tied, p=0.5, alternative="two-sided").pvalue if n_non_tied > 0 else np.nan
    hit1_ci_low, hit1_ci_high = bootstrap_mean_ci(delta_hit1)
    mrr_ci_low, mrr_ci_high = bootstrap_mean_ci(delta_mrr)
    dz_mrr = float(np.mean(delta_mrr) / np.std(delta_mrr, ddof=1)) if len(delta_mrr) > 1 and np.std(delta_mrr, ddof=1) > 0 else np.nan
    return {
        "scope": scope_name,
        "n_groups": len(df_scope),
        "baseline_Hit@1_mean": df_scope["Hit@1_baseline"].mean(),
        "hybrid_Hit@1_mean": df_scope["Hit@1_hybrid"].mean(),
        "delta_Hit@1_hybrid_minus_baseline": np.mean(delta_hit1),
        "delta_Hit@1_bootstrap_CI_low": hit1_ci_low,
        "delta_Hit@1_bootstrap_CI_high": hit1_ci_high,
        "baseline_MRR_mean": df_scope["MRR_baseline"].mean(),
        "hybrid_MRR_mean": df_scope["MRR_hybrid"].mean(),
        "delta_MRR_hybrid_minus_baseline": np.mean(delta_mrr),
        "delta_MRR_bootstrap_CI_low": mrr_ci_low,
        "delta_MRR_bootstrap_CI_high": mrr_ci_high,
        "paired_standardized_effect_size_dz_MRR": dz_mrr,
        "hybrid_better_by_MRR": better,
        "hybrid_worse_by_MRR": worse,
        "equal_by_MRR": equal,
        "exact_sign_test_p_value_ties_excluded": p_value,
    }

# ---------- Permutation feature importance ----------
selected_features_path = os.path.join(OUTPUT_DIR, "train_selected_features_after_filtering.txt")
if not os.path.exists(selected_features_path):
    selected_features_path = "train_selected_features_after_filtering.txt"

with open(selected_features_path, "r", encoding="utf-8") as f:
    selected_features = [line.strip() for line in f if line.strip()]

X_train = df_train[selected_features].copy()
X_test = df_test[selected_features].copy()
y_train = df_train["is_optimal"].astype(int).values
y_test = df_test["is_optimal"].astype(int).values

full_model_for_importance = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        penalty="l2", C=1.0, class_weight="balanced",
        solver="lbfgs", max_iter=5000, random_state=RANDOM_STATE
    )),
])
full_model_for_importance.fit(X_train, y_train)

perm_auc = permutation_importance(
    full_model_for_importance, X_test, y_test,
    scoring="roc_auc", n_repeats=50, random_state=RANDOM_STATE, n_jobs=1
)
perm_bacc = permutation_importance(
    full_model_for_importance, X_test, y_test,
    scoring="balanced_accuracy", n_repeats=50, random_state=RANDOM_STATE, n_jobs=1
)

coef = full_model_for_importance.named_steps["clf"].coef_[0]
feature_importance_df = pd.DataFrame({
    "feature": selected_features,
    "permutation_auc_mean_decrease": perm_auc.importances_mean,
    "permutation_auc_sd": perm_auc.importances_std,
    "permutation_balanced_accuracy_mean_decrease": perm_bacc.importances_mean,
    "permutation_balanced_accuracy_sd": perm_bacc.importances_std,
    "standardized_logistic_coefficient": coef,
    "abs_standardized_logistic_coefficient": np.abs(coef),
})
feature_importance_df["feature_family"] = feature_importance_df["feature"].apply(assign_feature_family)
feature_importance_df = feature_importance_df.sort_values("permutation_auc_mean_decrease", ascending=False).reset_index(drop=True)
feature_importance_df.to_csv(os.path.join(REVISION_OUTPUT_DIR, "feature_importance_permutation_full_model.csv"), index=False)
feature_importance_df.head(20).to_csv(os.path.join(REVISION_OUTPUT_DIR, "feature_importance_top20.csv"), index=False)

family_summary = (
    feature_importance_df.groupby("feature_family", dropna=False)
    .agg(
        n_features=("feature", "count"),
        mean_auc_decrease=("permutation_auc_mean_decrease", "mean"),
        max_auc_decrease=("permutation_auc_mean_decrease", "max"),
        n_positive_auc_importance=("permutation_auc_mean_decrease", lambda x: int((x > 0).sum())),
    )
    .reset_index()
    .sort_values("mean_auc_decrease", ascending=False)
)
family_summary.to_csv(os.path.join(REVISION_OUTPUT_DIR, "feature_importance_family_summary.csv"), index=False)

# ---------- Paired ranking effect sizes ----------
baseline_rank_df = pd.read_csv(os.path.join(OUTPUT_DIR, "baseline_group_ranking_results.csv"))
hybrid_rank_df = pd.read_csv(os.path.join(OUTPUT_DIR, "hybrid_group_ranking_results.csv"))
base_group = compute_group_metrics(baseline_rank_df, "rank_baseline")
hyb_group = compute_group_metrics(hybrid_rank_df, "rank_hybrid")
paired = base_group.merge(hyb_group, on=["doi", "analyte", "group_size"], suffixes=("_baseline", "_hybrid"))

effect_rows = [
    summarize_scope(paired, "all_groups"),
    summarize_scope(paired.loc[paired["group_size"] >= 2], "comparable_ge_2"),
    summarize_scope(paired.loc[paired["group_size"] >= 3], "comparable_ge_3"),
]
effect_df = pd.DataFrame(effect_rows)
effect_df.to_csv(os.path.join(REVISION_OUTPUT_DIR, "paired_ranking_effect_sizes.csv"), index=False)

main_table4 = effect_df[[
    "scope", "n_groups", "baseline_Hit@1_mean", "hybrid_Hit@1_mean",
    "baseline_MRR_mean", "hybrid_MRR_mean", "delta_MRR_hybrid_minus_baseline",
    "hybrid_better_by_MRR", "hybrid_worse_by_MRR", "equal_by_MRR",
    "exact_sign_test_p_value_ties_excluded",
]].copy()
main_table4.to_csv(os.path.join(REVISION_OUTPUT_DIR, "main_table4_revised_group_ranking.csv"), index=False)

print("Revision addendum outputs written to:", REVISION_OUTPUT_DIR)
print("Top feature importance rows:")
print(feature_importance_df.head(10).to_string(index=False))
print("\nPaired effect-size summary:")
print(effect_df.to_string(index=False))
