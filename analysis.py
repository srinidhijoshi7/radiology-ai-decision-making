"""
analysis.py
-----------
Five empirical analyses, each saving a publication-quality PNG to figures/:
  1. ROC curves per pathology (figures/01_roc_per_pathology.png)
  2. Reliability / calibration diagram (figures/02_calibration.png)
  3. Human-AI agreement matrix (figures/03_agreement_matrix.png)
  4. Subgroup performance audit — sex (figures/04_subgroup_sex.png)
  5. Subgroup performance audit — age (figures/05_subgroup_age.png)

Plus a summary table  data/results_summary.csv  used by the website.

Run:  python src/analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, roc_auc_score, brier_score_loss,
                             cohen_kappa_score, confusion_matrix)

# allow `python src/analysis.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.load_data import load, pathology_columns, PATHOLOGIES
from src.plotting import apply_style, PALETTE

apply_style()

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decision(scores, threshold=0.5):
    return (scores >= threshold).astype(int)


def _calibration_bin(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(y_prob, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    centers, freqs, counts = [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            centers.append((bins[b] + bins[b + 1]) / 2)
            freqs.append(y_true[mask].mean())
            counts.append(mask.sum())
    return np.array(centers), np.array(freqs), np.array(counts)


# ---------------------------------------------------------------------------
# 1. ROC curves
# ---------------------------------------------------------------------------

def plot_roc(df: pd.DataFrame) -> dict:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], color=PALETTE["muted"], linestyle="--",
            linewidth=1, label="Chance")

    auc_results = {}
    cmap = plt.colormaps["viridis"].resampled(len(PATHOLOGIES))
    for i, pathology in enumerate(PATHOLOGIES):
        cols = pathology_columns(pathology)
        fpr, tpr, _ = roc_curve(df[cols["truth"]], df[cols["ai_score"]])
        auc = roc_auc_score(df[cols["truth"]], df[cols["ai_score"]])
        auc_results[pathology] = round(float(auc), 3)
        ax.plot(fpr, tpr, color=cmap(i), linewidth=2,
                label=f"{pathology} (AUC={auc:.3f})")

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("AI diagnostic discrimination across five pathologies")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.savefig(FIG_DIR / "01_roc_per_pathology.png")
    plt.close(fig)
    return auc_results


# ---------------------------------------------------------------------------
# 2. Calibration / reliability diagram
# ---------------------------------------------------------------------------

def plot_calibration(df: pd.DataFrame) -> dict:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], color=PALETTE["muted"], linestyle="--",
            linewidth=1, label="Perfect calibration")

    brier = {}
    cmap = plt.colormaps["plasma"].resampled(len(PATHOLOGIES))
    for i, pathology in enumerate(PATHOLOGIES):
        cols = pathology_columns(pathology)
        y, p = df[cols["truth"]].values, df[cols["ai_score"]].values
        centers, freqs, counts = _calibration_bin(y, p, n_bins=8)
        ax.plot(centers, freqs, "o-", color=cmap(i), markersize=6,
                linewidth=1.8, label=pathology)
        brier[pathology] = round(float(brier_score_loss(y, p)), 3)

    ax.set_xlabel("AI predicted probability (binned)")
    ax.set_ylabel("Empirical positive rate")
    ax.set_title("Calibration: are AI confidence scores trustworthy?")
    ax.legend(loc="upper left")
    fig.savefig(FIG_DIR / "02_calibration.png")
    plt.close(fig)
    return brier


# ---------------------------------------------------------------------------
# 3. Human-AI agreement
# ---------------------------------------------------------------------------

def plot_agreement(df: pd.DataFrame) -> dict:
    """For each pathology, confusion of independent radiologist vs binarised AI."""
    fig, axes = plt.subplots(1, len(PATHOLOGIES),
                             figsize=(4 * len(PATHOLOGIES), 4.2))
    kappa_scores = {}
    for ax, pathology in zip(axes, PATHOLOGIES):
        cols = pathology_columns(pathology)
        rad = df[cols["rad_indep"]].values
        ai  = _decision(df[cols["ai_score"]].values)
        cm  = confusion_matrix(rad, ai, labels=[0, 1])
        kappa = cohen_kappa_score(rad, ai)
        kappa_scores[pathology] = round(float(kappa), 3)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["AI: neg", "AI: pos"],
                    yticklabels=["Rad: neg", "Rad: pos"],
                    annot_kws={"fontsize": 12, "weight": "bold"})
        ax.set_title(f"{pathology}\nκ = {kappa:.2f}")
        ax.set_xlabel(""); ax.set_ylabel("")
    fig.suptitle("Human–AI agreement (independent radiologist vs binarised AI)",
                 fontsize=13, fontweight="bold", y=1.04)
    fig.savefig(FIG_DIR / "03_agreement_matrix.png")
    plt.close(fig)
    return kappa_scores


# ---------------------------------------------------------------------------
# 4 & 5. Subgroup audits
# ---------------------------------------------------------------------------

def _subgroup_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Per-subgroup AUC, TPR, FPR — averaged across the 5 pathologies."""
    rows = []
    for grp, sub in df.groupby(group_col):
        per_pathology = []
        for pathology in PATHOLOGIES:
            cols = pathology_columns(pathology)
            y, p = sub[cols["truth"]].values, sub[cols["ai_score"]].values
            if len(np.unique(y)) < 2:  # AUC undefined
                continue
            decision = _decision(p)
            tpr = decision[y == 1].mean() if (y == 1).any() else np.nan
            fpr = decision[y == 0].mean() if (y == 0).any() else np.nan
            per_pathology.append({
                "auc": roc_auc_score(y, p),
                "tpr": tpr,
                "fpr": fpr,
            })
        if per_pathology:
            agg = pd.DataFrame(per_pathology).mean()
            rows.append({"group": grp, **agg.to_dict()})
    return pd.DataFrame(rows)


def plot_subgroup(df, group_col, fname, title):
    tbl = _subgroup_table(df, group_col).sort_values("group")
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(tbl))
    w = 0.28
    ax.bar(x - w, tbl["auc"], width=w, color=PALETTE["primary"], label="AUC")
    ax.bar(x,     tbl["tpr"], width=w, color=PALETTE["accent"],  label="True-positive rate")
    ax.bar(x + w, tbl["fpr"], width=w, color=PALETTE["secondary"], label="False-positive rate")

    ax.set_xticks(x)
    ax.set_xticklabels(tbl["group"])
    ax.set_ylim(0, 1)
    ax.set_xlabel(group_col.replace("_", " ").title())
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="upper right")

    for xi, (a, t, f) in enumerate(zip(tbl["auc"], tbl["tpr"], tbl["fpr"])):
        ax.text(xi - w, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(xi,     t + 0.02, f"{t:.2f}", ha="center", fontsize=8)
        ax.text(xi + w, f + 0.02, f"{f:.2f}", ha="center", fontsize=8)

    fig.savefig(FIG_DIR / fname)
    plt.close(fig)
    return tbl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df, source = load()
    print(f"Loaded {len(df)} studies from {source} source.\n")

    print("→ Plot 1: ROC per pathology")
    aucs = plot_roc(df)

    print("→ Plot 2: Calibration / reliability diagram")
    briers = plot_calibration(df)

    print("→ Plot 3: Human–AI agreement matrices")
    kappas = plot_agreement(df)

    print("→ Plot 4: Subgroup audit by sex")
    by_sex = plot_subgroup(df, "sex", "04_subgroup_sex.png",
                           "AI performance by patient sex")

    print("→ Plot 5: Subgroup audit by age band")
    by_age = plot_subgroup(df, "age_band", "05_subgroup_age.png",
                           "AI performance by age band")

    # Master results dict for the website
    summary = {
        "source": source,
        "n_studies": int(len(df)),
        "auc": aucs,
        "brier": briers,
        "cohen_kappa": kappas,
        "by_sex": by_sex.to_dict(orient="records"),
        "by_age": by_age.to_dict(orient="records"),
    }
    out = Path("website/assets/results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary → {out}")

    print("\nResults summary:")
    print(f"  AUC mean       = {np.mean(list(aucs.values())):.3f}")
    print(f"  Brier mean     = {np.mean(list(briers.values())):.3f}")
    print(f"  Cohen κ mean   = {np.mean(list(kappas.values())):.3f}")
    print(f"  Sex TPR gap    = {by_sex['tpr'].max() - by_sex['tpr'].min():+.3f}")
    print(f"  Age TPR gap    = {by_age['tpr'].max() - by_age['tpr'].min():+.3f}")


if __name__ == "__main__":
    main()
