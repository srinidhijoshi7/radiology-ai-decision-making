"""
anchoring_simulation.py
-----------------------
The behavioural-decision centrepiece. We model how a clinician's diagnostic
decision changes under three protocols:

    A. INDEPENDENT      – clinician reads the X-ray alone.
    B. AI_FIRST         – AI prediction is shown BEFORE the clinician decides
                          (anchoring condition).
    C. AI_AFTER         – clinician decides first, then sees AI as a check.

We use a Bayesian log-odds framework adapted from Gaube et al. (2021) and
Tschandl et al. (2020):

    log-odds(post) = w_clin * log-odds(clinician prior)
                   + w_ai   * log-odds(AI signal)

  * w_clin = 1, w_ai = 0   (independent)
  * w_clin = 0.55, w_ai = 0.45  (AI-first / anchored — Gaube reports a
    ~0.45 effective deference weight when AI is shown first)
  * w_clin = 0.85, w_ai = 0.15  (AI-after — clinician already committed,
    smaller adjustment)

Outputs:
    figures/06_anchoring_three_protocols.png  – grouped bar chart
    figures/07_anchoring_when_ai_wrong.png    – shows the cost of anchoring
                                                 when the AI is wrong
    figures/08_anchoring_sensitivity.png      – sensitivity to anchoring
                                                 weight (key chart for ILO1)
    website/assets/simulator_grid.json        – pre-computed grid used by the
                                                 interactive web simulator

Run:  python src/anchoring_simulation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.load_data import load, pathology_columns, PATHOLOGIES
from src.plotting import apply_style, PALETTE

apply_style()

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Bayesian belief-update model
# ---------------------------------------------------------------------------

def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def clinician_prior_prob(true_label: np.ndarray, base_acc: float, rng) -> np.ndarray:
    """
    Returns the clinician's calibrated probability of the *positive* class.
      * If the clinician would call the case correctly (prob = base_acc),
        their P(positive) is high on positives and low on negatives.
      * If they'd miss it, the inverse.
      * Confidence ~ Beta(8, 2) (mean ≈ 0.80) on whichever way they lean.
    """
    n = len(true_label)
    correct    = rng.random(n) < base_acc
    confidence = rng.beta(8, 2, size=n)        # confidence on whichever call they make

    # On positives:  correct → high P(pos);  wrong → low P(pos)
    # On negatives:  correct → low P(pos);   wrong → high P(pos)
    p_positive = np.where(
        true_label == 1,
        np.where(correct, confidence, 1 - confidence),
        np.where(correct, 1 - confidence, confidence),
    )
    return p_positive


def update(prior_prob: np.ndarray, ai_prob: np.ndarray,
           w_clin: float, w_ai: float) -> np.ndarray:
    """Log-odds blend; returns posterior probability of positive class."""
    post_logit = w_clin * _logit(prior_prob) + w_ai * _logit(ai_prob)
    return _sigmoid(post_logit)


# ---------------------------------------------------------------------------
# Protocol simulator
# ---------------------------------------------------------------------------

PROTOCOLS = {
    "Independent":     dict(w_clin=1.00, w_ai=0.00),
    "AI shown first":  dict(w_clin=0.55, w_ai=0.45),  # anchored
    "AI shown after":  dict(w_clin=0.85, w_ai=0.15),  # adjustment phase
}


def simulate_dataset(df: pd.DataFrame, base_acc: float = 0.86,
                     seed: int = 7) -> pd.DataFrame:
    """
    For every (study, pathology) pair, simulate the clinician's decision under
    each protocol and return a long-form results table.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for pathology in PATHOLOGIES:
        cols = pathology_columns(pathology)
        truth = df[cols["truth"]].values
        ai_p  = df[cols["ai_score"]].values
        prior = clinician_prior_prob(truth, base_acc, rng)
        ai_correct = ((ai_p >= 0.5).astype(int) == truth)

        for proto_name, params in PROTOCOLS.items():
            posterior = update(prior, ai_p, **params)
            decision  = (posterior >= 0.5).astype(int)
            for i in range(len(df)):
                rows.append({
                    "pathology":  pathology,
                    "protocol":   proto_name,
                    "truth":      int(truth[i]),
                    "ai_correct": bool(ai_correct[i]),
                    "decision":   int(decision[i]),
                    "correct":    int(decision[i] == truth[i]),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot 6 – overall accuracy across the three protocols
# ---------------------------------------------------------------------------

def plot_three_protocols(results: pd.DataFrame):
    agg = results.groupby(["pathology", "protocol"])["correct"].mean().unstack()
    agg = agg[list(PROTOCOLS)]                                       # column order

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(agg))
    w = 0.26
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"]]

    for i, proto in enumerate(agg.columns):
        ax.bar(x + (i - 1) * w, agg[proto], width=w, color=colors[i],
               label=proto, edgecolor="white", linewidth=0.5)
        for xi, v in zip(x + (i - 1) * w, agg[proto]):
            ax.text(xi, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(agg.index, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Diagnostic accuracy")
    ax.set_title("Clinician accuracy under three AI presentation protocols")
    ax.legend(loc="lower right")
    fig.savefig(FIG_DIR / "06_anchoring_three_protocols.png")
    plt.close(fig)
    return agg


# ---------------------------------------------------------------------------
# Plot 7 – the cost of anchoring when the AI is wrong
# ---------------------------------------------------------------------------

def plot_when_ai_wrong(results: pd.DataFrame):
    agg = (results.groupby(["protocol", "ai_correct"])["correct"].mean()
           .unstack().rename(columns={True: "AI correct", False: "AI wrong"}))
    agg = agg.loc[list(PROTOCOLS)]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(agg))
    w = 0.35
    ax.bar(x - w/2, agg["AI correct"], w, color=PALETTE["primary"],
           label="When AI is correct")
    ax.bar(x + w/2, agg["AI wrong"],   w, color=PALETTE["secondary"],
           label="When AI is wrong")

    for i, p in enumerate(agg.index):
        ax.text(i - w/2, agg.loc[p, "AI correct"] + 0.01,
                f"{agg.loc[p, 'AI correct']:.2f}", ha="center", fontsize=9)
        ax.text(i + w/2, agg.loc[p, "AI wrong"] + 0.01,
                f"{agg.loc[p, 'AI wrong']:.2f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(agg.index)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Clinician accuracy")
    ax.set_title("The price of anchoring: clinician accuracy split by whether AI was correct")
    ax.legend(loc="lower right")
    fig.savefig(FIG_DIR / "07_anchoring_when_ai_wrong.png")
    plt.close(fig)
    return agg


# ---------------------------------------------------------------------------
# Plot 8 – sensitivity to anchoring weight (THE figure for ILO1)
# ---------------------------------------------------------------------------

def plot_sensitivity(df: pd.DataFrame, base_acc: float = 0.86, seed: int = 7):
    rng = np.random.default_rng(seed)
    weights = np.linspace(0, 1, 21)
    correct_overall, correct_when_ai_right, correct_when_ai_wrong = [], [], []

    # cache priors once per pathology so curves are smooth
    cache = {}
    for pathology in PATHOLOGIES:
        cols = pathology_columns(pathology)
        truth = df[cols["truth"]].values
        ai_p  = df[cols["ai_score"]].values
        prior = clinician_prior_prob(truth, base_acc, rng)
        ai_correct = ((ai_p >= 0.5).astype(int) == truth)
        cache[pathology] = (truth, ai_p, prior, ai_correct)

    for w_ai in weights:
        all_correct, ai_right, ai_wrong = [], [], []
        for pathology in PATHOLOGIES:
            truth, ai_p, prior, ai_correct = cache[pathology]
            posterior = update(prior, ai_p, 1 - w_ai, w_ai)
            decision = (posterior >= 0.5).astype(int)
            corr = (decision == truth).astype(int)
            all_correct.extend(corr)
            ai_right.extend(corr[ai_correct])
            ai_wrong.extend(corr[~ai_correct])
        correct_overall.append(np.mean(all_correct))
        correct_when_ai_right.append(np.mean(ai_right))
        correct_when_ai_wrong.append(np.mean(ai_wrong))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(weights, correct_overall,        "o-",
            color=PALETTE["primary"],   linewidth=2, label="Overall accuracy")
    ax.plot(weights, correct_when_ai_right,  "s-",
            color=PALETTE["accent"],    linewidth=2, label="When AI is correct")
    ax.plot(weights, correct_when_ai_wrong,  "^-",
            color=PALETTE["secondary"], linewidth=2, label="When AI is wrong")

    ax.axvline(0.45, color=PALETTE["neutral"], linestyle=":", linewidth=1)
    ax.text(0.46, 0.05, "Empirical anchoring\n(Gaube et al. 2021)",
            fontsize=8, color=PALETTE["neutral"])
    ax.set_xlabel("Anchoring weight on AI signal  ($w_{AI}$)")
    ax.set_ylabel("Clinician accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, 1.0)
    ax.set_title("Sensitivity of clinician accuracy to anchoring weight")
    ax.legend(loc="center right")
    fig.savefig(FIG_DIR / "08_anchoring_sensitivity.png")
    plt.close(fig)

    return pd.DataFrame({
        "w_ai": weights,
        "overall": correct_overall,
        "ai_right": correct_when_ai_right,
        "ai_wrong": correct_when_ai_wrong,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df, source = load()
    print(f"Loaded {len(df)} studies from {source}.\n")

    results = simulate_dataset(df)

    print("→ Plot 6: three-protocol comparison")
    agg_protocols = plot_three_protocols(results)

    print("→ Plot 7: cost of anchoring when AI is wrong")
    agg_aiwrong = plot_when_ai_wrong(results)

    print("→ Plot 8: sensitivity to anchoring weight")
    sens = plot_sensitivity(df)

    # Snapshot for the website's interactive simulator
    out = Path("website/assets/simulator_grid.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "weights":  sens["w_ai"].tolist(),
        "overall":  sens["overall"].tolist(),
        "ai_right": sens["ai_right"].tolist(),
        "ai_wrong": sens["ai_wrong"].tolist(),
        "protocols": agg_protocols.round(3).reset_index().to_dict(orient="records"),
        "ai_wrong_by_protocol": agg_aiwrong.round(3).reset_index().to_dict(orient="records"),
    }, indent=2))
    print(f"\nWrote simulator grid → {out}")

    # Headline numbers for the report
    overall = results.groupby("protocol")["correct"].mean()
    print("\nHeadline accuracies:")
    for proto, acc in overall.items():
        print(f"  {proto:18s}  {acc:.3f}")
    print(f"\nAccuracy lift from AI-first vs Independent : {overall['AI shown first'] - overall['Independent']:+.3f}")
    print(f"Accuracy lift from AI-after vs Independent : {overall['AI shown after'] - overall['Independent']:+.3f}")

    when_wrong = results[~results["ai_correct"]].groupby("protocol")["correct"].mean()
    print("\nWhen the AI is wrong, accuracy by protocol:")
    for proto, acc in when_wrong.items():
        print(f"  {proto:18s}  {acc:.3f}")


if __name__ == "__main__":
    main()
