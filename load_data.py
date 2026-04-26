"""
load_data.py
------------
Single entry point for analysis scripts. Prefers real CheXpert validation data
if present; falls back to the synthesised replica.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

REAL_VALID_PATH = Path("data/raw/CheXpert-v1.0-small/valid.csv")
SYNTH_PATH      = Path("data/chexpert_sample.csv")

PATHOLOGIES = ["Atelectasis", "Cardiomegaly", "Consolidation",
               "Edema", "Pleural Effusion"]


def load(prefer_real: bool = True) -> tuple[pd.DataFrame, str]:
    """
    Returns
    -------
    df : pandas.DataFrame
    source : "real" | "synthesised"
    """
    if prefer_real and REAL_VALID_PATH.exists():
        df = pd.read_csv(REAL_VALID_PATH)
        return df, "real"
    if not SYNTH_PATH.exists():
        raise FileNotFoundError(
            f"Neither {REAL_VALID_PATH} nor {SYNTH_PATH} exist. "
            "Run `python data/synthesize_chexpert_like.py` first."
        )
    return pd.read_csv(SYNTH_PATH), "synthesised"


def pathology_columns(pathology: str) -> dict[str, str]:
    """Helper: return the column names for a given pathology in the synth CSV."""
    key = pathology.lower().replace(" ", "_")
    return {
        "truth":      f"truth_{key}",
        "ai_score":   f"ai_score_{key}",
        "rad_indep":  f"radiologist_indep_{key}",
        "rad_anchored": f"radiologist_anchored_{key}",
    }


if __name__ == "__main__":
    df, src = load()
    print(f"Loaded {len(df)} rows from {src} source.")
    print(df.head().to_string())
