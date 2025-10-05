"""Utility functions for data processing."""
from typing import List

import pandas as pd


def validate_columns(df: pd.DataFrame, required: List[str]):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def compute_unemployment_rate(df: pd.DataFrame, unemployed_col: str = "unemployed_count", laborforce_col: str = "labor_force_count") -> pd.DataFrame:
    """Compute unemployment_rate = unemployed_count / labor_force_count and return a new DataFrame with the column added."""
    out = df.copy()

    # If both count columns are present, compute the proportion and percent variants
    if unemployed_col in out.columns and laborforce_col in out.columns:
        try:
            out['unemployment_rate'] = out[unemployed_col].astype(float) / out[laborforce_col].astype(float)
            out['unemployed_rate'] = out['unemployment_rate'] * 100.0
            return out
        except Exception:
            # fall through and try other heuristics
            pass

    # If a precomputed 'unemployed_rate' exists (likely percent), normalize it
    if 'unemployed_rate' in out.columns:
        vals = pd.to_numeric(out['unemployed_rate'], errors='coerce')
        # Heuristic: if typical values > 1 assume percent (e.g. 3.1 -> 3.1%), else already proportion
        if vals.dropna().mean() if not vals.dropna().empty else 0 > 1:
            out['unemployment_rate'] = vals / 100.0
        else:
            out['unemployment_rate'] = vals
        out['unemployed_rate'] = out['unemployment_rate'] * 100.0
        return out

    # If a precomputed 'unemployment_rate' exists (proportion), create percent variant
    if 'unemployment_rate' in out.columns:
        vals = pd.to_numeric(out['unemployment_rate'], errors='coerce')
        out['unemployed_rate'] = vals * 100.0
        out['unemployment_rate'] = vals
        return out

    # If we get here, required inputs are missing
    raise ValueError(f"Missing required columns to compute unemployment rate. Provide either ({unemployed_col} and {laborforce_col}) or one of 'unemployed_rate'/'unemployment_rate'.")
