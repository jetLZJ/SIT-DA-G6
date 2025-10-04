"""Utility functions for data processing."""
from typing import List

import pandas as pd


def validate_columns(df: pd.DataFrame, required: List[str]):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def compute_unemployment_rate(df: pd.DataFrame, unemployed_col: str = "unemployed_count", laborforce_col: str = "labor_force_count") -> pd.DataFrame:
    """Compute unemployment_rate = unemployed_count / labor_force_count and return a new DataFrame with the column added."""
    validate_columns(df, [unemployed_col, laborforce_col])
    out = df.copy()
    out['unemployment_rate'] = out[unemployed_col].astype(float) / out[laborforce_col].astype(float)
    return out
