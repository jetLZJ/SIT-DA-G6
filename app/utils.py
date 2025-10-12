"""Utility functions for data processing and Streamlit helpers."""
from functools import lru_cache
import inspect
from typing import Any, List, Optional

import pandas as pd
import streamlit as st


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


def render_plotly_chart(
    figure: Any,
    *,
    key: Optional[str] = None,
    config: Optional[dict[str, Any]] = None,
    fill: str = 'stretch',
):
    """Render a Plotly figure with responsive defaults across Streamlit versions.

    Streamlit < 1.51 only supports ``use_container_width`` while newer releases
    accept the ``width`` keyword. This helper inspects the runtime signature and
    dispatches to the appropriate API so callers don't need to worry about
    deprecation warnings.

    Args:
        figure: Plotly figure to render.
        key: Optional Streamlit element key.
        config: Optional Plotly config dict (defaults to responsive layout).
        fill: Either ``'stretch'`` or ``'content'`` analogous to the new width API.
    """

    config = {'responsive': True, **(config or {})}

    if _plotly_supports_width_keyword():
        width_value = fill if fill in {'stretch', 'content'} else 'stretch'
        return st.plotly_chart(figure, width=width_value, config=config, key=key)

    kwargs: dict[str, Any] = {}
    if fill == 'content':
        kwargs['use_container_width'] = False

    return st.plotly_chart(
        figure,
        config=config,
        key=key,
        **kwargs,
    )


@lru_cache(maxsize=1)
def _plotly_supports_width_keyword() -> bool:
    try:
        signature = inspect.signature(st.plotly_chart)
    except (TypeError, ValueError):  # pragma: no cover - extremely unlikely
        return False
    return 'width' in signature.parameters
