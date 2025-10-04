"""Plotly visualization helpers."""
from typing import Optional

import pandas as pd
import plotly.express as px


def small_multiples_time_series(df: pd.DataFrame, group_col: str, x_col: str, y_col: str, max_panels: int = 12):
    """Return a Plotly figure with small multiples time-series (facet_col wrap)."""
    groups = list(df[group_col].unique())[:max_panels]
    df_f = df[df[group_col].isin(groups)].copy()
    fig = px.line(df_f, x=x_col, y=y_col, color=group_col, facet_col=group_col, facet_col_wrap=4, height=600)
    fig.update_layout(showlegend=False)
    return fig
