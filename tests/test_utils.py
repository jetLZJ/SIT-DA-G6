import pandas as pd
from app.utils import compute_unemployment_rate


def test_compute_unemployment_rate():
    df = pd.DataFrame({
        'unemployed_count': [5, 10],
        'labor_force_count': [100, 200]
    })
    out = compute_unemployment_rate(df)
    assert 'unemployment_rate' in out.columns
    assert out['unemployment_rate'].iloc[0] == 0.05
    assert out['unemployment_rate'].iloc[1] == 0.05
