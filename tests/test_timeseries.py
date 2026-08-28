import numpy as np
import pandas as pd

from tabular_autopilot.timeseries import analyze_time_series


def test_time_series_detects_trend_and_forecasts():
    n = 120
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    rng = np.random.default_rng(3)
    values = 100 + 0.5 * np.arange(n) + rng.normal(scale=2.0, size=n)
    df = pd.DataFrame({"date": dates, "sales": values})

    result = analyze_time_series(df, "date", "sales")

    assert result is not None
    assert result.n_obs == n
    assert result.trend_r_squared > 0.9
    assert len(result.acf_values) > 0
    assert len(result.pacf_values) > 0
    assert len(result.forecast) == 12
    assert len(result.forecast_index) == 12


def test_time_series_returns_none_for_too_few_points():
    df = pd.DataFrame({"date": pd.date_range("2021-01-01", periods=5), "sales": [1, 2, 3, 4, 5]})
    assert analyze_time_series(df, "date", "sales") is None
