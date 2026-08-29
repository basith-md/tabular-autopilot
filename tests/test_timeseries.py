import numpy as np
import pandas as pd

from tabular_autopilot.timeseries import analyze_intervention, analyze_time_series


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
    assert result.forecast_method in ("arima", "sarima", "exponential_smoothing")


def test_time_series_returns_none_for_too_few_points():
    df = pd.DataFrame({"date": pd.date_range("2021-01-01", periods=5), "sales": [1, 2, 3, 4, 5]})
    assert analyze_time_series(df, "date", "sales") is None


def test_time_series_aggregates_intraday_timestamps_to_calendar_days():
    """Discovered on a real NYC-taxi-style dataset: a datetime column with a
    time-of-day component (pickup timestamps, order times, ...) must be
    aggregated to one observation per calendar day, not grouped by the exact
    instant -- otherwise "N observations" silently becomes "N rows" and the
    series fed to ADF/ACF/ARIMA is raw per-event noise, not a real series."""
    rng = np.random.default_rng(8)
    n_days = 40
    rides_per_day = 50
    base_dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    rows = []
    for day_offset, day in enumerate(base_dates):
        times = day + pd.to_timedelta(rng.integers(0, 24 * 3600, size=rides_per_day), unit="s")
        fares = 10 + 0.2 * day_offset + rng.normal(scale=1.0, size=rides_per_day)
        rows.append(pd.DataFrame({"pickup": times, "fare": fares}))
    df = pd.concat(rows, ignore_index=True)
    assert df["pickup"].nunique() > n_days * (rides_per_day - 5)  # near-unique timestamps, like real ride data

    result = analyze_time_series(df, "pickup", "fare")

    assert result is not None
    assert result.n_obs == n_days  # aggregated to one point per day, not one per ride
    assert result.trend_r_squared > 0.5  # the daily upward trend should survive aggregation


def test_time_series_selects_an_arima_order_for_trending_data():
    n = 120
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    rng = np.random.default_rng(4)
    values = 100 + 0.5 * np.arange(n) + rng.normal(scale=2.0, size=n)
    df = pd.DataFrame({"date": dates, "sales": values})

    result = analyze_time_series(df, "date", "sales")

    assert result.forecast_method in ("arima", "sarima")
    assert result.arima_order is not None
    assert result.aic is not None


def test_time_series_picks_sarima_for_clear_weekly_seasonality():
    n = 90
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    rng = np.random.default_rng(5)
    weekday_effect = (dates.dayofweek < 5).astype(float) * 20.0  # weekday vs weekend split
    values = 50 + weekday_effect + rng.normal(scale=1.0, size=n)
    df = pd.DataFrame({"date": dates, "sales": values})

    result = analyze_time_series(df, "date", "sales")

    assert result.forecast_method == "sarima"
    assert result.seasonal_order == (1, 1, 1, 7)


def test_intervention_detects_a_real_level_shift():
    n = 100
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    rng = np.random.default_rng(6)
    cutoff_idx = 50
    values = np.concatenate(
        [rng.normal(loc=50, scale=1.0, size=cutoff_idx), rng.normal(loc=80, scale=1.0, size=n - cutoff_idx)]
    )
    df = pd.DataFrame({"date": dates, "value": values})

    result = analyze_intervention(df, "date", "value", str(dates[cutoff_idx].date()))

    assert result is not None
    assert result.significant_level_shift
    assert result.level_shift > 10  # a real ~30-unit jump, not noise
    assert result.n_pre == cutoff_idx
    assert result.n_post == n - cutoff_idx


def test_intervention_returns_none_outside_the_series_range():
    n = 60
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    df = pd.DataFrame({"date": dates, "value": np.arange(n, dtype=float)})

    assert analyze_intervention(df, "date", "value", "2019-01-01") is None
    assert analyze_intervention(df, "date", "value", "2030-01-01") is None


def test_intervention_returns_none_when_a_segment_is_too_small():
    n = 60
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    df = pd.DataFrame({"date": dates, "value": np.arange(n, dtype=float)})

    # Only 2 days after the cutoff -- below MIN_SEGMENT_SIZE.
    assert analyze_intervention(df, "date", "value", str(dates[-2].date())) is None


def test_intervention_returns_none_for_too_few_total_points():
    df = pd.DataFrame({"date": pd.date_range("2021-01-01", periods=5), "value": [1, 2, 3, 4, 5]})
    assert analyze_intervention(df, "date", "value", "2021-01-03") is None


def test_intervention_returns_none_for_an_unparseable_date():
    n = 60
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    df = pd.DataFrame({"date": dates, "value": np.arange(n, dtype=float)})

    assert analyze_intervention(df, "date", "value", "not-a-date") is None
