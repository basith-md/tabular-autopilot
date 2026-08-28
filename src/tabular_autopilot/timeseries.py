"""Time-series diagnostics: trend strength, stationarity, ACF/PACF, and a
short-horizon forecast.

This runs alongside (not instead of) the main tabular regression/
classification path whenever the dataset shape implies it — at least one
datetime column and a numeric target — independent of whether the user
picked that datetime column as a model feature. It generalizes the trend
regression -> stationarity check -> ACF/PACF order identification ->
AR/ETS forecast pipeline from classic time-series coursework into an
automatic, data-shape-triggered step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import acf, adfuller, pacf

FORECAST_HORIZON = 12
MIN_POINTS = 20


@dataclass
class TimeSeriesResult:
    date_col: str
    value_col: str
    n_obs: int
    trend_r_squared: float
    adf_statistic: float
    adf_pvalue: float
    is_stationary: bool
    acf_values: list[float] = field(default_factory=list)
    pacf_values: list[float] = field(default_factory=list)
    forecast: list[float] = field(default_factory=list)
    forecast_index: list[str] = field(default_factory=list)
    history_dates: list[str] = field(default_factory=list)
    history_values: list[float] = field(default_factory=list)
    note: str = ""


def _aggregate_series(df: pd.DataFrame, date_col: str, value_col: str) -> pd.Series:
    subset = df[[date_col, value_col]].dropna().copy()
    subset[date_col] = pd.to_datetime(subset[date_col], errors="coerce", format="mixed")
    subset = subset.dropna(subset=[date_col]).sort_values(date_col)
    return subset.groupby(date_col)[value_col].mean()


def analyze_time_series(df: pd.DataFrame, date_col: str, value_col: str) -> TimeSeriesResult | None:
    series = _aggregate_series(df, date_col, value_col)
    if len(series) < MIN_POINTS:
        return None

    t = np.arange(len(series))
    y = series.to_numpy(dtype=float)

    slope, intercept = np.polyfit(t, y, 1)
    fitted = slope * t + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    trend_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    adf_stat, adf_pvalue = np.nan, np.nan
    try:
        adf_stat, adf_pvalue = adfuller(y, autolag="AIC", result_object=False)[:2]
    except Exception:
        pass

    nlags = max(1, min(24, len(series) // 2 - 1))
    acf_values = acf(y, nlags=nlags, fft=True).tolist()
    pacf_values = pacf(y, nlags=nlags).tolist()

    forecast_values: list[float] = []
    forecast_index: list[str] = []
    note = ""
    try:
        model = ExponentialSmoothing(y, trend="add", seasonal=None).fit()
        fc = model.forecast(FORECAST_HORIZON)
        forecast_values = fc.tolist()
        freq = pd.infer_freq(series.index) or "D"
        future = pd.date_range(series.index[-1], periods=FORECAST_HORIZON + 1, freq=freq)[1:]
        forecast_index = [str(d.date()) for d in future]
    except Exception as exc:
        note = f"Forecast unavailable for this series: {exc}"

    return TimeSeriesResult(
        date_col=date_col,
        value_col=value_col,
        n_obs=len(series),
        trend_r_squared=float(trend_r2),
        adf_statistic=float(adf_stat) if adf_stat == adf_stat else 0.0,
        adf_pvalue=float(adf_pvalue) if adf_pvalue == adf_pvalue else 1.0,
        is_stationary=bool(adf_pvalue < 0.05) if adf_pvalue == adf_pvalue else False,
        acf_values=acf_values,
        pacf_values=pacf_values,
        forecast=forecast_values,
        forecast_index=forecast_index,
        history_dates=[str(d.date()) for d in series.index],
        history_values=y.tolist(),
        note=note,
    )
