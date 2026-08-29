"""Time-series diagnostics: trend strength, stationarity, ACF/PACF, a
formally order-selected ARIMA/SARIMA forecast, and an optional interrupted-
time-series (intervention) analysis around a known event date.

This runs alongside (not instead of) the main tabular regression/
classification path whenever the dataset shape implies it -- at least one
datetime column and a numeric target -- independent of whether the user
picked that datetime column as a model feature. It generalizes the trend
regression -> stationarity check -> ACF/PACF order identification ->
ARIMA/intervention-regression pipeline from classic time-series coursework
into an automatic, data-shape-triggered step.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, adfuller, pacf

FORECAST_HORIZON = 12
MIN_POINTS = 20
ARIMA_ORDER_RANGE = range(0, 3)  # p, q searched over {0, 1, 2}
SEASONAL_PERIOD_DAILY = 7
MIN_POINTS_FOR_SEASONAL = 2 * SEASONAL_PERIOD_DAILY
MIN_SEGMENT_SIZE = 5  # minimum pre/post observations for an intervention fit to mean anything


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
    forecast_method: str = ""
    arima_order: tuple[int, int, int] | None = None
    seasonal_order: tuple[int, int, int, int] | None = None
    aic: float | None = None
    note: str = ""


@dataclass
class InterventionResult:
    intervention_date: str
    n_pre: int
    n_post: int
    level_shift: float
    level_shift_pvalue: float
    slope_change: float
    slope_change_pvalue: float
    significant_level_shift: bool
    significant_slope_change: bool
    fitted: list[float] = field(default_factory=list)
    history_dates: list[str] = field(default_factory=list)
    history_values: list[float] = field(default_factory=list)
    note: str = ""


def _aggregate_series(df: pd.DataFrame, date_col: str, value_col: str) -> pd.Series:
    subset = df[[date_col, value_col]].dropna().copy()
    subset[date_col] = pd.to_datetime(subset[date_col], errors="coerce", format="mixed")
    subset = subset.dropna(subset=[date_col]).sort_values(date_col)
    return subset.groupby(date_col)[value_col].mean()


def _select_arima_order(y: np.ndarray, d: int) -> tuple[tuple[int, int, int], object] | None:
    """Grid search over (p, d, q) with d fixed by the ADF stationarity
    result, keeping the fit with the lowest AIC -- a bounded, automatic
    stand-in for manual ACF/PACF-driven order selection."""
    best: tuple[tuple[int, int, int], object] | None = None
    for p in ARIMA_ORDER_RANGE:
        for q in ARIMA_ORDER_RANGE:
            if p == 0 and q == 0 and d == 0:
                continue  # degenerate "no dynamics at all" case
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted = ARIMA(y, order=(p, d, q)).fit()
            except Exception:
                continue
            if best is None or fitted.aic < best[1].aic:
                best = ((p, d, q), fitted)
    return best


def _select_seasonal_model(y: np.ndarray, order: tuple[int, int, int], freq: str, n: int) -> object | None:
    """One seasonal candidate (weekly, for daily data) at the same (p, d, q)
    as the best non-seasonal fit -- compared by AIC, not searched further,
    to keep the total fit count bounded."""
    if freq != "D" or n < MIN_POINTS_FOR_SEASONAL:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SARIMAX(
                y, order=order, seasonal_order=(1, 1, 1, SEASONAL_PERIOD_DAILY),
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)
    except Exception:
        return None


def _forecast_index(series: pd.Series, freq: str) -> list[str]:
    future = pd.date_range(series.index[-1], periods=FORECAST_HORIZON + 1, freq=freq)[1:]
    return [str(d.date()) for d in future]


def analyze_time_series(df: pd.DataFrame, date_col: str, value_col: str) -> TimeSeriesResult | None:
    series = _aggregate_series(df, date_col, value_col)
    if len(series) < MIN_POINTS:
        return None

    t = np.arange(len(series))
    y = series.to_numpy(dtype=float)

    slope, intercept = np.polyfit(t, y, 1)
    fitted_trend = slope * t + intercept
    ss_res = float(np.sum((y - fitted_trend) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    trend_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    adf_stat, adf_pvalue = np.nan, np.nan
    try:
        adf_stat, adf_pvalue = adfuller(y, autolag="AIC", result_object=False)[:2]
    except Exception:
        pass
    is_stationary = bool(adf_pvalue < 0.05) if adf_pvalue == adf_pvalue else False

    nlags = max(1, min(24, len(series) // 2 - 1))
    acf_values = acf(y, nlags=nlags, fft=True).tolist()
    pacf_values = pacf(y, nlags=nlags).tolist()

    freq = pd.infer_freq(series.index) or "D"
    forecast_values: list[float] = []
    forecast_idx: list[str] = []
    forecast_method = ""
    arima_order: tuple[int, int, int] | None = None
    seasonal_order: tuple[int, int, int, int] | None = None
    aic: float | None = None
    note = ""

    d = 0 if is_stationary else 1
    best = _select_arima_order(y, d)
    final_model = None
    if best is not None:
        arima_order, final_model = best
        forecast_method = "arima"
        seasonal_model = _select_seasonal_model(y, arima_order, freq, len(series))
        if seasonal_model is not None and seasonal_model.aic < final_model.aic:
            final_model = seasonal_model
            forecast_method = "sarima"
            seasonal_order = (1, 1, 1, SEASONAL_PERIOD_DAILY)
        aic = float(final_model.aic)

    if final_model is not None:
        try:
            forecast_values = np.asarray(final_model.forecast(FORECAST_HORIZON)).tolist()
            forecast_idx = _forecast_index(series, freq)
        except Exception as exc:
            note = f"{forecast_method.upper()} fit succeeded but forecasting failed ({exc}); falling back."
            final_model = None
            forecast_method = ""

    if final_model is None:
        try:
            es_model = ExponentialSmoothing(y, trend="add", seasonal=None).fit()
            forecast_values = es_model.forecast(FORECAST_HORIZON).tolist()
            forecast_idx = _forecast_index(series, freq)
            forecast_method = "exponential_smoothing"
            arima_order, seasonal_order, aic = None, None, None
            if not note:
                note = "ARIMA/SARIMA order search did not converge for this series; used Exponential Smoothing instead."
        except Exception as exc:
            note = f"Forecast unavailable for this series: {exc}"

    return TimeSeriesResult(
        date_col=date_col,
        value_col=value_col,
        n_obs=len(series),
        trend_r_squared=float(trend_r2),
        adf_statistic=float(adf_stat) if adf_stat == adf_stat else 0.0,
        adf_pvalue=float(adf_pvalue) if adf_pvalue == adf_pvalue else 1.0,
        is_stationary=is_stationary,
        acf_values=acf_values,
        pacf_values=pacf_values,
        forecast=forecast_values,
        forecast_index=forecast_idx,
        history_dates=[str(d_.date()) for d_ in series.index],
        history_values=y.tolist(),
        forecast_method=forecast_method,
        arima_order=arima_order,
        seasonal_order=seasonal_order,
        aic=aic,
        note=note,
    )


def analyze_intervention(
    df: pd.DataFrame, date_col: str, value_col: str, intervention_date: str
) -> InterventionResult | None:
    """Segmented (interrupted-time-series) regression: y ~ t + post +
    (t - t0)*post, where ``post`` flags observations on/after the given
    date. ``post``'s coefficient is the level shift right at the event;
    the interaction term is the slope change afterward. Requires a real
    date supplied by the caller -- there's no way to auto-detect "the"
    event in an arbitrary dataset, so this only runs when asked."""
    series = _aggregate_series(df, date_col, value_col)
    if len(series) < MIN_POINTS:
        return None
    try:
        cutoff = pd.to_datetime(intervention_date)
    except (ValueError, TypeError):
        return None
    if cutoff <= series.index.min() or cutoff >= series.index.max():
        return None

    n = len(series)
    t = np.arange(n, dtype=float)
    post = (series.index >= cutoff).astype(float)
    n_pre, n_post = int((post == 0).sum()), int((post == 1).sum())
    if n_pre < MIN_SEGMENT_SIZE or n_post < MIN_SEGMENT_SIZE:
        return None

    t0 = float(np.argmax(post == 1))
    t_since = np.where(post == 1, t - t0, 0.0)
    X = sm.add_constant(np.column_stack([t, post, t_since]))
    y = series.to_numpy(dtype=float)
    model = sm.OLS(y, X).fit()
    _const, _slope, level_shift, slope_change = model.params
    _p_const, _p_slope, p_level, p_slope_change = model.pvalues

    return InterventionResult(
        intervention_date=str(cutoff.date()),
        n_pre=n_pre,
        n_post=n_post,
        level_shift=float(level_shift),
        level_shift_pvalue=float(p_level),
        slope_change=float(slope_change),
        slope_change_pvalue=float(p_slope_change),
        significant_level_shift=bool(p_level < 0.05),
        significant_slope_change=bool(p_slope_change < 0.05),
        fitted=model.fittedvalues.tolist(),
        history_dates=[str(d_.date()) for d_ in series.index],
        history_values=y.tolist(),
    )
