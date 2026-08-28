"""Chart builders. Each function returns a base64 PNG data URI so the HTML
report has no external image files to manage, and works identically whether
the report is written to disk or rendered inline in the Streamlit app.
"""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats

sns.set_theme(style="whitegrid")


def _fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def plot_missingness(df: pd.DataFrame, missing_by_col: dict[str, float]) -> str | None:
    if not missing_by_col:
        return None
    series = pd.Series(missing_by_col).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, max(2, 0.35 * len(series))))
    sns.barplot(x=series.values * 100, y=series.index, ax=ax, color="#4C72B0")
    ax.set_xlabel("% missing")
    ax.set_title("Missing values by column")
    return _fig_to_data_uri(fig)


def plot_numeric_distributions(df: pd.DataFrame, numeric_cols: list[str]) -> str | None:
    cols = [c for c in numeric_cols if c in df.columns][:12]
    if not cols:
        return None
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, col in zip(axes, cols):
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#4C72B0")
        ax.set_title(col, fontsize=10)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    return _fig_to_data_uri(fig)


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list[str]) -> str | None:
    cols = [c for c in numeric_cols if c in df.columns]
    if len(cols) < 2:
        return None
    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(cols)), max(5, 0.5 * len(cols))))
    sns.heatmap(corr, annot=len(cols) <= 15, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap")
    return _fig_to_data_uri(fig)


def plot_target_by_category(df: pd.DataFrame, target: str, cat_col: str) -> str | None:
    if target not in df.columns or cat_col not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    order = df.groupby(cat_col)[target].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x=cat_col, y=target, order=order, ax=ax, color="#4C72B0")
    ax.set_title(f"{target} by {cat_col}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return _fig_to_data_uri(fig)


def plot_geo_scatter(df: pd.DataFrame, lat_col: str, lon_col: str, color_col: str | None) -> str | None:
    if lat_col not in df.columns or lon_col not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(7, 6))
    if color_col and color_col in df.columns:
        sc = ax.scatter(df[lon_col], df[lat_col], c=df[color_col], cmap="viridis", s=10, alpha=0.6)
        fig.colorbar(sc, ax=ax, label=color_col)
    else:
        ax.scatter(df[lon_col], df[lat_col], s=10, alpha=0.6, color="#4C72B0")
    ax.set_xlabel(lon_col)
    ax.set_ylabel(lat_col)
    ax.set_title("Geospatial distribution")
    return _fig_to_data_uri(fig)


def plot_residuals_vs_fitted(fitted: list[float], residuals: list[float]) -> str | None:
    if not fitted or not residuals:
        return None
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(fitted, residuals, alpha=0.4, s=12, color="#4C72B0")
    ax.axhline(0, color="crimson", linestyle="--")
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs fitted")
    return _fig_to_data_uri(fig)


def plot_qq(residuals: list[float]) -> str | None:
    if not residuals:
        return None
    fig, ax = plt.subplots(figsize=(5, 5))
    scipy_stats.probplot(np.array(residuals), dist="norm", plot=ax)
    ax.set_title("Q-Q plot of residuals")
    return _fig_to_data_uri(fig)


def plot_feature_importance(importances: dict[str, float]) -> str | None:
    if not importances:
        return None
    series = pd.Series(importances).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(series))))
    sns.barplot(x=series.values, y=series.index, ax=ax, color="#55A868")
    ax.set_xlabel("Permutation importance")
    ax.set_title("Feature importance")
    fig.tight_layout()
    return _fig_to_data_uri(fig)


def plot_time_series_trend(
    history_dates: list[str],
    history_values: list[float],
    forecast_index: list[str],
    forecast: list[float],
) -> str | None:
    if not history_dates:
        return None
    fig, ax = plt.subplots(figsize=(9, 4))
    hist_x = pd.to_datetime(history_dates)
    ax.plot(hist_x, history_values, color="#4C72B0", label="observed")
    if forecast:
        fc_x = pd.to_datetime(forecast_index)
        ax.plot(fc_x, forecast, color="#DD8452", linestyle="--", marker="o", markersize=3, label="forecast")
    ax.set_title("Time series: history and forecast")
    ax.legend()
    fig.autofmt_xdate()
    return _fig_to_data_uri(fig)


def plot_acf_pacf(acf_values: list[float], pacf_values: list[float]) -> str | None:
    if not acf_values:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, values, title in zip(axes, [acf_values, pacf_values], ["ACF", "PACF"]):
        lags = np.arange(len(values))
        ax.vlines(lags, 0, values, color="#4C72B0")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Lag")
    fig.tight_layout()
    return _fig_to_data_uri(fig)


def plot_confusion_matrix(matrix: list[list[int]], labels: list[str]) -> str | None:
    if not matrix:
        return None
    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 2, 1.2 * len(labels) + 2))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    return _fig_to_data_uri(fig)
