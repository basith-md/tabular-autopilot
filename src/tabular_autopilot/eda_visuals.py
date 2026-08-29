"""Chart builders. Each function returns a base64 PNG data URI so the HTML
report has no external image files to manage, and works identically whether
the report is written to disk or rendered inline in the Streamlit app.

Color usage follows the validated default palette from the dataviz skill
(references/palette.md): a fixed 8-hue categorical order, a single-hue blue
ramp for magnitude/sequential encodings, and a blue<->red diverging pair for
polarity (correlation). A chart with one series always uses categorical slot
1 (blue) rather than an arbitrary color, so "this is the metric being
measured" reads consistently across every section of the report.
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
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats as scipy_stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree

# --- Palette (dataviz skill, references/palette.md) -------------------------
CAT_BLUE = "#2a78d6"
CAT_ORANGE = "#eb6834"
CAT_AQUA = "#1baf7a"
CAT_YELLOW = "#eda100"
CAT_MAGENTA = "#e87ba4"
CAT_GREEN = "#008300"
CAT_VIOLET = "#4a3aa7"
CAT_RED = "#e34948"
CATEGORICAL = [CAT_BLUE, CAT_ORANGE, CAT_AQUA, CAT_YELLOW, CAT_MAGENTA, CAT_GREEN, CAT_VIOLET, CAT_RED]

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
DIVERGING_MID = "#f0efec"

DIVERGING_CMAP = LinearSegmentedColormap.from_list("diverging_blue_red", [CAT_RED, DIVERGING_MID, CAT_BLUE])
SEQUENTIAL_BLUE_CMAP = LinearSegmentedColormap.from_list("sequential_blue", ["#cde2fb", "#3987e5", "#0d366b"])

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_SECONDARY,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.7,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    }
)


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
    sns.barplot(x=series.values * 100, y=series.index, ax=ax, color=CAT_BLUE)
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
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color=CAT_BLUE)
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
    sns.heatmap(
        corr,
        annot=len(cols) <= 15,
        fmt=".2f",
        cmap=DIVERGING_CMAP,
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "correlation"},
        linewidths=0.5,
        linecolor=SURFACE,
    )
    ax.set_title("Correlation heatmap")
    return _fig_to_data_uri(fig)


def plot_target_by_category(df: pd.DataFrame, target: str, cat_col: str) -> str | None:
    if target not in df.columns or cat_col not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    order = df.groupby(cat_col)[target].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x=cat_col, y=target, order=order, ax=ax, color=CAT_BLUE)
    ax.set_title(f"{target} by {cat_col}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return _fig_to_data_uri(fig)


def plot_geo_scatter(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    target_col: str | None,
    task: str | None = None,
    cluster_col: str | None = None,
) -> str | None:
    """Two panels, both at an equal aspect ratio so the point cloud reads as
    an actual map rather than a stretched blob: one colored by the target
    (what the data looks like), one colored by the ``geo_cluster`` feature
    the pipeline engineered (what feature engineering actually did to it)."""
    if lat_col not in df.columns or lon_col not in df.columns:
        return None
    has_cluster = cluster_col is not None and cluster_col in df.columns and df[cluster_col].nunique() > 1
    n_panels = 2 if has_cluster else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, 5.6))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    if target_col and target_col in df.columns:
        if task == "classification":
            classes = sorted(df[target_col].astype(str).unique())[: len(CATEGORICAL)]
            for i, cls in enumerate(classes):
                mask = df[target_col].astype(str) == cls
                ax.scatter(
                    df.loc[mask, lon_col], df.loc[mask, lat_col], s=10, alpha=0.7,
                    color=CATEGORICAL[i % len(CATEGORICAL)], label=cls, edgecolors="none",
                )
            ax.legend(frameon=False, fontsize=8, title=target_col, loc="best")
        else:
            sc = ax.scatter(df[lon_col], df[lat_col], c=df[target_col], cmap=SEQUENTIAL_BLUE_CMAP, s=10, alpha=0.7)
            fig.colorbar(sc, ax=ax, label=target_col)
        ax.set_title(f"Colored by {target_col}")
    else:
        ax.scatter(df[lon_col], df[lat_col], s=10, alpha=0.6, color=CAT_BLUE, edgecolors="none")
        ax.set_title("Point density")
    ax.set_xlabel(lon_col)
    ax.set_ylabel(lat_col)
    ax.set_aspect("equal", adjustable="datalim")

    if has_cluster:
        ax2 = axes[1]
        clusters = sorted(c for c in df[cluster_col].unique() if c != -1)
        for i, cluster_id in enumerate(clusters):
            mask = df[cluster_col] == cluster_id
            ax2.scatter(
                df.loc[mask, lon_col], df.loc[mask, lat_col], s=10, alpha=0.7,
                color=CATEGORICAL[i % len(CATEGORICAL)], label=f"cluster {cluster_id}", edgecolors="none",
            )
        ax2.set_xlabel(lon_col)
        ax2.set_ylabel(lat_col)
        ax2.set_title(f"Colored by spatial cluster (KMeans, k={len(clusters)})")
        ax2.set_aspect("equal", adjustable="datalim")
        ax2.legend(frameon=False, fontsize=7, ncol=2, loc="best")

    fig.tight_layout()
    return _fig_to_data_uri(fig)


def plot_pca_scatter(X: pd.DataFrame, y: pd.Series | None, task: str | None) -> str | None:
    """A 2D PCA projection of the final engineered feature matrix -- a quick
    "does this data separate at all" visual, run before modeling commits to
    any one algorithm. Most useful once TF-IDF adds dozens of columns."""
    numeric_X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    numeric_X = numeric_X.loc[:, numeric_X.std(numeric_only=True) > 1e-8]
    if numeric_X.shape[1] < 2 or numeric_X.shape[0] < 3:
        return None
    scaled = StandardScaler().fit_transform(numeric_X)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    if y is not None and task == "classification":
        labels = pd.Series(y).astype(str).reset_index(drop=True)
        for i, cls in enumerate(sorted(labels.unique())[: len(CATEGORICAL)]):
            mask = (labels == cls).to_numpy()
            ax.scatter(
                coords[mask, 0], coords[mask, 1], s=12, alpha=0.7,
                color=CATEGORICAL[i % len(CATEGORICAL)], label=cls, edgecolors="none",
            )
        ax.legend(frameon=False, fontsize=8, loc="best")
    elif y is not None:
        sc = ax.scatter(
            coords[:, 0], coords[:, 1], c=pd.to_numeric(y, errors="coerce"), cmap=SEQUENTIAL_BLUE_CMAP, s=12, alpha=0.7
        )
        fig.colorbar(sc, ax=ax, label="target")
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=12, alpha=0.6, color=CAT_BLUE, edgecolors="none")

    var1, var2 = pca.explained_variance_ratio_[:2] * 100
    ax.set_xlabel(f"PC1 ({var1:.0f}% of variance)")
    ax.set_ylabel(f"PC2 ({var2:.0f}% of variance)")
    ax.set_title("PCA projection of engineered features")
    fig.tight_layout()
    return _fig_to_data_uri(fig)


def plot_cluster_scatter(df: pd.DataFrame, columns_used: list[str], labels: list[int]) -> str | None:
    """The standalone-clustering result (clustering.py) projected to 2D via
    PCA and colored by cluster id -- independent of any modeling target,
    it's the general-purpose "what natural segments exist here" view."""
    cols = [c for c in columns_used if c in df.columns]
    if len(cols) < 2:
        return None
    X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    labels_arr = np.asarray(labels)
    for i, cluster_id in enumerate(sorted(set(labels_arr.tolist()))):
        mask = labels_arr == cluster_id
        ax.scatter(
            coords[mask, 0], coords[mask, 1], s=12, alpha=0.7,
            color=CATEGORICAL[i % len(CATEGORICAL)], label=f"segment {cluster_id}", edgecolors="none",
        )
    ax.legend(frameon=False, fontsize=8, loc="best")
    var1, var2 = pca.explained_variance_ratio_[:2] * 100
    ax.set_xlabel(f"PC1 ({var1:.0f}% of variance)")
    ax.set_ylabel(f"PC2 ({var2:.0f}% of variance)")
    ax.set_title("Segments found in the data (KMeans, PCA-projected)")
    fig.tight_layout()
    return _fig_to_data_uri(fig)


def plot_residuals_vs_fitted(fitted: list[float], residuals: list[float]) -> str | None:
    if not fitted or not residuals:
        return None
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(fitted, residuals, alpha=0.45, s=14, color=CAT_BLUE, edgecolors="none")
    ax.axhline(0, color=CAT_RED, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs fitted")
    return _fig_to_data_uri(fig)


def plot_qq(residuals: list[float]) -> str | None:
    if not residuals:
        return None
    fig, ax = plt.subplots(figsize=(5, 5))
    scipy_stats.probplot(np.array(residuals), dist="norm", plot=ax)
    lines = ax.get_lines()
    if len(lines) >= 1:
        lines[0].set_markerfacecolor(CAT_BLUE)
        lines[0].set_markeredgecolor(CAT_BLUE)
        lines[0].set_alpha(0.6)
    if len(lines) >= 2:
        lines[1].set_color(CAT_RED)
        lines[1].set_linewidth(1.5)
    ax.set_title("Q-Q plot of residuals")
    return _fig_to_data_uri(fig)


def plot_feature_importance(importances: dict[str, float]) -> str | None:
    if not importances:
        return None
    series = pd.Series(importances).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(series))))
    sns.barplot(x=series.values, y=series.index, ax=ax, color=CAT_BLUE)
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
    ax.plot(hist_x, history_values, color=CAT_BLUE, linewidth=1.6, label="observed")
    if forecast:
        fc_x = pd.to_datetime(forecast_index)
        ax.plot(fc_x, forecast, color=CAT_ORANGE, linestyle="--", marker="o", markersize=3, label="forecast")
    ax.set_title("Time series: history and forecast")
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    return _fig_to_data_uri(fig)


def plot_intervention(
    history_dates: list[str], history_values: list[float], fitted: list[float], intervention_date: str
) -> str | None:
    if not history_dates:
        return None
    fig, ax = plt.subplots(figsize=(9, 4))
    x = pd.to_datetime(history_dates)
    ax.plot(x, history_values, color=CAT_BLUE, linewidth=1.2, alpha=0.55, label="observed")
    ax.plot(x, fitted, color=CAT_ORANGE, linewidth=2, label="segmented fit (pre/post)")
    ax.axvline(pd.to_datetime(intervention_date), color=CAT_RED, linestyle="--", linewidth=1.5, label="intervention")
    ax.set_title("Interrupted time series: level & slope around the intervention")
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    return _fig_to_data_uri(fig)


def plot_acf_pacf(acf_values: list[float], pacf_values: list[float]) -> str | None:
    if not acf_values:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, values, title in zip(axes, [acf_values, pacf_values], ["ACF", "PACF"]):
        lags = np.arange(len(values))
        ax.vlines(lags, 0, values, color=CAT_BLUE, linewidth=1.5)
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Lag")
    fig.tight_layout()
    return _fig_to_data_uri(fig)


def plot_confusion_matrix(matrix: list[list[int]], labels: list[str]) -> str | None:
    if not matrix:
        return None
    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 2, 1.2 * len(labels) + 2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap=SEQUENTIAL_BLUE_CMAP,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.5,
        linecolor=SURFACE,
        cbar=False,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    return _fig_to_data_uri(fig)


def plot_decision_tree(estimator, feature_names: list[str], class_names: list[str] | None = None) -> str | None:
    """Renders one shallow, depth-limited decision tree so a tree-ensemble
    winner (Random Forest / Gradient Boosting) is explainable at a glance --
    not the actual production model (which is hundreds of deeper trees
    averaged together), but a faithful illustration of the kind of split
    logic those ensembles are built from."""
    if estimator is None:
        return None
    fig, ax = plt.subplots(figsize=(15, 7))
    plot_tree(
        estimator,
        feature_names=feature_names,
        class_names=class_names,
        filled=False,
        rounded=False,
        fontsize=8,
        ax=ax,
        proportion=True,
    )
    for text in ax.texts:
        text.set_color(INK_PRIMARY)
    ax.set_title("How one representative tree splits the data (max depth 3, for illustration)")
    return _fig_to_data_uri(fig)
