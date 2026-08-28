"""Descriptive profiling: missingness, distribution shape, outliers.

Runs after schema inference and before cleaning, so that cleaning decisions
(which columns to impute, which to log-transform) are driven by what this
module observes rather than hardcoded per-dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tabular_autopilot.schema import SchemaResult

SKEW_THRESHOLD = 1.0
OUTLIER_IQR_MULT = 1.5


@dataclass
class NumericProfile:
    name: str
    mean: float
    std: float
    min: float
    max: float
    median: float
    skew: float
    is_skewed: bool
    n_outliers: int
    pct_outliers: float


@dataclass
class ProfileReport:
    n_rows: int
    n_cols: int
    missing_by_col: dict[str, float]
    numeric_profiles: dict[str, NumericProfile]
    categorical_top_values: dict[str, dict[str, int]]
    duplicate_rows: int


def _iqr_outlier_count(series: pd.Series) -> int:
    clean = series.dropna()
    if clean.empty:
        return 0
    q1, q3 = clean.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return 0
    lower, upper = q1 - OUTLIER_IQR_MULT * iqr, q3 + OUTLIER_IQR_MULT * iqr
    return int(((clean < lower) | (clean > upper)).sum())


def profile_dataframe(df: pd.DataFrame, schema: SchemaResult) -> ProfileReport:
    n_rows, n_cols = df.shape
    missing_by_col = {
        col: prof.pct_missing for col, prof in schema.columns.items() if prof.pct_missing > 0
    }

    numeric_profiles: dict[str, NumericProfile] = {}
    for col in schema.numeric_cols + [schema.target] if schema.target else schema.numeric_cols:
        if col is None or col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        skew = float(series.skew())
        n_out = _iqr_outlier_count(series)
        numeric_profiles[col] = NumericProfile(
            name=col,
            mean=float(series.mean()),
            std=float(series.std()),
            min=float(series.min()),
            max=float(series.max()),
            median=float(series.median()),
            skew=skew,
            is_skewed=abs(skew) >= SKEW_THRESHOLD,
            n_outliers=n_out,
            pct_outliers=n_out / len(series) if len(series) else 0.0,
        )

    categorical_top_values: dict[str, dict[str, int]] = {}
    for col in schema.categorical_low_cols + schema.categorical_high_cols:
        counts = df[col].value_counts(dropna=True).head(10)
        categorical_top_values[col] = {str(k): int(v) for k, v in counts.items()}

    return ProfileReport(
        n_rows=n_rows,
        n_cols=n_cols,
        missing_by_col=missing_by_col,
        numeric_profiles=numeric_profiles,
        categorical_top_values=categorical_top_values,
        duplicate_rows=int(df.duplicated().sum()),
    )
