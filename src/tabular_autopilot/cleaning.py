"""Per-role cleaning: imputation and skew correction.

Decisions here are driven by the ``ProfileReport`` produced upstream, not
hardcoded to any one dataset:

- Numeric columns: median imputation for missing values; right-skewed
  columns (|skew| >= threshold, all non-negative) get a ``log1p`` transform,
  recorded so the report can show before/after distributions.
- Categorical columns: mode imputation, missing values falling back to an
  explicit "Unknown" category when there is no mode (all-missing column).
- Datetime columns: left untouched here; ``feature_engineering`` expands
  them into numeric parts, filling gaps at that stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tabular_autopilot.profiling import ProfileReport
from tabular_autopilot.schema import SchemaResult


@dataclass
class CleaningReport:
    imputed_numeric: dict[str, float] = field(default_factory=dict)
    imputed_categorical: dict[str, str] = field(default_factory=dict)
    log_transformed: list[str] = field(default_factory=list)


NUMERIC_IMPUTE_STRATEGIES = ("median", "mean")


def clean_dataframe(
    df: pd.DataFrame,
    schema: SchemaResult,
    profile: ProfileReport,
    numeric_impute_strategy: str = "median",
) -> tuple[pd.DataFrame, CleaningReport]:
    if numeric_impute_strategy not in NUMERIC_IMPUTE_STRATEGIES:
        raise ValueError(f"numeric_impute_strategy must be one of {NUMERIC_IMPUTE_STRATEGIES}")

    out = df.copy()
    report = CleaningReport()

    for col in schema.numeric_cols:
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if out[col].isna().any():
            if out[col].notna().any():
                fill_value = float(out[col].mean()) if numeric_impute_strategy == "mean" else float(out[col].median())
            else:
                fill_value = 0.0
            out[col] = out[col].fillna(fill_value)
            report.imputed_numeric[col] = fill_value

        num_profile = profile.numeric_profiles.get(col)
        if num_profile is not None and num_profile.is_skewed and (out[col] >= 0).all():
            out[f"{col}_log"] = np.log1p(out[col])
            report.log_transformed.append(col)

    for col in schema.categorical_low_cols + schema.categorical_high_cols:
        if col not in out.columns:
            continue
        out[col] = out[col].astype("object")
        if out[col].isna().any():
            mode = out[col].mode(dropna=True)
            fill_value = str(mode.iloc[0]) if not mode.empty else "Unknown"
            out[col] = out[col].fillna(fill_value)
            report.imputed_categorical[col] = fill_value

    if schema.target and schema.target in out.columns:
        out = out[out[schema.target].notna()].reset_index(drop=True)

    return out, report
