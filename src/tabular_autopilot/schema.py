"""Column role inference for arbitrary tabular datasets.

Every other pipeline stage (cleaning, feature engineering, modeling,
visualization) branches on the roles computed here, so this is the one
place that encodes "how do we tell what kind of column this is."
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

_LAT_PATTERN = re.compile(r"^(lat|latitude)$", re.IGNORECASE)
_LON_PATTERN = re.compile(r"^(lon|lng|long|longitude)$", re.IGNORECASE)
_ID_PATTERN = re.compile(r"(^id$|_id$|^id_|uuid|guid)", re.IGNORECASE)
_DATE_NAME_PATTERN = re.compile(r"(date|time|_at$|_dt$|year|month|day)", re.IGNORECASE)

# Column-cardinality thresholds used to distinguish a small set of category
# labels from a high-cardinality categorical (e.g. postcode) or free text.
CATEGORICAL_LOW_MAX_UNIQUE = 20
CATEGORICAL_HIGH_MAX_RATIO = 0.5
TEXT_MIN_AVG_LEN = 25
IDENTIFIER_MIN_UNIQUE_RATIO = 0.98


class ColumnRole(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL_LOW = "categorical_low"
    CATEGORICAL_HIGH = "categorical_high"
    DATETIME = "datetime"
    GEO_LAT = "geo_lat"
    GEO_LON = "geo_lon"
    TEXT = "text"
    IDENTIFIER = "identifier"
    CONSTANT = "constant"
    TARGET = "target"


@dataclass
class ColumnProfile:
    name: str
    role: ColumnRole
    dtype: str
    n_missing: int
    pct_missing: float
    n_unique: int


@dataclass
class SchemaResult:
    columns: dict[str, ColumnProfile]
    target: str | None
    task: str | None  # "regression" | "classification" | None
    geo_lat_col: str | None = None
    geo_lon_col: str | None = None
    datetime_cols: list[str] = field(default_factory=list)
    numeric_cols: list[str] = field(default_factory=list)
    categorical_low_cols: list[str] = field(default_factory=list)
    categorical_high_cols: list[str] = field(default_factory=list)
    text_cols: list[str] = field(default_factory=list)
    identifier_cols: list[str] = field(default_factory=list)
    constant_cols: list[str] = field(default_factory=list)

    @property
    def has_geo(self) -> bool:
        return self.geo_lat_col is not None and self.geo_lon_col is not None

    @property
    def feature_cols(self) -> list[str]:
        """Columns usable as model input: everything except the target,
        identifiers, constants and raw datetime (datetime is expanded into
        engineered parts elsewhere, not fed in raw)."""
        return [
            name
            for name, prof in self.columns.items()
            if name != self.target
            and prof.role
            not in (ColumnRole.IDENTIFIER, ColumnRole.CONSTANT, ColumnRole.DATETIME)
        ]


def _looks_like_datetime(series: pd.Series, name: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    sample = non_null.sample(min(50, len(non_null)), random_state=0)
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    parse_rate = parsed.notna().mean()
    if parse_rate >= 0.95:
        return True
    return bool(_DATE_NAME_PATTERN.search(name)) and parse_rate >= 0.7


def infer_schema(df: pd.DataFrame, target: str | None = None) -> SchemaResult:
    """Infer a role for every column in ``df``.

    Parameters
    ----------
    df:
        The raw dataframe, before any cleaning.
    target:
        Name of the target/label column, if this dataset will be modeled.
        Its role is always ``TARGET`` regardless of its dtype.
    """
    n_rows = len(df)
    columns: dict[str, ColumnProfile] = {}
    lat_col: str | None = None
    lon_col: str | None = None

    for name in df.columns:
        series = df[name]
        n_missing = int(series.isna().sum())
        pct_missing = n_missing / n_rows if n_rows else 0.0
        n_unique = int(series.nunique(dropna=True))
        dtype = str(series.dtype)

        if name == target:
            role = ColumnRole.TARGET
        elif n_unique <= 1:
            role = ColumnRole.CONSTANT
        elif _looks_like_datetime(series, name):
            role = ColumnRole.DATETIME
        elif pd.api.types.is_numeric_dtype(series):
            unique_ratio = n_unique / n_rows if n_rows else 0.0
            if _LAT_PATTERN.match(name) and series.between(-90, 90).mean() > 0.95:
                role = ColumnRole.GEO_LAT
            elif _LON_PATTERN.match(name) and series.between(-180, 180).mean() > 0.95:
                role = ColumnRole.GEO_LON
            elif (
                _ID_PATTERN.search(name)
                and unique_ratio >= IDENTIFIER_MIN_UNIQUE_RATIO
            ):
                role = ColumnRole.IDENTIFIER
            else:
                role = ColumnRole.NUMERIC
        else:
            unique_ratio = n_unique / n_rows if n_rows else 0.0
            avg_len = series.dropna().astype(str).str.len().mean() if n_unique else 0.0
            if unique_ratio >= IDENTIFIER_MIN_UNIQUE_RATIO and avg_len < TEXT_MIN_AVG_LEN:
                role = ColumnRole.IDENTIFIER
            elif n_unique <= CATEGORICAL_LOW_MAX_UNIQUE:
                role = ColumnRole.CATEGORICAL_LOW
            elif unique_ratio < CATEGORICAL_HIGH_MAX_RATIO and avg_len < TEXT_MIN_AVG_LEN:
                role = ColumnRole.CATEGORICAL_HIGH
            else:
                role = ColumnRole.TEXT

        columns[name] = ColumnProfile(
            name=name,
            role=role,
            dtype=dtype,
            n_missing=n_missing,
            pct_missing=pct_missing,
            n_unique=n_unique,
        )
        if role is ColumnRole.GEO_LAT:
            lat_col = name
        elif role is ColumnRole.GEO_LON:
            lon_col = name

    task = None
    if target is not None and target in df.columns:
        target_series = df[target].dropna()
        if pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 15:
            task = "regression"
        else:
            task = "classification"

    result = SchemaResult(
        columns=columns, target=target, task=task, geo_lat_col=lat_col, geo_lon_col=lon_col
    )
    for name, prof in columns.items():
        if prof.role is ColumnRole.NUMERIC:
            result.numeric_cols.append(name)
        elif prof.role is ColumnRole.CATEGORICAL_LOW:
            result.categorical_low_cols.append(name)
        elif prof.role is ColumnRole.CATEGORICAL_HIGH:
            result.categorical_high_cols.append(name)
        elif prof.role is ColumnRole.DATETIME:
            result.datetime_cols.append(name)
        elif prof.role is ColumnRole.TEXT:
            result.text_cols.append(name)
        elif prof.role is ColumnRole.IDENTIFIER:
            result.identifier_cols.append(name)
        elif prof.role is ColumnRole.CONSTANT:
            result.constant_cols.append(name)
    return result
