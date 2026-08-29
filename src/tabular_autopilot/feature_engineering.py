"""Role-driven feature engineering.

Each column role gets a fixed, generalizable treatment instead of
dataset-specific hand-crafted features:

- ``categorical_low``  -> one-hot encoding (few enough levels to be safe).
- ``categorical_high`` -> frequency encoding (avoids one-hot blow-up).
- ``datetime``         -> calendar parts (year/month/day/day-of-week/
                          is_weekend) plus cyclical sin/cos encodings of
                          month and day-of-week.
- geospatial lat/lon pair -> KMeans spatial cluster id + distance from the
                          point to its cluster centroid.
- ``text``             -> TF-IDF vectorized (top terms, capped) when there's
                          enough data for it to be meaningful; otherwise
                          dropped, same as ``identifier``/``constant``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from tabular_autopilot.schema import SchemaResult

N_GEO_CLUSTERS = 8
MAX_TEXT_COLUMNS_VECTORIZED = 2
MAX_TEXT_FEATURES_PER_COL = 20
MIN_ROWS_FOR_TEXT_VECTORIZATION = 20


HIGH_CARDINALITY_ENCODINGS = ("frequency", "target")


@dataclass
class FeatureEngineeringReport:
    one_hot_encoded: list[str] = field(default_factory=list)
    frequency_encoded: list[str] = field(default_factory=list)
    datetime_expanded: dict[str, list[str]] = field(default_factory=dict)
    geo_features_added: list[str] = field(default_factory=list)
    text_vectorized: dict[str, list[str]] = field(default_factory=dict)
    dropped_columns: list[str] = field(default_factory=list)
    final_feature_columns: list[str] = field(default_factory=list)
    deferred_target_encoding: list[str] = field(default_factory=list)


def _expand_datetime(df: pd.DataFrame, col: str) -> list[str]:
    dt = pd.to_datetime(df[col], errors="coerce", format="mixed")
    new_cols = []
    for part in ("year", "month", "day", "dayofweek"):
        name = f"{col}_{part}"
        df[name] = getattr(dt.dt, part)
        new_cols.append(name)
    weekend_col = f"{col}_is_weekend"
    df[weekend_col] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    new_cols.append(weekend_col)
    for part, period in (("month", 12), ("dayofweek", 7)):
        base = df[f"{col}_{part}"].astype(float)
        sin_col, cos_col = f"{col}_{part}_sin", f"{col}_{part}_cos"
        df[sin_col] = np.sin(2 * np.pi * base / period)
        df[cos_col] = np.cos(2 * np.pi * base / period)
        new_cols += [sin_col, cos_col]
    for name in new_cols:
        df[name] = df[name].fillna(df[name].median() if df[name].notna().any() else 0)
    return new_cols


def _add_geo_features(df: pd.DataFrame, lat_col: str, lon_col: str) -> list[str]:
    coords = df[[lat_col, lon_col]].dropna()
    if len(coords) < N_GEO_CLUSTERS:
        return []
    kmeans = KMeans(n_clusters=N_GEO_CLUSTERS, n_init=10, random_state=42)
    labels = kmeans.fit_predict(coords)
    cluster_col, dist_col = "geo_cluster", "geo_dist_to_centroid"
    df[cluster_col] = np.nan
    df.loc[coords.index, cluster_col] = labels
    df[cluster_col] = df[cluster_col].fillna(-1).astype(int)

    centroids = kmeans.cluster_centers_
    dists = np.linalg.norm(coords.values - centroids[labels], axis=1)
    df[dist_col] = np.nan
    df.loc[coords.index, dist_col] = dists
    df[dist_col] = df[dist_col].fillna(df[dist_col].median() if df[dist_col].notna().any() else 0)
    return [cluster_col, dist_col]


def _vectorize_text(df: pd.DataFrame, col: str, max_features: int = MAX_TEXT_FEATURES_PER_COL) -> list[str]:
    texts = df[col].fillna("").astype(str)
    if texts.str.strip().eq("").all():
        return []
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english", min_df=2)
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return []  # e.g. empty vocabulary after stopword removal
    terms = list(vectorizer.get_feature_names_out())
    if not terms:
        return []
    dense = matrix.toarray()
    for i, term in enumerate(terms):
        df[f"{col}_tfidf_{term}"] = dense[:, i]
    return terms


def engineer_features(
    df: pd.DataFrame, schema: SchemaResult, vectorize_text: bool = True, high_cardinality_encoding: str = "frequency"
) -> tuple[pd.DataFrame, FeatureEngineeringReport]:
    if high_cardinality_encoding not in HIGH_CARDINALITY_ENCODINGS:
        raise ValueError(f"high_cardinality_encoding must be one of {HIGH_CARDINALITY_ENCODINGS}")
    # Target encoding needs a train/test split (to stay leak-free), which
    # doesn't exist yet at this generic, target-agnostic stage -- so it's
    # only deferred to modeling.py when there's actually a target to encode
    # against; otherwise frequency encoding (which needs no target) applies.
    use_target_encoding = high_cardinality_encoding == "target" and schema.target is not None

    out = df.copy()
    report = FeatureEngineeringReport()

    for col in schema.categorical_low_cols:
        if col not in out.columns:
            continue
        dummies = pd.get_dummies(out[col], prefix=col, dtype=int)
        out = pd.concat([out.drop(columns=[col]), dummies], axis=1)
        report.one_hot_encoded.append(col)

    for col in schema.categorical_high_cols:
        if col not in out.columns:
            continue
        if use_target_encoding:
            report.deferred_target_encoding.append(col)
            continue
        freq = out[col].value_counts(normalize=True)
        out[f"{col}_freq"] = out[col].map(freq).fillna(0.0)
        out = out.drop(columns=[col])
        report.frequency_encoded.append(col)

    for col in schema.datetime_cols:
        if col not in out.columns:
            continue
        new_cols = _expand_datetime(out, col)
        report.datetime_expanded[col] = new_cols
        out = out.drop(columns=[col])

    if schema.has_geo:
        geo_cols = _add_geo_features(out, schema.geo_lat_col, schema.geo_lon_col)
        report.geo_features_added = geo_cols

    if vectorize_text and len(out) >= MIN_ROWS_FOR_TEXT_VECTORIZATION:
        for col in schema.text_cols[:MAX_TEXT_COLUMNS_VECTORIZED]:
            if col not in out.columns:
                continue
            terms = _vectorize_text(out, col)
            if terms:
                report.text_vectorized[col] = terms
                out = out.drop(columns=[col])

    to_drop = [c for c in schema.identifier_cols + schema.constant_cols + schema.text_cols if c in out.columns]
    if to_drop:
        out = out.drop(columns=to_drop)
        report.dropped_columns = to_drop

    report.final_feature_columns = [c for c in out.columns if c != schema.target]
    return out, report
