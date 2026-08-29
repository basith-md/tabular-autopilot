"""Standalone unsupervised segmentation: KMeans over every numeric column,
independent of geography or a modeling target.

Runs automatically whenever there's enough numeric signal to cluster on --
not gated on a target being present, since "what natural segments exist in
this data" is a question worth answering even in EDA-only mode. This is
deliberately separate from the geospatial clustering in
``feature_engineering.py`` (which is a modeling *feature*, fit only on a
lat/lon pair); this module answers "what segments exist in the data as a
whole" as a general profiling step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

MIN_ROWS_FOR_CLUSTERING = 20
MIN_NUMERIC_COLUMNS = 2
K_RANGE = range(2, 9)  # silhouette-searched cluster counts
RANDOM_STATE = 42


@dataclass
class ClusteringResult:
    k: int
    silhouette_score: float
    cluster_sizes: dict[int, int]
    labels: list[int] = field(default_factory=list)
    columns_used: list[str] = field(default_factory=list)


def _best_k_by_silhouette(scaled: np.ndarray) -> tuple[int, float, np.ndarray] | None:
    best: tuple[int, float, np.ndarray] | None = None
    for k in K_RANGE:
        if k >= len(scaled):
            break
        labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(scaled)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(scaled, labels))
        if best is None or score > best[1]:
            best = (k, score, labels)
    return best


def run_clustering(df: pd.DataFrame, numeric_cols: list[str]) -> ClusteringResult | None:
    cols = [c for c in numeric_cols if c in df.columns]
    if len(cols) < MIN_NUMERIC_COLUMNS or len(df) < MIN_ROWS_FOR_CLUSTERING:
        return None

    X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = X.loc[:, X.std(numeric_only=True) > 1e-8]
    if X.shape[1] < MIN_NUMERIC_COLUMNS:
        return None

    scaled = StandardScaler().fit_transform(X)
    best = _best_k_by_silhouette(scaled)
    if best is None:
        return None
    k, score, labels = best

    sizes = pd.Series(labels).value_counts().sort_index()
    return ClusteringResult(
        k=k,
        silhouette_score=score,
        cluster_sizes={int(i): int(c) for i, c in sizes.items()},
        labels=[int(v) for v in labels],
        columns_used=list(X.columns),
    )
