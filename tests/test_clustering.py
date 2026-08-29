import time

import numpy as np
import pandas as pd

from tabular_autopilot.clustering import run_clustering


def test_clustering_finds_three_well_separated_blobs():
    rng = np.random.default_rng(0)
    n_per = 40
    centers = [(0, 0), (10, 10), (-10, 10)]
    rows = []
    for cx, cy in centers:
        rows.append(
            pd.DataFrame({"x": rng.normal(cx, 0.5, n_per), "y": rng.normal(cy, 0.5, n_per)})
        )
    df = pd.concat(rows, ignore_index=True)

    result = run_clustering(df, ["x", "y"])

    assert result is not None
    assert result.k == 3
    assert result.silhouette_score > 0.5
    assert sum(result.cluster_sizes.values()) == len(df)
    assert len(result.labels) == len(df)


def test_clustering_returns_none_for_too_few_rows():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    assert run_clustering(df, ["x", "y"]) is None


def test_clustering_returns_none_for_fewer_than_two_numeric_columns():
    df = pd.DataFrame({"x": np.random.default_rng(0).normal(size=50)})
    assert run_clustering(df, ["x"]) is None


def test_clustering_ignores_constant_columns():
    rng = np.random.default_rng(1)
    n = 60
    df = pd.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n), "constant": [5] * n})
    result = run_clustering(df, ["x", "y", "constant"])
    assert result is not None
    assert "constant" not in result.columns_used


def test_clustering_stays_fast_on_a_large_dataset():
    """Regression guard: silhouette_score defaults to a full O(n^2) pairwise
    distance matrix when scored on every row. run_clustering must score each
    candidate k on a bounded sample instead, or this blows up on datasets
    much bigger than the tiny fixtures above (discovered via a real 54k-row
    dataset, where it took ~200s before being capped)."""
    rng = np.random.default_rng(2)
    n = 20_000
    centers = [(0, 0, 0), (15, 15, 15), (-15, 15, -15)]
    rows = []
    for cx, cy, cz in centers:
        n_per = n // len(centers)
        rows.append(
            pd.DataFrame(
                {"x": rng.normal(cx, 1.0, n_per), "y": rng.normal(cy, 1.0, n_per), "z": rng.normal(cz, 1.0, n_per)}
            )
        )
    df = pd.concat(rows, ignore_index=True)

    start = time.time()
    result = run_clustering(df, ["x", "y", "z"])
    elapsed = time.time() - start

    assert result is not None
    assert result.k == 3
    assert elapsed < 20, f"run_clustering took {elapsed:.1f}s on 20k rows -- silhouette sampling may have regressed"
