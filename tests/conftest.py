import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mixed_type_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
        {
            "row_id": np.arange(n),
            "price": rng.gamma(shape=2.0, scale=50_000, size=n),
            "rooms": rng.integers(1, 10, size=n),
            "city": rng.choice(["NYC", "LA", "Chicago", "Houston"], size=n),
            "notes": [
                f"This is a fairly long free-text note about record number {i} in the dataset."
                for i in range(n)
            ],
            "constant_col": ["same"] * n,
            "signup_date": pd.date_range("2022-01-01", periods=n, freq="D").astype(str),
            "lat": rng.uniform(32.0, 42.0, size=n),
            "lon": rng.uniform(-118.0, -87.0, size=n),
        }
    )


@pytest.fixture
def regression_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    cat = rng.choice(["A", "B", "C"], size=n)
    cat_effect = pd.Series(cat).map({"A": 0.0, "B": 5.0, "C": -3.0}).to_numpy()
    y = 10 + 3 * x1 - 2 * x2 + cat_effect + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "category": cat, "target": y})
    df.loc[rng.choice(n, size=15, replace=False), "x1"] = np.nan
    return df


@pytest.fixture
def classification_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    cat = rng.choice(["yes", "no"], size=n)
    logit = 1.5 * x1 - x2 + (cat == "yes") * 1.0
    prob = 1 / (1 + np.exp(-logit))
    label = (rng.uniform(size=n) < prob).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "flag": cat, "label": label})


@pytest.fixture
def imbalanced_classification_df() -> pd.DataFrame:
    """~9:1 class imbalance, well above the 1.5:1 threshold, with a real
    signal so class-weighting can plausibly change which model wins."""
    rng = np.random.default_rng(2)
    n_majority, n_minority = 270, 30
    x1_maj = rng.normal(loc=0.0, size=n_majority)
    x2_maj = rng.normal(loc=0.0, size=n_majority)
    x1_min = rng.normal(loc=2.5, size=n_minority)
    x2_min = rng.normal(loc=2.5, size=n_minority)
    x1 = np.concatenate([x1_maj, x1_min])
    x2 = np.concatenate([x2_maj, x2_min])
    label = np.concatenate([np.zeros(n_majority), np.ones(n_minority)]).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "label": label})


@pytest.fixture
def wide_regression_df() -> pd.DataFrame:
    """A handful of real signal columns plus many pure-noise columns, so
    feature selection has something meaningful to prune."""
    rng = np.random.default_rng(3)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 5 + 4 * x1 - 3 * x2 + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "target": y})
    for i in range(80):
        df[f"noise_{i}"] = rng.normal(size=n)
    return df
