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
