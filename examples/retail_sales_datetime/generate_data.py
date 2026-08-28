"""Generate a synthetic retail sales dataset with trend + weekly seasonality.

Fully synthetic and deterministic (fixed seed) so this example has zero
external download dependency, unlike the other two examples which vendor
real public CSVs. Demonstrates the datetime feature-engineering and
time-series diagnostics paths on data shaped like a real retail feed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_PATH = Path(__file__).parent / "data" / "retail_sales.csv"


def generate(n_days: int = 730, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    regions = rng.choice(["North", "South", "East", "West"], size=n_days)
    categories = rng.choice(["Grocery", "Electronics", "Apparel", "Home"], size=n_days)
    promo = rng.choice([0, 1], size=n_days, p=[0.8, 0.2])

    day_index = np.arange(n_days)
    trend = 50 + 0.05 * day_index
    weekly_seasonality = 15 * np.sin(2 * np.pi * day_index / 7)
    promo_effect = promo * rng.uniform(10, 30, size=n_days)
    noise = rng.normal(scale=6.0, size=n_days)

    units_sold = np.clip(trend + weekly_seasonality + promo_effect + noise, 0, None).round(1)
    price = np.round(rng.uniform(5, 120, size=n_days), 2)

    return pd.DataFrame(
        {
            "order_date": dates.astype(str),
            "region": regions,
            "category": categories,
            "price": price,
            "promo_flag": promo,
            "units_sold": units_sold,
        }
    )


if __name__ == "__main__":
    df = generate()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")
