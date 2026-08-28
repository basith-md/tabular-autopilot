# Example: California Housing (regression + geospatial)

Predicts `median_house_value` for California census block groups.

- **Data**: `data/housing.csv` — the classic California Housing dataset (20,640 rows, 10 columns including `ocean_proximity` and lat/lon), vendored from the public [ageron/handson-ml2](https://github.com/ageron/handson-ml2) repository so this example has no live/personal data dependency.
- **Run it**: `python examples/california_housing/run_example.py`
- **What the pipeline does automatically here**: median-imputes `total_bedrooms`, log-transforms the right-skewed room/population columns, one-hot encodes `ocean_proximity`, clusters the lat/lon pairs into 8 geospatial regions and adds a distance-to-centroid feature, then compares 5 regression models and runs a full OLS diagnostic baseline (VIF pruning, Breusch-Pagan, Q-Q, residuals-vs-fitted).

## Results (best of 5 compared models)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Gradient Boosting (best)** | **$32,128** | **$47,866** | **0.825** |

The interpretable OLS baseline (with VIF-pruned features) is included in the full report for comparison and assumption-checking.

## Acknowledgment

This example was inspired by an earlier 5-author bootcamp group project ("California Home Price Prediction") that first explored this dataset with OLS regression and a small neural network. `tabular-autopilot` itself is new, solo, general-purpose work — it is not a copy of that project, and none of its code originates there — but the dataset choice and the diagnostic rigor (VIF, heteroscedasticity, Q-Q checks) are a nod to that earlier analysis.
