# tabular-autopilot

**Point it at any CSV or Excel file and a target column — it profiles the data, cleans it, engineers features appropriate to whatever column types are present, compares 3–5 candidate models, runs full statistical diagnostics, and (if there's a datetime column) a separate time-series analysis. One command or one click; zero dataset-specific code.**

Built as a generalized, testable system rather than a one-off analysis notebook: the column-role inference, cleaning, feature engineering, and modeling logic is identical no matter which dataset you give it — the *decisions* it makes (what to impute, how to encode, which model wins) change with the data.

## Why this exists

Most portfolio data-science projects are a single notebook analyzing a single dataset. This is instead a small, tested **system**: an inference engine that looks at a dataframe, works out what kind of data is in each column (numeric, categorical, datetime, geospatial, free text, identifier), and routes each one through the right treatment — the same way a data scientist would triage an unfamiliar dataset by hand, but automatically and repeatably. See [`docs/methodology.md`](docs/methodology.md) for the full reasoning behind every decision it makes, including what it deliberately does *not* do yet.

## What it does, step by step

1. **Schema inference** — infers a role for every column using inspectable rules (not a black box): numeric, categorical (low/high cardinality), datetime, geospatial lat/lon pair, free text, identifier, or constant.
2. **Profiling** — missingness, skewness, IQR-based outlier counts, top category values, duplicate rows.
3. **Cleaning** — median imputation for numeric columns, mode imputation for categorical, automatic `log1p` transform for right-skewed columns, dropped rows with a missing target.
4. **Feature engineering** — one-hot encoding for low-cardinality categoricals, frequency encoding for high-cardinality ones, full calendar + cyclical (sin/cos) expansion for datetime columns, KMeans spatial clustering + centroid distance for lat/lon pairs.
5. **Modeling** — trains and compares **5 regression models** (Linear, Ridge, Lasso, Random Forest, Gradient Boosting) or **3 classification models** (Logistic Regression, Random Forest, Gradient Boosting) on an identical split and automatically picks the best by held-out score.
6. **Statistical diagnostics** — an interpretable OLS/Logit baseline alongside the model comparison: VIF-based multicollinearity pruning, coefficient significance, a Breusch-Pagan heteroscedasticity test, residual and Q-Q plots.
7. **Time-series diagnostics** *(auto-triggered)* — whenever a datetime column and numeric target are both present: trend strength, an Augmented Dickey-Fuller stationarity test, ACF/PACF, and a short-horizon forecast.
8. **Reporting** — a single self-contained HTML report (`tabular_autopilot.report`), or the same analysis live in a Streamlit app with a tab per section.

## Quickstart

```bash
pip install -e ".[app,dev]"

# CLI: analyze any CSV/Excel file, get an HTML report
python -m tabular_autopilot run examples/california_housing/data/housing.csv --target median_house_value

# Interactive app: upload your own file or pick a bundled example
streamlit run app.py
```

The Streamlit app is deploy-ready as-is on Streamlit Community Cloud or Hugging Face Spaces — point the platform at `app.py` and it's live, no code changes required.

## Worked examples (3 datasets, 3 data shapes)

| Example | Task | What it exercises | Best model | Result |
|---|---|---|---|---|
| [California Housing](examples/california_housing/) | Regression | Geospatial clustering, skew correction, VIF diagnostics | Gradient Boosting | R² = 0.825, MAE = $32,128 |
| [Titanic](examples/titanic_classification/) | Classification | Mixed types, heavy missingness, high-cardinality text/id columns | Logistic Regression | Accuracy = 0.805, ROC-AUC = 0.838 |
| [Retail Sales](examples/retail_sales_datetime/) | Regression + time series | Datetime feature expansion, auto-triggered stationarity/ACF/forecast | Ridge (CV) | R² = 0.884, trend-R² = 0.34 |

Each example folder has its own README with the full breakdown; run any of them with `python examples/<name>/run_example.py`.

## Project layout

```
src/tabular_autopilot/   the engine: schema, profiling, cleaning, feature_engineering,
                          modeling, timeseries, eda_visuals, report, pipeline, cli
app.py                   Streamlit UI on top of the same pipeline — no separate logic
templates/                Jinja2 HTML report template
examples/                 3 worked examples spanning regression, classification, time series
tests/                    unit + end-to-end tests, synthetic edge cases for every column role
docs/methodology.md       step-by-step explanation of every automated decision, plus a
                          deliberate "known gaps" section (chi-square/ANOVA, PCA/clustering,
                          true ARIMA, intervention analysis — not yet automated)
.github/workflows/        CI (lint + tests) and an examples workflow that runs the pipeline
                          against all 3 datasets on every push, uploading the HTML reports
                          as build artifacts
```

## Testing

```bash
pytest -q
```

26 tests cover schema inference for every column role (numeric, categorical, datetime, geospatial, text, identifier, constant), cleaning, feature engineering, both baseline+comparison modeling paths, time-series diagnostics, and 6 full end-to-end pipeline runs including edge cases (all-numeric data, EDA-only with no target, a geospatial dataset, a datetime+time-series dataset).

## Acknowledgment

Inspired by an earlier 5-author bootcamp group project analyzing California housing prices, and by a university predictive-analytics course covering regression, trees, regularization, and time-series methods. `tabular-autopilot` is new, solo, general-purpose work built on top of that experience — see [`examples/california_housing/README.md`](examples/california_housing/README.md) and [`docs/methodology.md`](docs/methodology.md) for specifics on what carried over as design inspiration versus what's genuinely new here.

## License

MIT — see [LICENSE](LICENSE).
