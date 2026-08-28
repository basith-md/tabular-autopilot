# Methodology

This document walks through every automated decision `tabular-autopilot` makes, in the order it makes them, and explains *why* — plus an honest list of what is intentionally out of scope for v0.1.

## 1. Schema inference (`schema.py`)

Every other stage depends on a role assigned to each column. Roles are inferred with simple, inspectable rules rather than a black-box classifier, so behavior is predictable and debuggable:

| Role | Rule |
|---|---|
| `datetime` | Already a datetime dtype, or a sample of values parses as dates at ≥95% (or ≥70% when the column name itself looks date-like: `date`, `time`, `_at`, `year`, ...). |
| `geo_lat` / `geo_lon` | Column name matches `lat`/`latitude` or `lon`/`lng`/`longitude` **and** ≥95% of values fall in the valid range ([-90,90] or [-180,180]). |
| `identifier` | Name looks like an id (`id`, `*_id`, `uuid`, ...) and ≥98% of values are unique, **or** (for text columns) ≥98% unique with short average length — i.e. it looks like a key, not free text. |
| `constant` | Exactly one distinct value — carries no information. |
| `numeric` | Numeric dtype, none of the above. |
| `categorical_low` | Non-numeric, ≤20 distinct values. |
| `categorical_high` | Non-numeric, more distinct values but still <50% unique and short average string length (e.g. postcodes, ticket numbers). |
| `text` | Everything else non-numeric — long strings or very high uniqueness without looking like an id. |

A `target` role is assigned to whichever column the caller names, regardless of its dtype, and the **task** (regression vs. classification) is inferred from it: numeric with >15 distinct values → regression, everything else → classification.

## 2. Profiling (`profiling.py`)

For every numeric column: mean/std/median, skewness, and IQR-based outlier count. For every categorical column: top-10 value counts. Missingness and duplicate-row counts are computed dataset-wide. This report is what drives the cleaning decisions in the next stage — nothing here is dataset-specific.

## 3. Cleaning (`cleaning.py`)

- **Numeric**: median imputation for missing values by default (mean is a configurable alternative — median is more robust to outliers, which is why it's the default rather than the more commonly-taught mean). Columns flagged as skewed by the profiler (`|skew| ≥ 1`) and non-negative get a `log1p` companion column, so the report can show before/after distributions without destroying the original scale.
- **Categorical**: mode imputation; falls back to an explicit `"Unknown"` category if a column is entirely missing.
- **Target**: rows with a missing target are dropped (nothing to learn from or evaluate against).

## 4. Feature engineering (`feature_engineering.py`)

- `categorical_low` → one-hot encoding.
- `categorical_high` → frequency encoding (avoids a one-hot explosion for things like postcodes or ticket numbers).
- `datetime` → expanded into `year`, `month`, `day`, `dayofweek`, `is_weekend`, plus cyclical `sin`/`cos` encodings of month and day-of-week (so "December" and "January" are recognized as adjacent). The raw datetime column is then dropped — models don't take raw timestamps.
- **Geospatial pair** (lat+lon both detected) → KMeans into 8 spatial clusters, plus a distance-from-point-to-its-cluster-centroid feature.
- `identifier` / `constant` / `text` → dropped from the feature matrix. They aren't useful (or are out of scope — see below) as model input.

## 5. Modeling (`modeling.py`)

### 5a. Model comparison

Rather than committing to one algorithm, the pipeline fits several candidates on an identical train/test split and reports all of them, picking the best by held-out score:

- **Regression**: Linear Regression, Ridge (CV-tuned), Lasso (CV-tuned), Random Forest, Gradient Boosting (`HistGradientBoostingRegressor`). Linear models are fit on standardized features; trees are not (scale-invariant).
- **Classification**: Logistic Regression, Random Forest, Gradient Boosting (`HistGradientBoostingClassifier`).

This mirrors the breadth of a full predictive-analytics course sequence — OLS → regularization → trees → ensembles — as a comparison rather than a single fixed choice.

### 5b. Interpretable statistical baseline & diagnostics

Independent of which model wins the comparison, an OLS (regression) or Logit (binary classification) baseline is always fit for interpretability, with the same rigor as a from-scratch statistics course:

1. **VIF-based multicollinearity pruning** — iteratively drop the feature with the highest variance inflation factor until all remaining features are below 10 (or only one remains).
2. **Coefficients & p-values** from the full `statsmodels` summary.
3. **Breusch-Pagan test** for heteroscedasticity (regression only) — flagged explicitly in the report if p < 0.05, since it means the OLS standard errors are unreliable even if the point estimates are fine.
4. **Residuals-vs-fitted and Q-Q plots**, for visual linearity/normality checks.

Multiclass targets skip the interpretable baseline (Logit doesn't generalize cleanly past binary without multinomial machinery) and are noted as such in the report rather than silently omitted.

### 5c. Feature importance

Computed via **permutation importance** on the held-out test set against whichever model won the comparison — this works identically regardless of model type (unlike relying on a model-specific `.feature_importances_` attribute, which HistGradientBoosting doesn't even expose).

### 5d. Explaining a tree-ensemble winner

Random Forest and Gradient Boosting are ensembles of tens to hundreds of trees averaged together — there is no single tree to draw that represents the actual production model. When one of them wins the comparison, a separate shallow decision tree (max depth 3) of the same task is fit fresh on the same training split, purely to illustrate the *kind* of split logic that ensemble is built from. It is explicitly labeled as an illustration, not the model itself, in both the report and the browser demo.

### 5e. User-configurable settings

Three knobs are exposed all the way from `run_pipeline()` down to the browser demo's settings panel, each defaulting to the original fixed behavior so nothing changes unless asked: **test split size** (`test_size`, default 0.2), **which candidate models to include** (`model_names`, default: all of them for the detected task), and **numeric imputation strategy** (`numeric_impute_strategy`, `"median"` or `"mean"`, default median).

## 6. Time-series diagnostics (`timeseries.py`)

Triggered automatically — independent of the main tabular model — whenever the dataset has **both** a datetime column and a numeric target, with at least 20 observations after aggregating by date:

1. Linear trend regression against a time index, reported as trend-R² (how much of the variance is explained by a straight-line trend alone).
2. **Augmented Dickey-Fuller test** for stationarity.
3. **ACF/PACF** at up to 24 lags.
4. A 12-period **Exponential Smoothing (Holt's method)** forecast, plotted with the observed history.

This runs *alongside* the tabular model, not instead of it — the tabular model may still use the date as engineered features (day-of-week, month, etc.) for a cross-sectional-style fit, while this section specifically evaluates the column as a genuine time series.

## Provenance and scope decisions

This tool's design was informed by two prior bodies of work, neither of which it reuses code from directly:

- A **5-author bootcamp group project** analyzing California housing prices (OLS + a small neural net), which is why the geospatial clustering, VIF/heteroscedasticity/Q-Q diagnostic rigor, and the choice of California Housing as the flagship example exist.
- A **university predictive-analytics course** (~21 R scripts spanning OLS, logistic regression, decision trees, random forests, best-subset/stepwise selection, ridge/lasso regularization, and seven time-series scripts covering stationarity, ACF/PACF, AR/MA simulation, and interrupted-time-series/intervention regression). That breadth is why the model-comparison step exists instead of a single hardcoded algorithm, and why time-series diagnostics are a first-class, automatically-triggered path rather than an afterthought.

### Known gaps / roadmap

Deliberately out of scope for v0.1 (flagged here rather than silently missing):

- **Formal hypothesis/association tests** beyond regression-coefficient significance — no chi-square test of independence, ANOVA, Shapiro-Wilk normality test, or non-parametric tests (Wilcoxon/Mann-Whitney). Normality is currently assessed only visually (Q-Q/residual plots).
- **Unsupervised methods** — no PCA/dimensionality reduction, no standalone clustering report (KMeans is used internally only for geospatial features, not exposed as a general-purpose EDA step).
- **True ARIMA/SARIMA modeling** — the current forecast uses Exponential Smoothing rather than `statsmodels`' ARIMA/SARIMAX with formal order selection; there's no seasonal decomposition or seasonal ARIMA for multi-seasonality data.
- **Interrupted time series / intervention analysis** — detecting a level-shift around a known event date (e.g., a policy change or product launch) is a distinct, valuable technique that was not automated here.
- **Text columns** are detected and reported on, but not analyzed beyond that (no NLP features, embeddings, or sentiment).

These are natural v0.2 candidates and are listed here rather than glossed over, since knowing what a tool *doesn't* do is as important as knowing what it does.
