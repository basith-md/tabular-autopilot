# Methodology

This document walks through every automated decision Tabular Autopilot makes, in the order it makes them, and explains *why* — plus an honest list of what is intentionally out of scope for v0.1.

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

For every numeric column: mean/std/median, skewness, and IQR-based outlier count. For every categorical column: top-10 value counts. Missingness and duplicate-row counts are computed dataset-wide. Every numeric-column pair with |Pearson r| ≥ 0.9 is also flagged as a **redundant pair** — independent of which model ends up winning, two columns that move almost interchangeably (e.g. `total_bedrooms` and `households` in the California Housing example) are worth knowing about before modeling even starts. This report is what drives the cleaning decisions in the next stage — nothing here is dataset-specific.

## 3. Segments & association tests (`clustering.py`, `hypothesis_tests.py`)

Two independent, always-on analyses that run on the cleaned data, before modeling commits to anything:

- **Standalone clustering**: KMeans over every numeric column (scaled first), with the cluster count chosen automatically by silhouette score over k = 2..8 — the best-scoring k wins, no fixed default. This is deliberately separate from the geospatial clustering in feature engineering (which only runs on a lat/lon pair and feeds the model); this one answers "what natural segments exist in the data as a whole," independent of geography or any target, and runs even in EDA-only mode. Requires at least 2 usable numeric columns and 20 rows; returns nothing otherwise rather than forcing a meaningless split. KMeans itself fits on every row, but each candidate k is *scored* on a fixed 5,000-row random sample rather than the full dataset — `silhouette_score` computes a full pairwise-distance matrix otherwise, an O(n²) cost that took ~200s on a real 54,000-row dataset before this cap (discovered by testing against real data, not synthetic fixtures).
- **Formal association tests** *(classification targets only)*: every categorical feature gets a **chi-square test of independence** against the target; every numeric feature gets either a **one-way ANOVA F-test** (three or more classes) or a **Mann-Whitney U test** (exactly two classes) — Mann-Whitney specifically instead of ANOVA for the binary case, since ANOVA on two groups just reduces to a less robust version of the same question. On the Titanic example this reproduces exactly the associations a manual EDA would find: `Sex` and `Embarked` significant by chi-square, `Pclass`/`Fare`/`Parch`/`SibSp` significant by Mann-Whitney, `Age` and `PassengerId` not significant.

## 4. Cleaning (`cleaning.py`)

- **Numeric**: median imputation for missing values by default (mean is a configurable alternative — median is more robust to outliers, which is why it's the default rather than the more commonly-taught mean). Columns flagged as skewed by the profiler (`|skew| ≥ 1`) and non-negative get a `log1p` companion column, so the report can show before/after distributions without destroying the original scale.
- **Outlier capping** *(opt-in, off by default)*: when enabled, numeric values are clipped to the IQR fence (`Q1 - 1.5×IQR`, `Q3 + 1.5×IQR`) instead of just being counted. Off by default because capping changes the data — it's a deliberate choice, not a silent default.
- **Categorical**: mode imputation; falls back to an explicit `"Unknown"` category if a column is entirely missing.
- **Target**: rows with a missing target are dropped (nothing to learn from or evaluate against).

## 5. Feature engineering (`feature_engineering.py`)

- `categorical_low` → one-hot encoding.
- `categorical_high` → **frequency encoding by default**, or **out-of-fold target encoding** when explicitly selected (see 6f) — never both.
- `datetime` → expanded into `year`, `month`, `day`, `dayofweek`, `is_weekend`, plus cyclical `sin`/`cos` encodings of month and day-of-week (so "December" and "January" are recognized as adjacent). The raw datetime column is then dropped — models don't take raw timestamps.
- **Geospatial pair** (lat+lon both detected) → KMeans into 8 spatial clusters, plus a distance-from-point-to-its-cluster-centroid feature. The report's geospatial chart plots the point cloud at an equal aspect ratio (so the shape reads correctly instead of stretching) with two panels — one colored by the target, one colored by the `geo_cluster` id — so the chart shows what the clustering step actually did, not just the raw target again.
- `identifier` / `constant` / `text` → dropped from the feature matrix. They aren't useful (or are out of scope — see below) as model input.
- **Feature separability snapshot**: once a target is present, a 2D PCA projection of every engineered feature is plotted, colored by target/class — a cheap "does this data separate at all" visual before committing to any one model, most useful once TF-IDF has added dozens of columns.

## 6. Modeling (`modeling.py`)

### 6a. Model comparison

Rather than committing to one algorithm, the pipeline fits several candidates on an identical train/test split and reports all of them, picking the best by held-out score:

- **Regression**: Linear Regression, Ridge (CV-tuned), Lasso (CV-tuned), Random Forest, Gradient Boosting (`HistGradientBoostingRegressor`). Linear models are fit on standardized features; trees are not (scale-invariant).
- **Classification**: Logistic Regression, Random Forest, Gradient Boosting (`HistGradientBoostingClassifier`).

This mirrors the breadth of a full predictive-analytics course sequence — OLS → regularization → trees → ensembles — as a comparison rather than a single fixed choice.

### 6b. Interpretable statistical baseline & diagnostics

Independent of which model wins the comparison, an OLS (regression) or Logit (binary classification) baseline is always fit for interpretability, with the same rigor as a from-scratch statistics course:

1. **VIF-based multicollinearity pruning** — iteratively drop the feature with the highest variance inflation factor until all remaining features are below 10 (or only one remains).
2. **Coefficients & p-values** from the full `statsmodels` summary.
3. **Breusch-Pagan test** for heteroscedasticity (regression only) — flagged explicitly in the report if p < 0.05, since it means the OLS standard errors are unreliable even if the point estimates are fine.
4. **Shapiro-Wilk test** for normality of the residuals (regression only) — a formal complement to the visual Q-Q/residuals-vs-fitted plots below, flagged if p < 0.05.
5. **Residuals-vs-fitted and Q-Q plots**, for visual linearity/normality checks.

Multiclass targets skip the interpretable baseline (Logit doesn't generalize cleanly past binary without multinomial machinery) and are noted as such in the report rather than silently omitted.

### 6c. Feature importance

Computed via **permutation importance** on the held-out test set against whichever model won the comparison — this works identically regardless of model type (unlike relying on a model-specific `.feature_importances_` attribute, which HistGradientBoosting doesn't even expose).

### 6d. Explaining a tree-ensemble winner

Random Forest and Gradient Boosting are ensembles of tens to hundreds of trees averaged together — there is no single tree to draw that represents the actual production model. When one of them wins the comparison, a separate shallow decision tree (max depth 3) of the same task is fit fresh on the same training split, purely to illustrate the *kind* of split logic that ensemble is built from. It is explicitly labeled as an illustration, not the model itself, in both the report and the browser demo.

### 6e. User-configurable settings

Three knobs are exposed all the way from `run_pipeline()` down to the browser demo's settings panel, each defaulting to the original fixed behavior so nothing changes unless asked: **test split size** (`test_size`, default 0.2), **which candidate models to include** (`model_names`, default: all of them for the detected task), and **numeric imputation strategy** (`numeric_impute_strategy`, `"median"` or `"mean"`, default median).

### 6f. Text encoding (TF-IDF)

Free-text columns were previously dropped outright. Now, when there's enough data for it to be meaningful (≥20 rows) and text vectorization is enabled, up to 2 text columns are TF-IDF vectorized (English stop words removed, top 20 terms per column by default) and the resulting term columns feed into modeling like any other numeric feature. This is fit on the full cleaned dataset before the train/test split — the same mild-leakage tolerance already accepted for one-hot/frequency encoding elsewhere in the pipeline, not a new exception.

### 6g. Class imbalance (classification)

Target class counts are always computed and reported; a majority:minority ratio ≥ 1.5 is flagged as imbalanced. When enabled (the default), imbalance is addressed via `class_weight="balanced"` for Logistic Regression and Random Forest (both accept it as a constructor argument), and via `sample_weight` at fit time for Gradient Boosting (`HistGradientBoostingClassifier` has no `class_weight` parameter). A **balanced accuracy** metric is always reported alongside accuracy/F1/ROC-AUC regardless of whether weighting is applied, since plain accuracy is misleading on imbalanced targets even when you're not actively correcting for it.

### 6h. Automatic feature selection

After feature engineering (and before modeling), near-zero-variance columns are dropped, and — if more than 50 features remain — `SelectKBest` (ANOVA F-test: `f_regression` or `f_classif`) keeps only the top 50, fit on the training split only so the test split can't leak into which features are chosen. This matters most once TF-IDF vectorization is in play, since it can add dozens of columns per text field.

### 6i. Cross-validation as an alternative to a single split

Off by default (it multiplies runtime by the fold count, a real cost in the browser demo specifically). When enabled, each candidate model's headline metrics become k-fold cross-validation means over the full dataset instead of a single train/test split's numbers. Diagnostics that need one concrete fitted model — the confusion matrix, permutation importance, and the illustrative tree — still come from a single held-out split, refit after the winner is chosen; the report is explicit about which numbers come from which source so the two aren't conflated.

### 6j. Configurable model hyperparameters

Each candidate exposes the one knob a practitioner would reach for first, all defaulting to today's fixed values so nothing changes unless touched: **Ridge (CV)** and **Lasso (CV)** take an alpha search range (they already CV-tune within it internally — the *chosen* alpha is always reported, not just the range); **Logistic Regression** takes its inverse regularization strength `C`; **Random Forest** takes tree count and max depth; **Gradient Boosting** takes learning rate and boosting-round count. Linear Regression has no meaningful knob and is left alone.

### 6k. Optional hyperparameter search — narrow or broad

Off by default. When enabled, Random Forest, Gradient Boosting, and Logistic Regression are each wrapped in a search over the configured knob(s) above, and the winning combination is reported alongside the metrics (Ridge/Lasso are skipped here since `RidgeCV`/`LassoCV` already search their own alpha range by construction):

- **Narrow (default)**: a small 3-fold `GridSearchCV` over a 2-3 value grid centered on the configured value (e.g. Random Forest searches `{n_estimators/2, n_estimators, n_estimators×2} × {max_depth, 8, 16}`).
- **Broad (opt-in)**: a 3-fold `RandomizedSearchCV` sampling 10 combinations from a wider distribution per model that also tunes parameters beyond the one narrow knob — e.g. Random Forest additionally searches `min_samples_leaf` and `max_features`; Gradient Boosting additionally searches tree depth and max leaf nodes. Sampled rather than exhaustively evaluated, so widening the ranges doesn't blow up the fit count.

Either mode is automatically disabled whenever cross-validation (6i) is also on — nesting a search inside k-fold CV multiplies the fit count by both the fold count *and* the search size, which is too slow for the in-browser demo specifically.

### 6l. Model-comparison significance (Wilcoxon signed-rank)

Available only under cross-validation (6i), where each candidate has a per-fold score to compare. The winning model's per-fold scores are compared against the runner-up's with a **Wilcoxon signed-rank test** — a paired, non-parametric test appropriate for the small, matched samples k-fold CV produces (the same fold split underlies every model's score, so the comparison is paired by construction). This answers a question the metrics table alone can't: is the top model *actually* better, or just luck of the fold split?

### 6m. Out-of-fold target encoding for high-cardinality categoricals

An alternative to the frequency encoding described in section 5, selectable per run. When chosen, `feature_engineering.py` leaves the high-cardinality column raw instead of encoding it, and `modeling.py` encodes it *after* the train/test split: each training row's encoded value comes from a 5-fold mean-target computation that excludes the fold it belongs to (so a row can't see its own contribution to its own encoding), and the test split is encoded from the full training mapping. This is the standard leakage-safe recipe for target encoding — frequency encoding remains the default since it needs no target and no fold machinery. One simplification worth naming: under cross-validation (6i), the encoding used across all folds is still the one derived from the single held-out split, not re-derived per fold — a reasonable trade-off given the added complexity of properly nesting target encoding inside CV, but worth knowing about.

## 7. Time-series diagnostics (`timeseries.py`)

Triggered automatically — independent of the main tabular model — whenever the dataset has **both** a datetime column and a numeric target, with at least 20 observations after aggregating by date. Aggregation always floors the datetime column to the calendar day before grouping, regardless of whether it carries a time-of-day component — a column of per-event timestamps (ride pickups, order times, ...) would otherwise group by the exact instant, silently turning "N observations" into "N rows" and feeding ADF/ACF/ARIMA raw per-event noise instead of a real daily series (caught by testing against a real per-ride taxi dataset, where this took observation counts from 6,414 raw rows down to the true 32 calendar days):

1. Linear trend regression against a time index, reported as trend-R² (how much of the variance is explained by a straight-line trend alone).
2. **Augmented Dickey-Fuller test** for stationarity — its result (stationary vs. not) sets `d` (0 or 1) for the ARIMA order search below.
3. **ACF/PACF** at up to 24 lags.
4. **ARIMA/SARIMA order selection**: a bounded grid search over `p, q ∈ {0, 1, 2}` at the ADF-implied `d`, keeping whichever `(p, d, q)` has the lowest AIC. If the data is daily and there's at least two weeks of it, one additional seasonal candidate — the same `(p, d, q)` plus a weekly seasonal term, `SARIMAX(p, d, q) × (1, 1, 1, 7)` — is fit and kept instead if its AIC is lower. This is a deliberately bounded stand-in for full auto-ARIMA order selection (see known gaps below), not an exhaustive search. Whichever model wins forecasts 12 periods ahead. If nothing in the grid converges, the pipeline falls back to the original **Exponential Smoothing (Holt's method)** forecast rather than failing outright, and says so in the report.
5. **Interrupted time series / intervention analysis** *(opt-in — needs a date)*: given an `intervention_date`, a segmented regression `y ~ t + post + (t - t₀)·post` is fit, where `post` flags observations on/after that date and `t₀` is its time index. The `post` coefficient is the level shift right at the event; the interaction term is the slope change afterward — both reported with p-values, plus a chart showing the fitted pre/post segments against the observed history. There's no way to auto-detect "the" event in an arbitrary dataset, so this only runs when a date is supplied (CLI/Streamlit/browser-demo settings panel).

This runs *alongside* the tabular model, not instead of it — the tabular model may still use the date as engineered features (day-of-week, month, etc.) for a cross-sectional-style fit, while this section specifically evaluates the column as a genuine time series.

## Provenance and scope decisions

This tool's design was informed by two prior bodies of work, neither of which it reuses code from directly:

- A **5-author bootcamp group project** analyzing California housing prices (OLS + a small neural net), which is why the geospatial clustering, VIF/heteroscedasticity/Q-Q diagnostic rigor, and the choice of California Housing as the flagship example exist.
- A **university predictive-analytics course** (~21 R scripts spanning OLS, logistic regression, decision trees, random forests, best-subset/stepwise selection, ridge/lasso regularization, and seven time-series scripts covering stationarity, ACF/PACF, AR/MA simulation, and interrupted-time-series/intervention regression). That breadth is why the model-comparison step exists instead of a single hardcoded algorithm, why time-series diagnostics are a first-class, automatically-triggered path rather than an afterthought, and — after cross-checking this pipeline against that course's full scope — why formal hypothesis tests (3), ARIMA/SARIMA order selection, and interrupted-time-series analysis (7) were added as the remaining pieces of that coursework's breadth.
- Kevin Markham's **"Master Machine Learning with scikit-learn"** (20-chapter book/course on the full sklearn workflow) — cross-checking its chapter list against this pipeline is what prompted adding TF-IDF text encoding, class-imbalance handling, automatic feature selection, cross-validation, and — in this later pass — target encoding and a broader hyperparameter search, all covered above.

### Known gaps / roadmap

Deliberately out of scope for v0.1 (flagged here rather than silently missing):

- **Feature-vs-feature association tests** — the chi-square/ANOVA/Mann-Whitney suite in section 3 tests each feature against the *target* only; there's no general pairwise categorical-vs-categorical or distribution-comparison testing between arbitrary feature pairs (the correlation heatmap covers numeric-vs-numeric).
- **Full auto-ARIMA order selection** — the search in 7.4 covers `p, q ∈ {0,1,2}` and one fixed seasonal candidate `(1,1,1,7)`; it isn't a full `pmdarima`-style search over differencing orders, seasonal periods, or higher-order terms.
- **Seasonal decomposition** for multi-seasonality data (e.g. both weekly and yearly patterns in the same series) — only one seasonal period is ever tried.
- **Exhaustive hyperparameter tuning** — even the broad search in 6k samples 10 combinations from a bounded distribution per model; it isn't a full Bayesian-optimization sweep over every hyperparameter, and Ridge/Lasso only ever tune within their own internal CV.
- **Per-fold target re-encoding under cross-validation** — noted in 6m: target encoding is derived once from the single train/test split, not re-derived inside each CV fold.

These are natural v0.3 candidates and are listed here rather than glossed over, since knowing what a tool *doesn't* do is as important as knowing what it does.
