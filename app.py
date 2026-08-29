"""Streamlit UI on top of the tabular_autopilot pipeline.

Run locally with:  streamlit run app.py
Deployable as-is to Streamlit Community Cloud or Hugging Face Spaces —
no code changes needed, just point the platform at this file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from tabular_autopilot.modeling import available_model_names
from tabular_autopilot.pipeline import load_dataframe, run_pipeline
from tabular_autopilot.report import render_html

st.set_page_config(page_title="Tabular Autopilot", layout="wide")

EXAMPLES = {
    "California Housing (regression + geospatial)": (
        "examples/california_housing/data/housing.csv",
        "median_house_value",
    ),
    "Titanic (classification, mixed types)": (
        "examples/titanic_classification/data/titanic.csv",
        "Survived",
    ),
    "Retail Sales (datetime features)": (
        "examples/retail_sales_datetime/data/retail_sales.csv",
        "units_sold",
    ),
}


def _img(uri: str | None):
    if uri:
        st.markdown(f'<img src="{uri}" style="max-width:100%;border-radius:8px;">', unsafe_allow_html=True)
    else:
        st.caption("Not applicable for this dataset.")


def _association_table(tests, digits: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature": t.feature,
                "statistic": round(t.statistic, digits),
                "p_value": round(t.p_value, 4),
                "significant": t.significant,
            }
            for t in tests
        ]
    )


st.title("Tabular Autopilot")
st.caption(
    "Upload any tabular dataset and get automated profiling, cleaning, feature engineering, "
    "modeling and diagnostics — the column roles, cleaning strategy and model choice all adapt "
    "to the data you give it."
)

with st.sidebar:
    st.header("1. Choose data")
    source = st.radio("Source", ["Bundled example", "Upload your own CSV"])

    df = None
    default_target = None
    dataset_name = "dataset"

    if source == "Bundled example":
        choice = st.selectbox("Example dataset", list(EXAMPLES.keys()))
        path, default_target = EXAMPLES[choice]
        full_path = Path(__file__).parent / path
        if full_path.exists():
            df = load_dataframe(full_path)
            dataset_name = full_path.stem
        else:
            st.error(f"Example file missing: {path}")
    else:
        uploaded = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"])
        if uploaded is not None:
            try:
                df = load_dataframe(uploaded, filename=uploaded.name)
                dataset_name = Path(uploaded.name).stem
            except Exception as exc:
                st.error(f"Could not read file: {exc}")

    target = None
    if df is not None:
        st.header("2. Choose target (optional)")
        options = ["(none — EDA only)"] + list(df.columns)
        idx = options.index(default_target) if default_target in options else 0
        target_choice = st.selectbox("Target column", options, index=idx)
        target = None if target_choice.startswith("(none") else target_choice

    st.header("3. Configure the pipeline")
    st.caption(
        "Grouped by the stage of the pipeline each setting affects — everything defaults to "
        "sensible values, so you only need to open a section if you want to change it."
    )

    with st.expander("Data & cleaning", expanded=False):
        test_size_pct = st.slider("Test set size (%)", 10, 40, 20, step=5)
        impute_strategy = st.radio("Numeric missing-value strategy", ["median", "mean"], horizontal=True)
        cap_outliers = st.checkbox(
            "Cap outliers at IQR fences",
            value=False,
            help="Clips extreme numeric values to the IQR fence instead of just counting them. "
            "Off by default since it changes the data.",
        )

    with st.expander("Feature engineering", expanded=False):
        vectorize_text = st.checkbox("TF-IDF vectorize free-text columns (instead of dropping them)", value=True)
        feature_selection = st.checkbox(
            "Automatic feature selection (near-zero-variance + top-50 SelectKBest)", value=True
        )
        high_cardinality_encoding = st.radio(
            "High-cardinality categorical encoding",
            ["frequency", "target"],
            horizontal=True,
            help="Target encoding is out-of-fold (leak-free) and needs a target column; "
            "falls back to frequency encoding for EDA-only runs.",
        )

    with st.expander("Class balance (classification targets)", expanded=False):
        handle_imbalance = st.checkbox("Balance class weighting for imbalanced targets", value=True)

    with st.expander("Models to compare & hyperparameters", expanded=False):
        selected_models = st.multiselect(
            "Models to compare (irrelevant ones for the detected task are ignored)",
            available_model_names(),
            default=available_model_names(),
        )
        hyperparameter_search = st.checkbox(
            "Grid-search each model's key hyperparameter",
            value=False,
            help="Runs a small 3-fold search centered on the values below instead of using them "
            "directly. Disabled together with cross-validation, to avoid nested CV.",
        )
        broad_hyperparameter_search = st.checkbox(
            "Broaden the search (Random Forest / Gradient Boosting / Logistic Regression)",
            value=False,
            disabled=not hyperparameter_search,
            help="Switches to RandomizedSearchCV over several more parameters per model "
            "(e.g. Random Forest also tunes min_samples_leaf and max_features), sampling "
            "10 combinations instead of exhaustively evaluating the small grid above.",
        )
        st.markdown("**Ridge (CV) / Lasso (CV)** — regularization search range")
        rc1, rc2 = st.columns(2)
        ridge_alpha_min = rc1.number_input("Ridge alpha min", value=1e-3, format="%.4f", min_value=1e-6)
        ridge_alpha_max = rc2.number_input("Ridge alpha max", value=1e3, format="%.1f", min_value=ridge_alpha_min)
        lc1, lc2 = st.columns(2)
        lasso_alpha_min = lc1.number_input("Lasso alpha min", value=1e-3, format="%.4f", min_value=1e-6)
        lasso_alpha_max = lc2.number_input("Lasso alpha max", value=1e2, format="%.1f", min_value=lasso_alpha_min)
        st.markdown("**Logistic Regression**")
        logreg_C = st.slider("Inverse regularization strength (C)", 0.01, 10.0, 1.0)
        st.markdown("**Random Forest**")
        rf_n_estimators = st.slider("Number of trees", 50, 500, 200, step=50)
        rf_max_depth_choice = st.select_slider("Max depth", options=["unlimited", 4, 8, 12, 16, 24], value="unlimited")
        rf_max_depth = None if rf_max_depth_choice == "unlimited" else rf_max_depth_choice
        st.markdown("**Gradient Boosting**")
        gb_learning_rate = st.slider("Learning rate", 0.01, 0.5, 0.1, step=0.01)
        gb_max_iter = st.slider("Boosting rounds", 20, 300, 100, step=10)

    with st.expander("Evaluation strategy", expanded=False):
        use_cv = st.checkbox("Use k-fold cross-validation instead of a single split (slower)", value=False)
        cv_folds = st.slider("CV folds", 2, 10, 5, disabled=not use_cv) if use_cv else 0

    with st.expander("Time series (optional)", expanded=False):
        st.caption(
            "Only used when the dataset has a datetime column and a numeric target. "
            "Tests whether the series shifted level or trend right at this date "
            "(e.g. a policy change or product launch)."
        )
        intervention_date_input = st.text_input("Intervention date (YYYY-MM-DD)", value="")
        intervention_date = intervention_date_input.strip() or None

    run_clicked = st.button("Analyze", type="primary", disabled=df is None)

if df is None:
    st.info("Choose a bundled example or upload a CSV from the sidebar to get started.")
    st.stop()

st.subheader(f"Preview — {dataset_name}")
st.dataframe(df.head(10), width="stretch")

if not run_clicked:
    st.stop()

try:
    with st.spinner("Running automated pipeline..."):
        result = run_pipeline(
            df,
            target=target,
            dataset_name=dataset_name,
            numeric_impute_strategy=impute_strategy,
            test_size=test_size_pct / 100,
            model_names=selected_models or None,
            vectorize_text=vectorize_text,
            handle_imbalance=handle_imbalance,
            feature_selection=feature_selection,
            cv_folds=cv_folds,
            cap_outliers=cap_outliers,
            ridge_alpha_range=(ridge_alpha_min, ridge_alpha_max),
            lasso_alpha_range=(lasso_alpha_min, lasso_alpha_max),
            logreg_C=logreg_C,
            rf_n_estimators=rf_n_estimators,
            rf_max_depth=rf_max_depth,
            gb_learning_rate=gb_learning_rate,
            gb_max_iter=gb_max_iter,
            hyperparameter_search=hyperparameter_search,
            broad_hyperparameter_search=broad_hyperparameter_search,
            high_cardinality_encoding=high_cardinality_encoding,
            intervention_date=intervention_date,
        )
except Exception as exc:
    st.error(f"Analysis failed: {exc}")
    st.stop()

schema, profile = result.schema, result.profile
st.success(f"Done — {profile.n_rows} rows x {profile.n_cols} columns.")
st.download_button(
    "Download full HTML report",
    data=render_html(result),
    file_name=f"{dataset_name}_report.html",
    mime="text/html",
)

tabs = st.tabs(
    [
        "Overview",
        "Missing data",
        "Distributions",
        "Segments",
        "Correlations",
        "Geospatial",
        "Feature engineering",
        "Model results",
        "Diagnostics",
        "Time series",
    ]
)

with tabs[0]:
    st.subheader("Column roles")
    rows = [
        {
            "column": name,
            "role": prof.role.value,
            "dtype": prof.dtype,
            "% missing": round(prof.pct_missing * 100, 1),
            "unique": prof.n_unique,
        }
        for name, prof in schema.columns.items()
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch")
    if profile.duplicate_rows:
        st.warning(f"{profile.duplicate_rows} duplicate row(s) detected.")

with tabs[1]:
    st.subheader("Missing values")
    _img(result.charts.get("missingness"))
    if result.cleaning.imputed_numeric:
        strategy = result.cleaning.numeric_impute_strategy
        st.write(f"Numeric columns imputed ({strategy}):", result.cleaning.imputed_numeric)
    if result.cleaning.imputed_categorical:
        st.write("Categorical columns imputed (mode):", result.cleaning.imputed_categorical)

with tabs[2]:
    st.subheader("Numeric distributions")
    _img(result.charts.get("numeric_distributions"))
    if result.cleaning.log_transformed:
        st.write("Log-transformed (right-skewed):", result.cleaning.log_transformed)
    if result.cleaning.outlier_capping_applied:
        st.write("Outliers capped at IQR fences:", list(result.cleaning.outlier_capped.keys()) or "none needed capping")
    if profile.numeric_profiles:
        st.dataframe(
            pd.DataFrame(
                {
                    "mean": {k: v.mean for k, v in profile.numeric_profiles.items()},
                    "std": {k: v.std for k, v in profile.numeric_profiles.items()},
                    "skew": {k: v.skew for k, v in profile.numeric_profiles.items()},
                    "outliers": {k: v.n_outliers for k, v in profile.numeric_profiles.items()},
                }
            ),
            width="stretch",
        )

with tabs[3]:
    st.subheader("Segments (unsupervised clustering)")
    st.caption(
        "KMeans over every numeric column, cluster count chosen automatically by silhouette "
        "score — independent of geography or any modeling target."
    )
    if result.clustering:
        c1, c2 = st.columns(2)
        c1.metric("Segments found", result.clustering.k)
        c2.metric("Silhouette score", f"{result.clustering.silhouette_score:.3f}")
        st.write("Segment sizes:", result.clustering.cluster_sizes)
        st.caption(f"Based on: {', '.join(result.clustering.columns_used)}")
        _img(result.charts.get("cluster_scatter"))
    else:
        st.caption("Not enough numeric columns or rows to cluster meaningfully.")

with tabs[4]:
    st.subheader("Correlation heatmap")
    _img(result.charts.get("correlation_heatmap"))
    st.subheader("Target by category")
    _img(result.charts.get("target_by_category"))
    if profile.redundant_pairs:
        st.subheader("Redundant feature pairs (|r| ≥ 0.9)")
        st.caption("These columns move almost interchangeably — worth dropping one regardless of the winning model.")
        pairs_rows = [
            {"Column A": p.col_a, "Column B": p.col_b, "Correlation": round(p.correlation, 3)}
            for p in profile.redundant_pairs
        ]
        st.dataframe(pd.DataFrame(pairs_rows), width="stretch")
    if result.hypothesis_tests:
        st.subheader(f"Formal association tests (target: `{schema.target}`)")
        ht = result.hypothesis_tests
        if ht.chi_square:
            st.write("Chi-square test of independence — each categorical column vs the target:")
            st.dataframe(_association_table(ht.chi_square), width="stretch")
        if ht.anova:
            st.write("One-way ANOVA (>2 classes) — each numeric column vs the target:")
            st.dataframe(_association_table(ht.anova), width="stretch")
        if ht.mann_whitney:
            st.write("Mann-Whitney U (binary target, non-parametric) — each numeric column vs the target:")
            st.dataframe(_association_table(ht.mann_whitney, digits=1), width="stretch")

with tabs[5]:
    st.subheader("Geospatial distribution")
    if schema.has_geo:
        _img(result.charts.get("geo_scatter"))
        st.write("Engineered geo features:", result.feature_engineering.geo_features_added)
    else:
        st.caption("No latitude/longitude column pair detected in this dataset.")

with tabs[6]:
    st.subheader("Feature engineering summary")
    st.write("One-hot encoded:", result.feature_engineering.one_hot_encoded or "none")
    st.write("Frequency encoded:", result.feature_engineering.frequency_encoded or "none")
    if result.feature_engineering.deferred_target_encoding:
        st.write(
            "Target-encoded (out-of-fold, applied after the train/test split):",
            result.feature_engineering.deferred_target_encoding,
        )
    st.write("Datetime expanded:", list(result.feature_engineering.datetime_expanded.keys()) or "none")
    st.write("Text columns TF-IDF vectorized:", list(result.feature_engineering.text_vectorized.keys()) or "none")
    st.write("Dropped (identifier/constant/text):", result.feature_engineering.dropped_columns or "none")
    if result.charts.get("pca_scatter"):
        st.subheader("Feature separability (PCA projection)")
        st.caption(
            "A 2D projection of every engineered feature, colored by target/class — a quick "
            "visual for whether the data separates at all before modeling."
        )
        _img(result.charts.get("pca_scatter"))

with tabs[7]:
    if result.modeling is None:
        st.caption("No target column selected — modeling was skipped (EDA-only run).")
    else:
        m = result.modeling
        st.subheader(f"{m.task.title()} model on `{m.target}`")
        split_desc = f"{m.cv_folds}-fold cross-validation" if m.cv_folds >= 2 else "an identical train/test split"
        st.caption(f"{len(m.model_comparison)} candidate models compared using {split_desc}.")
        if m.feature_selection_applied:
            st.caption(
                f"Feature selection: {m.n_features_before_selection} engineered features → "
                f"{m.n_features_after_selection} used for modeling."
            )
        if m.task == "classification" and m.is_imbalanced:
            note = " (balanced class weighting applied)" if m.class_weight_applied else " (weighting disabled)"
            st.warning(f"Target is imbalanced ({profile.class_balance.imbalance_ratio:.1f}:1){note}.")
        comp_df = pd.DataFrame(m.model_comparison).T
        st.dataframe(comp_df, width="stretch")
        if m.best_vs_runner_up_test:
            wt = m.best_vs_runner_up_test
            verdict = (
                "statistically significant, not just noise between folds"
                if wt.significant
                else "not statistically significant — within noise of each other"
            )
            st.caption(
                f"Wilcoxon signed-rank, {m.best_model_name} vs runner-up `{m.runner_up_model_name}` "
                f"across {m.cv_folds} CV folds: p={wt.p_value:.4f} — the difference is {verdict}."
            )
        st.markdown(f"**Best model:** `{m.best_model_name}`")
        if m.best_hyperparameters:
            if m.broad_hyperparameter_search_applied:
                search_note = " (found via broad randomized search)"
            elif m.hyperparameter_search_applied:
                search_note = " (found via grid search)"
            else:
                search_note = ""
            params = ", ".join(f"{k}={v}" for k, v in m.best_hyperparameters.items())
            st.caption(f"Hyperparameters{search_note}: {params}")
        cols = st.columns(len(m.metrics))
        for c, (name, val) in zip(cols, m.metrics.items()):
            c.metric(name, f"{val:.4f}")
        _img(result.charts.get("feature_importance"))
        _img(result.charts.get("confusion_matrix"))
        if result.charts.get("decision_tree"):
            st.markdown(f"**How a tree-based model like `{m.best_model_name}` decides**")
            st.caption(
                "This ensemble averages many trees together, which can't be drawn directly. "
                "This is one shallow tree (depth 3), fit fresh on the same data, showing the "
                "kind of split logic the ensemble is built from."
            )
            _img(result.charts.get("decision_tree"))

with tabs[8]:
    if result.modeling is None or result.modeling.baseline is None:
        st.caption("No diagnostics available for this run.")
    elif result.modeling.baseline.kind == "skipped":
        st.caption(result.modeling.baseline.note)
    else:
        b = result.modeling.baseline
        label = "R²" if b.kind == "ols" else "Pseudo R²"
        cols = st.columns(4)
        cols[0].metric(label, f"{b.pseudo_or_r_squared:.3f}")
        if b.adj_r_squared is not None:
            cols[1].metric("Adj. R²", f"{b.adj_r_squared:.3f}")
        if b.breusch_pagan_pvalue is not None:
            cols[2].metric("Breusch-Pagan p", f"{b.breusch_pagan_pvalue:.4f}")
            if b.heteroscedastic:
                st.warning("Heteroscedasticity detected (p < 0.05) — baseline standard errors may be unreliable.")
            else:
                st.success("No significant heteroscedasticity detected.")
        if b.normality_test:
            cols[3].metric("Shapiro-Wilk p", f"{b.normality_test.p_value:.4f}")
            if b.normality_test.significant:
                st.warning("Shapiro-Wilk rejects normality of the residuals (p < 0.05) — check the Q-Q plot below.")
            else:
                st.success("Shapiro-Wilk does not reject normality of the residuals.")
        if b.dropped_for_multicollinearity:
            st.warning(f"Dropped for multicollinearity (VIF > 10): {b.dropped_for_multicollinearity}")
        _img(result.charts.get("residuals_vs_fitted"))
        _img(result.charts.get("qq_plot"))
        with st.expander("Full statistical summary"):
            st.code(b.summary_text)

with tabs[9]:
    ts = result.timeseries
    if ts is None:
        st.caption("No datetime column + numeric target combination detected — time series diagnostics skipped.")
    else:
        st.subheader(f"{ts.value_col} over {ts.date_col}")
        cols = st.columns(3)
        cols[0].metric("Observations", ts.n_obs)
        cols[1].metric("Trend R²", f"{ts.trend_r_squared:.3f}")
        cols[2].metric("ADF p-value", f"{ts.adf_pvalue:.4f}")
        if ts.is_stationary:
            st.success("Augmented Dickey-Fuller test rejects a unit root — series appears stationary.")
        else:
            st.warning("Augmented Dickey-Fuller test cannot reject a unit root — series appears non-stationary.")
        if ts.forecast_method in ("arima", "sarima"):
            seasonal_note = f" × seasonal {ts.seasonal_order}" if ts.seasonal_order else ""
            st.caption(
                f"Forecast method: **{ts.forecast_method.upper()}** order {ts.arima_order}{seasonal_note}, "
                f"chosen by lowest AIC ({ts.aic:.1f}) over a small (p,d,q) grid."
            )
        elif ts.forecast_method == "exponential_smoothing":
            st.caption("Forecast method: **Exponential Smoothing** (ARIMA/SARIMA order search did not converge).")
        _img(result.charts.get("ts_trend"))
        _img(result.charts.get("ts_acf_pacf"))
        if ts.note:
            st.caption(ts.note)

        if result.intervention:
            iv = result.intervention
            st.subheader(f"Interrupted time series — intervention on {iv.intervention_date}")
            st.caption(
                f"Segmented regression fit on {iv.n_pre} pre-intervention and {iv.n_post} "
                "post-intervention observations."
            )
            cols = st.columns(4)
            cols[0].metric("Level shift", f"{iv.level_shift:.3f}")
            cols[1].metric("Level shift p", f"{iv.level_shift_pvalue:.4f}")
            cols[2].metric("Slope change", f"{iv.slope_change:.4f}")
            cols[3].metric("Slope change p", f"{iv.slope_change_pvalue:.4f}")
            if iv.significant_level_shift:
                st.success("The level of the series shifted significantly at the intervention date.")
            else:
                st.caption("No statistically significant level shift detected at the intervention date.")
            if iv.significant_slope_change:
                st.success("The trend's slope also changed significantly after the intervention.")
            else:
                st.caption("No statistically significant slope change detected after the intervention.")
            _img(result.charts.get("intervention"))
