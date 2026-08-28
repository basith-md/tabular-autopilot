"""Streamlit UI on top of the tabular_autopilot pipeline.

Run locally with:  streamlit run app.py
Deployable as-is to Streamlit Community Cloud or Hugging Face Spaces —
no code changes needed, just point the platform at this file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from tabular_autopilot.pipeline import load_dataframe, run_pipeline
from tabular_autopilot.report import render_html

st.set_page_config(page_title="tabular-autopilot", layout="wide")

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


st.title("tabular-autopilot")
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
        result = run_pipeline(df, target=target, dataset_name=dataset_name)
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
        st.write("Numeric columns imputed (median):", result.cleaning.imputed_numeric)
    if result.cleaning.imputed_categorical:
        st.write("Categorical columns imputed (mode):", result.cleaning.imputed_categorical)

with tabs[2]:
    st.subheader("Numeric distributions")
    _img(result.charts.get("numeric_distributions"))
    if result.cleaning.log_transformed:
        st.write("Log-transformed (right-skewed):", result.cleaning.log_transformed)
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
    st.subheader("Correlation heatmap")
    _img(result.charts.get("correlation_heatmap"))
    st.subheader("Target by category")
    _img(result.charts.get("target_by_category"))

with tabs[4]:
    st.subheader("Geospatial distribution")
    if schema.has_geo:
        _img(result.charts.get("geo_scatter"))
        st.write("Engineered geo features:", result.feature_engineering.geo_features_added)
    else:
        st.caption("No latitude/longitude column pair detected in this dataset.")

with tabs[5]:
    st.subheader("Feature engineering summary")
    st.write("One-hot encoded:", result.feature_engineering.one_hot_encoded or "none")
    st.write("Frequency encoded:", result.feature_engineering.frequency_encoded or "none")
    st.write("Datetime expanded:", list(result.feature_engineering.datetime_expanded.keys()) or "none")
    st.write("Dropped (identifier/constant/text):", result.feature_engineering.dropped_columns or "none")

with tabs[6]:
    if result.modeling is None:
        st.caption("No target column selected — modeling was skipped (EDA-only run).")
    else:
        m = result.modeling
        st.subheader(f"{m.task.title()} model on `{m.target}`")
        st.caption(f"{len(m.model_comparison)} candidate models compared on an identical train/test split.")
        comp_df = pd.DataFrame(m.model_comparison).T
        st.dataframe(comp_df, width="stretch")
        st.markdown(f"**Best model:** `{m.best_model_name}`")
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

with tabs[7]:
    if result.modeling is None or result.modeling.baseline is None:
        st.caption("No diagnostics available for this run.")
    elif result.modeling.baseline.kind == "skipped":
        st.caption(result.modeling.baseline.note)
    else:
        b = result.modeling.baseline
        label = "R²" if b.kind == "ols" else "Pseudo R²"
        cols = st.columns(3)
        cols[0].metric(label, f"{b.pseudo_or_r_squared:.3f}")
        if b.adj_r_squared is not None:
            cols[1].metric("Adj. R²", f"{b.adj_r_squared:.3f}")
        if b.breusch_pagan_pvalue is not None:
            cols[2].metric("Breusch-Pagan p", f"{b.breusch_pagan_pvalue:.4f}")
            if b.heteroscedastic:
                st.warning("Heteroscedasticity detected (p < 0.05) — baseline standard errors may be unreliable.")
            else:
                st.success("No significant heteroscedasticity detected.")
        if b.dropped_for_multicollinearity:
            st.warning(f"Dropped for multicollinearity (VIF > 10): {b.dropped_for_multicollinearity}")
        _img(result.charts.get("residuals_vs_fitted"))
        _img(result.charts.get("qq_plot"))
        with st.expander("Full statistical summary"):
            st.code(b.summary_text)

with tabs[8]:
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
        _img(result.charts.get("ts_trend"))
        _img(result.charts.get("ts_acf_pacf"))
        if ts.note:
            st.caption(ts.note)
