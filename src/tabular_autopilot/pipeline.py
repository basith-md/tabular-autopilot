"""Orchestrator: wires schema inference -> profiling -> cleaning ->
feature engineering -> modeling -> charts into a single call, and renders
the result to an HTML report. This is the one function both the CLI and the
Streamlit app call — neither interface contains analysis logic of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tabular_autopilot import eda_visuals as viz
from tabular_autopilot.cleaning import CleaningReport, clean_dataframe
from tabular_autopilot.feature_engineering import FeatureEngineeringReport, engineer_features
from tabular_autopilot.modeling import ModelingResult, run_modeling
from tabular_autopilot.profiling import ProfileReport, profile_dataframe
from tabular_autopilot.schema import SchemaResult, infer_schema
from tabular_autopilot.timeseries import TimeSeriesResult, analyze_time_series


@dataclass
class AnalysisResult:
    dataset_name: str
    schema: SchemaResult
    profile: ProfileReport
    cleaning: CleaningReport
    feature_engineering: FeatureEngineeringReport
    modeling: ModelingResult | None
    charts: dict[str, str | None]
    cleaned_df: pd.DataFrame
    timeseries: TimeSeriesResult | None = None


def load_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".parquet",):
        return pd.read_parquet(path)
    if path.suffix.lower() in (".xls", ".xlsx"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def _build_charts(
    cleaned_df: pd.DataFrame, schema: SchemaResult, profile: ProfileReport, modeling: ModelingResult | None
) -> dict[str, str | None]:
    charts: dict[str, str | None] = {}
    charts["missingness"] = viz.plot_missingness(cleaned_df, profile.missing_by_col)
    charts["numeric_distributions"] = viz.plot_numeric_distributions(cleaned_df, schema.numeric_cols)

    corr_cols = list(schema.numeric_cols)
    if schema.target and schema.task == "regression":
        corr_cols = corr_cols + [schema.target]
    charts["correlation_heatmap"] = viz.plot_correlation_heatmap(cleaned_df, corr_cols)

    if schema.target and schema.task == "regression" and schema.categorical_low_cols:
        charts["target_by_category"] = viz.plot_target_by_category(
            cleaned_df, schema.target, schema.categorical_low_cols[0]
        )
    else:
        charts["target_by_category"] = None

    if schema.has_geo:
        color_col = schema.target if schema.task == "regression" else None
        charts["geo_scatter"] = viz.plot_geo_scatter(
            cleaned_df, schema.geo_lat_col, schema.geo_lon_col, color_col
        )
    else:
        charts["geo_scatter"] = None

    if modeling and modeling.baseline and modeling.baseline.kind == "ols":
        charts["residuals_vs_fitted"] = viz.plot_residuals_vs_fitted(
            modeling.baseline.fitted, modeling.baseline.residuals
        )
        charts["qq_plot"] = viz.plot_qq(modeling.baseline.residuals)
    else:
        charts["residuals_vs_fitted"] = None
        charts["qq_plot"] = None

    if modeling:
        charts["feature_importance"] = viz.plot_feature_importance(modeling.feature_importances)
        if modeling.confusion is not None:
            charts["confusion_matrix"] = viz.plot_confusion_matrix(
                modeling.confusion, modeling.class_labels
            )
        else:
            charts["confusion_matrix"] = None
    else:
        charts["feature_importance"] = None
        charts["confusion_matrix"] = None

    return charts


def _build_timeseries_charts(ts: TimeSeriesResult | None) -> dict[str, str | None]:
    if ts is None:
        return {"ts_trend": None, "ts_acf_pacf": None}
    return {
        "ts_trend": viz.plot_time_series_trend(
            ts.history_dates, ts.history_values, ts.forecast_index, ts.forecast
        ),
        "ts_acf_pacf": viz.plot_acf_pacf(ts.acf_values, ts.pacf_values),
    }


def run_pipeline(
    df: pd.DataFrame, target: str | None = None, dataset_name: str = "dataset"
) -> AnalysisResult:
    schema = infer_schema(df, target=target)
    profile = profile_dataframe(df, schema)
    cleaned_df, cleaning_report = clean_dataframe(df, schema, profile)
    featured_df, fe_report = engineer_features(cleaned_df, schema)

    modeling_result = None
    if schema.target and schema.task:
        modeling_result = run_modeling(
            featured_df, schema.target, fe_report.final_feature_columns, schema.task
        )

    charts = _build_charts(cleaned_df, schema, profile, modeling_result)

    ts_result = None
    if schema.target and schema.task == "regression" and schema.datetime_cols:
        ts_result = analyze_time_series(cleaned_df, schema.datetime_cols[0], schema.target)
    charts.update(_build_timeseries_charts(ts_result))

    return AnalysisResult(
        dataset_name=dataset_name,
        schema=schema,
        profile=profile,
        cleaning=cleaning_report,
        feature_engineering=fe_report,
        modeling=modeling_result,
        charts=charts,
        cleaned_df=cleaned_df,
        timeseries=ts_result,
    )


def run_pipeline_from_csv(
    path: str | Path, target: str | None = None, dataset_name: str | None = None
) -> AnalysisResult:
    path = Path(path)
    df = load_dataframe(path)
    return run_pipeline(df, target=target, dataset_name=dataset_name or path.stem)
