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
from tabular_autopilot.clustering import ClusteringResult, run_clustering
from tabular_autopilot.feature_engineering import FeatureEngineeringReport, engineer_features
from tabular_autopilot.hypothesis_tests import (
    HypothesisTestSuite,
    anova_tests,
    chi_square_tests,
    mann_whitney_tests,
)
from tabular_autopilot.modeling import ModelingResult, run_modeling
from tabular_autopilot.profiling import ProfileReport, profile_dataframe
from tabular_autopilot.schema import SchemaResult, infer_schema
from tabular_autopilot.timeseries import (
    InterventionResult,
    TimeSeriesResult,
    analyze_intervention,
    analyze_time_series,
)


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
    clustering: ClusteringResult | None = None
    hypothesis_tests: HypothesisTestSuite | None = None
    intervention: InterventionResult | None = None


# Tried in order against CSVs of unknown provenance: real UTF-8 first (most
# common and the only one that round-trips non-Latin text correctly), then
# utf-8-sig (Excel's "CSV UTF-8" export adds a BOM), then cp1252 (Excel's
# *default* Windows export encoding -- "smart quotes" and en-dashes live in
# the 0x80-0x9F range that plain UTF-8 rejects outright), then latin-1 as a
# final catch-all that maps every byte to a codepoint and therefore never
# raises -- guaranteeing this loop terminates.
_CSV_ENCODING_FALLBACKS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


def _read_csv_robust(path_or_buffer) -> pd.DataFrame:
    for encoding in _CSV_ENCODING_FALLBACKS:
        try:
            return pd.read_csv(path_or_buffer, encoding=encoding)
        except UnicodeDecodeError:
            if hasattr(path_or_buffer, "seek"):
                path_or_buffer.seek(0)
            continue
    return pd.read_csv(path_or_buffer)  # pragma: no cover -- latin-1 above always succeeds


def load_dataframe(path_or_buffer, filename: str | None = None) -> pd.DataFrame:
    """Load a CSV/Excel/Parquet file from a path, or a file-like object plus
    its original ``filename`` (needed to tell the formats apart when the
    object itself -- e.g. a Streamlit/browser upload -- has no extension)."""
    name = filename if filename is not None else str(path_or_buffer)
    suffix = Path(name).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path_or_buffer)
    if suffix in (".xls", ".xlsx"):
        return pd.read_excel(path_or_buffer)
    return _read_csv_robust(path_or_buffer)


def _build_charts(
    cleaned_df: pd.DataFrame,
    featured_df: pd.DataFrame,
    fe_report: FeatureEngineeringReport,
    schema: SchemaResult,
    profile: ProfileReport,
    modeling: ModelingResult | None,
    clustering: ClusteringResult | None,
) -> dict[str, str | None]:
    charts: dict[str, str | None] = {}
    charts["missingness"] = viz.plot_missingness(cleaned_df, profile.missing_by_col)
    charts["numeric_distributions"] = viz.plot_numeric_distributions(cleaned_df, schema.numeric_cols)

    if clustering:
        charts["cluster_scatter"] = viz.plot_cluster_scatter(cleaned_df, clustering.columns_used, clustering.labels)
    else:
        charts["cluster_scatter"] = None

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
        geo_df = cleaned_df.copy()
        if "geo_cluster" in featured_df.columns:
            geo_df["geo_cluster"] = featured_df["geo_cluster"].to_numpy()
        charts["geo_scatter"] = viz.plot_geo_scatter(
            geo_df,
            schema.geo_lat_col,
            schema.geo_lon_col,
            schema.target,
            task=schema.task,
            cluster_col="geo_cluster" if "geo_cluster" in geo_df.columns else None,
        )
    else:
        charts["geo_scatter"] = None

    if modeling and schema.target and fe_report.final_feature_columns:
        charts["pca_scatter"] = viz.plot_pca_scatter(
            featured_df[fe_report.final_feature_columns], featured_df[schema.target], schema.task
        )
    else:
        charts["pca_scatter"] = None

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
        if modeling.is_tree_based and modeling.illustrative_tree is not None:
            charts["decision_tree"] = viz.plot_decision_tree(
                modeling.illustrative_tree,
                modeling.illustrative_tree_features,
                modeling.class_labels if modeling.task == "classification" else None,
            )
        else:
            charts["decision_tree"] = None
    else:
        charts["feature_importance"] = None
        charts["confusion_matrix"] = None
        charts["decision_tree"] = None

    return charts


def _build_timeseries_charts(
    ts: TimeSeriesResult | None, intervention: InterventionResult | None
) -> dict[str, str | None]:
    if ts is None:
        charts = {"ts_trend": None, "ts_acf_pacf": None}
    else:
        charts = {
            "ts_trend": viz.plot_time_series_trend(
                ts.history_dates, ts.history_values, ts.forecast_index, ts.forecast
            ),
            "ts_acf_pacf": viz.plot_acf_pacf(ts.acf_values, ts.pacf_values),
        }
    if intervention:
        charts["intervention"] = viz.plot_intervention(
            intervention.history_dates, intervention.history_values, intervention.fitted, intervention.intervention_date
        )
    else:
        charts["intervention"] = None
    return charts


def run_pipeline(
    df: pd.DataFrame,
    target: str | None = None,
    dataset_name: str = "dataset",
    numeric_impute_strategy: str = "median",
    test_size: float = 0.2,
    model_names: list[str] | None = None,
    vectorize_text: bool = True,
    handle_imbalance: bool = True,
    feature_selection: bool = True,
    cv_folds: int = 0,
    cap_outliers: bool = False,
    ridge_alpha_range: tuple[float, float] = (1e-3, 1e3),
    lasso_alpha_range: tuple[float, float] = (1e-3, 1e2),
    logreg_C: float = 1.0,
    rf_n_estimators: int = 200,
    rf_max_depth: int | None = None,
    gb_learning_rate: float = 0.1,
    gb_max_iter: int = 100,
    hyperparameter_search: bool = False,
    broad_hyperparameter_search: bool = False,
    high_cardinality_encoding: str = "frequency",
    intervention_date: str | None = None,
) -> AnalysisResult:
    schema = infer_schema(df, target=target)
    profile = profile_dataframe(df, schema)
    cleaned_df, cleaning_report = clean_dataframe(
        df, schema, profile, numeric_impute_strategy, cap_outliers=cap_outliers
    )
    featured_df, fe_report = engineer_features(
        cleaned_df, schema, vectorize_text=vectorize_text, high_cardinality_encoding=high_cardinality_encoding
    )

    clustering_result = run_clustering(cleaned_df, schema.numeric_cols)

    hypothesis_result = None
    if schema.target and schema.task == "classification":
        categorical_cols = schema.categorical_low_cols + schema.categorical_high_cols
        hypothesis_result = HypothesisTestSuite(
            chi_square=chi_square_tests(cleaned_df, categorical_cols, schema.target),
            anova=anova_tests(cleaned_df, schema.numeric_cols, schema.target),
            mann_whitney=mann_whitney_tests(cleaned_df, schema.numeric_cols, schema.target),
        )

    modeling_result = None
    if schema.target and schema.task:
        modeling_result = run_modeling(
            featured_df,
            schema.target,
            fe_report.final_feature_columns,
            schema.task,
            test_size=test_size,
            model_names=model_names,
            handle_imbalance=handle_imbalance,
            feature_selection=feature_selection,
            cv_folds=cv_folds,
            ridge_alpha_range=ridge_alpha_range,
            lasso_alpha_range=lasso_alpha_range,
            logreg_C=logreg_C,
            rf_n_estimators=rf_n_estimators,
            rf_max_depth=rf_max_depth,
            gb_learning_rate=gb_learning_rate,
            gb_max_iter=gb_max_iter,
            hyperparameter_search=hyperparameter_search,
            broad_hyperparameter_search=broad_hyperparameter_search,
            target_encode_cols=fe_report.deferred_target_encoding,
        )

    charts = _build_charts(cleaned_df, featured_df, fe_report, schema, profile, modeling_result, clustering_result)

    ts_result = None
    intervention_result = None
    if schema.target and schema.task == "regression" and schema.datetime_cols:
        ts_result = analyze_time_series(cleaned_df, schema.datetime_cols[0], schema.target)
        if intervention_date:
            intervention_result = analyze_intervention(
                cleaned_df, schema.datetime_cols[0], schema.target, intervention_date
            )
    charts.update(_build_timeseries_charts(ts_result, intervention_result))

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
        clustering=clustering_result,
        hypothesis_tests=hypothesis_result,
        intervention=intervention_result,
    )


def run_pipeline_from_csv(
    path: str | Path, target: str | None = None, dataset_name: str | None = None
) -> AnalysisResult:
    path = Path(path)
    df = load_dataframe(path)
    return run_pipeline(df, target=target, dataset_name=dataset_name or path.stem)
