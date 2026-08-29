import numpy as np
import pandas as pd

from tabular_autopilot.pipeline import run_pipeline


def test_pipeline_regression_end_to_end(regression_df):
    result = run_pipeline(regression_df, target="target", dataset_name="reg_demo")

    assert result.schema.task == "regression"
    assert result.modeling is not None
    assert result.modeling.metrics["R2"] > 0.4
    assert result.charts["correlation_heatmap"] is not None
    assert result.charts["numeric_distributions"] is not None
    assert result.charts["pca_scatter"] is not None


def test_pipeline_classification_end_to_end(classification_df):
    result = run_pipeline(classification_df, target="label", dataset_name="clf_demo")

    assert result.schema.task == "classification"
    assert result.modeling is not None
    assert "Accuracy" in result.modeling.metrics
    assert result.charts["confusion_matrix"] is not None


def test_pipeline_eda_only_when_no_target(mixed_type_df):
    df = mixed_type_df.copy()
    df.loc[0:5, "rooms"] = np.nan
    result = run_pipeline(df, target=None, dataset_name="eda_demo")

    assert result.schema.target is None
    assert result.modeling is None
    assert result.charts["missingness"] is not None
    assert result.charts["numeric_distributions"] is not None


def test_pipeline_geospatial_dataset_produces_geo_chart(mixed_type_df):
    result = run_pipeline(mixed_type_df, target="price", dataset_name="geo_demo")

    assert result.schema.has_geo
    assert result.charts["geo_scatter"] is not None
    assert "geo_cluster" in result.feature_engineering.geo_features_added


def test_pipeline_outlier_capping_reaches_the_cleaning_report(outlier_df):
    result = run_pipeline(outlier_df, target="target", dataset_name="outlier_demo", cap_outliers=True)

    assert result.cleaning.outlier_capping_applied
    assert "x1" in result.cleaning.outlier_capped


def test_pipeline_redundant_pairs_reach_the_profile(redundant_pairs_df):
    result = run_pipeline(redundant_pairs_df, target="target", dataset_name="redundancy_demo")

    pairs = {frozenset((p.col_a, p.col_b)) for p in result.profile.redundant_pairs}
    assert frozenset({"x1", "x1_twin"}) in pairs


def test_pipeline_model_hyperparameters_reach_the_modeling_result(regression_df):
    result = run_pipeline(
        regression_df,
        target="target",
        dataset_name="hp_demo",
        model_names=["Random Forest"],
        rf_n_estimators=42,
        rf_max_depth=6,
    )

    assert result.modeling.best_model_name == "Random Forest"
    assert result.modeling.best_hyperparameters == {"n_estimators": 42, "max_depth": 6}


def test_pipeline_triggers_timeseries_when_datetime_and_numeric_target_present():
    n = 60
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "order_date": dates.astype(str),
            "region": rng.choice(["East", "West"], size=n),
            "units_sold": 50 + 0.3 * np.arange(n) + rng.normal(scale=3.0, size=n),
        }
    )
    result = run_pipeline(df, target="units_sold", dataset_name="retail_demo")

    assert result.timeseries is not None
    assert result.timeseries.n_obs == n
    assert result.charts["ts_trend"] is not None


def test_pipeline_all_numeric_dataset_runs_clean():
    rng = np.random.default_rng(9)
    df = pd.DataFrame(rng.normal(size=(150, 5)), columns=[f"f{i}" for i in range(5)])
    df["target"] = df["f0"] * 2 + df["f1"] - df["f2"] + rng.normal(scale=0.1, size=150)
    result = run_pipeline(df, target="target", dataset_name="numeric_only")
    assert result.modeling.metrics["R2"] > 0.8
