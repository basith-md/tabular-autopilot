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


def test_pipeline_runs_standalone_clustering_automatically(regression_df):
    result = run_pipeline(regression_df, target="target", dataset_name="cluster_demo")

    assert result.clustering is not None
    assert result.clustering.k >= 2
    assert result.charts["cluster_scatter"] is not None


def test_pipeline_clustering_runs_even_without_a_target(regression_df):
    result = run_pipeline(regression_df, target=None, dataset_name="cluster_eda_demo")
    assert result.clustering is not None


def test_pipeline_hypothesis_tests_populated_for_classification(classification_df):
    result = run_pipeline(classification_df, target="label", dataset_name="hyp_demo")

    assert result.hypothesis_tests is not None
    assert result.hypothesis_tests.chi_square  # "flag" is categorical vs binary label
    assert result.hypothesis_tests.mann_whitney  # binary label -> Mann-Whitney, not ANOVA
    assert result.hypothesis_tests.anova == []


def test_pipeline_hypothesis_tests_absent_for_regression(regression_df):
    result = run_pipeline(regression_df, target="target", dataset_name="hyp_reg_demo")
    assert result.hypothesis_tests is None


def test_pipeline_high_cardinality_target_encoding_end_to_end():
    rng = np.random.default_rng(10)
    n = 300
    codes = rng.choice([f"C{i}" for i in range(30)], size=n)
    offset = pd.Series(codes).astype("category").cat.codes.to_numpy().astype(float)
    y = offset + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"code": codes, "target": y})

    result = run_pipeline(df, target="target", dataset_name="target_encode_demo", high_cardinality_encoding="target")

    assert result.feature_engineering.deferred_target_encoding == ["code"]
    assert result.modeling.target_encoded_features == ["code"]


def test_pipeline_broad_hyperparameter_search_end_to_end(regression_df):
    result = run_pipeline(
        regression_df,
        target="target",
        dataset_name="broad_search_demo",
        model_names=["Random Forest"],
        hyperparameter_search=True,
        broad_hyperparameter_search=True,
    )

    assert result.modeling.broad_hyperparameter_search_applied
    assert "min_samples_leaf" in result.modeling.best_hyperparameters


def test_pipeline_intervention_analysis_when_date_supplied():
    n = 100
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    rng = np.random.default_rng(11)
    values = np.concatenate(
        [rng.normal(loc=50, scale=1.0, size=50), rng.normal(loc=80, scale=1.0, size=50)]
    )
    df = pd.DataFrame({"order_date": dates.astype(str), "units_sold": values})

    result = run_pipeline(
        df, target="units_sold", dataset_name="intervention_demo", intervention_date=str(dates[50].date())
    )

    assert result.intervention is not None
    assert result.intervention.significant_level_shift
    assert result.charts["intervention"] is not None


def test_pipeline_intervention_absent_without_a_date():
    n = 60
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    rng = np.random.default_rng(12)
    df = pd.DataFrame({"order_date": dates.astype(str), "units_sold": rng.normal(size=n)})

    result = run_pipeline(df, target="units_sold", dataset_name="no_intervention_demo")
    assert result.intervention is None
