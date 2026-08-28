from tabular_autopilot.cleaning import clean_dataframe
from tabular_autopilot.feature_engineering import engineer_features
from tabular_autopilot.modeling import run_modeling
from tabular_autopilot.profiling import profile_dataframe
from tabular_autopilot.schema import infer_schema


def _featured(df, target):
    schema = infer_schema(df, target=target)
    profile = profile_dataframe(df, schema)
    cleaned, _ = clean_dataframe(df, schema, profile)
    featured, fe_report = engineer_features(cleaned, schema)
    return schema, featured, fe_report


def test_regression_produces_ols_baseline_and_metrics(regression_df):
    schema, featured, fe = _featured(regression_df, "target")
    result = run_modeling(featured, "target", fe.final_feature_columns, "regression")

    assert result.task == "regression"
    assert set(result.metrics) == {"MAE", "RMSE", "R2"}
    assert result.metrics["R2"] > 0.5  # data was generated with a strong linear signal
    assert result.baseline.kind == "ols"
    assert result.baseline.pseudo_or_r_squared > 0.5
    assert result.feature_importances
    assert len(result.model_comparison) == 5
    assert result.best_model_name in result.model_comparison


def test_classification_produces_logit_baseline_and_metrics(classification_df):
    schema, featured, fe = _featured(classification_df, "label")
    result = run_modeling(featured, "label", fe.final_feature_columns, "classification")

    assert result.task == "classification"
    assert "Accuracy" in result.metrics
    assert "ROC_AUC" in result.metrics
    assert result.confusion is not None
    assert result.baseline.kind == "logit"
    assert len(result.model_comparison) == 3
    assert result.best_model_name in result.model_comparison
