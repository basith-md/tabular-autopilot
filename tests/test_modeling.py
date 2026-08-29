import numpy as np
import pandas as pd

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
    assert "Balanced_Accuracy" in result.metrics
    assert result.confusion is not None
    assert result.baseline.kind == "logit"
    assert len(result.model_comparison) == 3
    assert result.best_model_name in result.model_comparison


def test_tree_based_winner_gets_an_illustrative_tree():
    # A classic XOR-style interaction: no linear boundary separates the
    # classes, so a tree-based model should clearly outperform logistic
    # regression and win the comparison, triggering the illustrative tree.
    rng = np.random.default_rng(5)
    n = 400
    x1 = rng.uniform(-1, 1, size=n)
    x2 = rng.uniform(-1, 1, size=n)
    label = ((x1 > 0) != (x2 > 0)).astype(int)
    df = pd.DataFrame({"x1": x1, "x2": x2, "label": label})

    schema = infer_schema(df, target="label")
    profile = profile_dataframe(df, schema)
    cleaned, _ = clean_dataframe(df, schema, profile)
    featured, fe = engineer_features(cleaned, schema)
    result = run_modeling(featured, "label", fe.final_feature_columns, "classification")

    assert result.best_model_name in ("Random Forest", "Gradient Boosting")
    assert result.is_tree_based
    assert result.illustrative_tree is not None
    assert result.illustrative_tree_features == ["x1", "x2"]
    assert result.illustrative_tree.get_depth() <= 3


def test_imbalance_is_detected_and_class_weight_applied(imbalanced_classification_df):
    schema, featured, fe = _featured(imbalanced_classification_df, "label")
    result = run_modeling(featured, "label", fe.final_feature_columns, "classification", handle_imbalance=True)

    assert result.is_imbalanced
    assert result.class_weight_applied


def test_imbalance_handling_can_be_disabled(imbalanced_classification_df):
    schema, featured, fe = _featured(imbalanced_classification_df, "label")
    result = run_modeling(featured, "label", fe.final_feature_columns, "classification", handle_imbalance=False)

    assert result.is_imbalanced  # still detected/reported
    assert not result.class_weight_applied  # but not acted on


def test_feature_selection_prunes_a_wide_feature_set(wide_regression_df):
    schema, featured, fe = _featured(wide_regression_df, "target")
    assert len(fe.final_feature_columns) > 50  # 2 signal + 80 noise columns

    result = run_modeling(featured, "target", fe.final_feature_columns, "regression", feature_selection=True)

    assert result.feature_selection_applied
    assert result.n_features_before_selection > 50
    assert result.n_features_after_selection <= 50
    assert result.metrics["R2"] > 0.5  # signal survives selection


def test_feature_selection_can_be_disabled(wide_regression_df):
    schema, featured, fe = _featured(wide_regression_df, "target")

    result = run_modeling(featured, "target", fe.final_feature_columns, "regression", feature_selection=False)

    assert not result.feature_selection_applied
    assert result.n_features_after_selection == result.n_features_before_selection


def test_cross_validation_mode_reports_metrics_and_still_fits_a_model(regression_df):
    schema, featured, fe = _featured(regression_df, "target")
    result = run_modeling(featured, "target", fe.final_feature_columns, "regression", cv_folds=5)

    assert result.cv_folds == 5
    assert result.metrics["R2"] > 0.5
    assert result.feature_importances  # requires a concretely fitted best_model
    assert result.baseline.kind == "ols"  # baseline is unaffected by cv_folds


def test_cross_validation_disabled_by_default(regression_df):
    schema, featured, fe = _featured(regression_df, "target")
    result = run_modeling(featured, "target", fe.final_feature_columns, "regression")

    assert result.cv_folds == 0


def test_ridge_alpha_range_bounds_the_chosen_alpha(regression_df):
    schema, featured, fe = _featured(regression_df, "target")
    result = run_modeling(
        featured,
        "target",
        fe.final_feature_columns,
        "regression",
        model_names=["Ridge (CV)"],
        ridge_alpha_range=(1.0, 10.0),
    )

    assert result.best_model_name == "Ridge (CV)"
    assert 1.0 <= result.best_hyperparameters["alpha"] <= 10.0


def test_random_forest_hyperparameters_are_applied_and_reported(regression_df):
    schema, featured, fe = _featured(regression_df, "target")
    result = run_modeling(
        featured,
        "target",
        fe.final_feature_columns,
        "regression",
        model_names=["Random Forest"],
        rf_n_estimators=37,
        rf_max_depth=4,
    )

    assert result.best_model_name == "Random Forest"
    assert result.best_hyperparameters == {"n_estimators": 37, "max_depth": 4}
    assert not result.hyperparameter_search_applied


def test_hyperparameter_search_finds_a_value_from_the_grid(regression_df):
    schema, featured, fe = _featured(regression_df, "target")
    result = run_modeling(
        featured,
        "target",
        fe.final_feature_columns,
        "regression",
        model_names=["Random Forest"],
        rf_n_estimators=100,
        rf_max_depth=None,
        hyperparameter_search=True,
    )

    assert result.hyperparameter_search_applied
    assert result.best_hyperparameters["n_estimators"] in (50, 100, 200)
    assert result.best_hyperparameters["max_depth"] in (None, 8, 16)


def test_hyperparameter_search_is_skipped_under_cross_validation(regression_df):
    schema, featured, fe = _featured(regression_df, "target")
    result = run_modeling(
        featured,
        "target",
        fe.final_feature_columns,
        "regression",
        model_names=["Random Forest"],
        hyperparameter_search=True,
        cv_folds=3,
    )

    # Nested CV (grid search inside k-fold CV) is deliberately disabled --
    # the refit-on-single-split winner should carry the fixed defaults, not
    # a grid search result.
    assert not result.hyperparameter_search_applied
