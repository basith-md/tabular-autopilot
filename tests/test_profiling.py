from tabular_autopilot.profiling import profile_dataframe
from tabular_autopilot.schema import infer_schema


def test_profile_reports_missingness_and_shape(mixed_type_df):
    schema = infer_schema(mixed_type_df, target="price")
    profile = profile_dataframe(mixed_type_df, schema)

    assert profile.n_rows == len(mixed_type_df)
    assert profile.n_cols == mixed_type_df.shape[1]
    assert "price" in profile.numeric_profiles
    assert "rooms" in profile.numeric_profiles


def test_profile_detects_missing_and_skew(regression_df):
    schema = infer_schema(regression_df, target="target")
    profile = profile_dataframe(regression_df, schema)

    assert profile.missing_by_col.get("x1", 0) > 0


def test_profile_categorical_top_values(mixed_type_df):
    schema = infer_schema(mixed_type_df, target="price")
    profile = profile_dataframe(mixed_type_df, schema)

    assert "city" in profile.categorical_top_values
    assert sum(profile.categorical_top_values["city"].values()) <= len(mixed_type_df)


def test_class_balance_none_for_regression(regression_df):
    schema = infer_schema(regression_df, target="target")
    profile = profile_dataframe(regression_df, schema)
    assert profile.class_balance is None


def test_class_balance_detects_imbalance(imbalanced_classification_df):
    schema = infer_schema(imbalanced_classification_df, target="label")
    profile = profile_dataframe(imbalanced_classification_df, schema)

    assert profile.class_balance is not None
    assert profile.class_balance.is_imbalanced
    assert profile.class_balance.imbalance_ratio > 1.5
    assert profile.class_balance.majority_class == "0"


def test_class_balance_not_flagged_when_roughly_even(classification_df):
    schema = infer_schema(classification_df, target="label")
    profile = profile_dataframe(classification_df, schema)

    assert profile.class_balance is not None
    assert profile.class_balance.imbalance_ratio < 1.5
