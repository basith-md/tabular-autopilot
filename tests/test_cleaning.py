from tabular_autopilot.cleaning import clean_dataframe
from tabular_autopilot.profiling import profile_dataframe
from tabular_autopilot.schema import infer_schema


def test_cleaning_imputes_missing_numeric(regression_df):
    schema = infer_schema(regression_df, target="target")
    profile = profile_dataframe(regression_df, schema)
    cleaned, report = clean_dataframe(regression_df, schema, profile)

    assert cleaned["x1"].isna().sum() == 0
    assert "x1" in report.imputed_numeric


def test_cleaning_imputes_missing_categorical(mixed_type_df):
    df = mixed_type_df.copy()
    df.loc[0:5, "city"] = None
    schema = infer_schema(df, target="price")
    profile = profile_dataframe(df, schema)
    cleaned, report = clean_dataframe(df, schema, profile)

    assert cleaned["city"].isna().sum() == 0
    assert "city" in report.imputed_categorical


def test_cleaning_log_transforms_skewed_columns(mixed_type_df):
    schema = infer_schema(mixed_type_df, target="price")
    profile = profile_dataframe(mixed_type_df, schema)
    cleaned, report = clean_dataframe(mixed_type_df, schema, profile)

    if report.log_transformed:
        for col in report.log_transformed:
            assert f"{col}_log" in cleaned.columns


def test_cleaning_drops_rows_with_missing_target(regression_df):
    df = regression_df.copy()
    df.loc[0, "target"] = None
    schema = infer_schema(df, target="target")
    profile = profile_dataframe(df, schema)
    cleaned, _ = clean_dataframe(df, schema, profile)

    assert cleaned["target"].isna().sum() == 0
    assert len(cleaned) == len(df) - 1


def test_outlier_capping_is_off_by_default(outlier_df):
    schema = infer_schema(outlier_df, target="target")
    profile = profile_dataframe(outlier_df, schema)
    cleaned, report = clean_dataframe(outlier_df, schema, profile)

    assert report.outlier_capping_applied is False
    assert report.outlier_capped == {}
    assert cleaned["x1"].max() > 400  # untouched extreme values still present


def test_outlier_capping_clips_extreme_values_when_enabled(outlier_df):
    schema = infer_schema(outlier_df, target="target")
    profile = profile_dataframe(outlier_df, schema)
    cleaned, report = clean_dataframe(outlier_df, schema, profile, cap_outliers=True)

    assert report.outlier_capping_applied is True
    assert "x1" in report.outlier_capped
    lower, upper = report.outlier_capped["x1"]
    assert cleaned["x1"].max() <= upper
    assert cleaned["x1"].min() >= lower
    assert cleaned["x1"].max() < 400  # the injected extremes got capped
