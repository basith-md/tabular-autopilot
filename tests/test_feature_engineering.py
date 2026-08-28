from tabular_autopilot.cleaning import clean_dataframe
from tabular_autopilot.feature_engineering import engineer_features
from tabular_autopilot.profiling import profile_dataframe
from tabular_autopilot.schema import infer_schema


def _cleaned(df, target):
    schema = infer_schema(df, target=target)
    profile = profile_dataframe(df, schema)
    cleaned, _ = clean_dataframe(df, schema, profile)
    return schema, cleaned


def test_low_cardinality_categorical_is_one_hot_encoded(regression_df):
    schema, cleaned = _cleaned(regression_df, "target")
    featured, report = engineer_features(cleaned, schema)

    assert "category" in report.one_hot_encoded
    assert any(col.startswith("category_") for col in featured.columns)
    assert "category" not in featured.columns


def test_datetime_column_is_expanded_and_dropped(mixed_type_df):
    schema, cleaned = _cleaned(mixed_type_df, "price")
    featured, report = engineer_features(cleaned, schema)

    assert "signup_date" in report.datetime_expanded
    assert "signup_date" not in featured.columns
    for expected_suffix in ("_year", "_month", "_day", "_dayofweek", "_is_weekend"):
        assert f"signup_date{expected_suffix}" in featured.columns


def test_geo_columns_produce_cluster_and_distance_features(mixed_type_df):
    schema, cleaned = _cleaned(mixed_type_df, "price")
    featured, report = engineer_features(cleaned, schema)

    assert "geo_cluster" in featured.columns
    assert "geo_dist_to_centroid" in featured.columns
    assert report.geo_features_added == ["geo_cluster", "geo_dist_to_centroid"]


def test_identifier_constant_and_text_columns_are_dropped(mixed_type_df):
    schema, cleaned = _cleaned(mixed_type_df, "price")
    featured, report = engineer_features(cleaned, schema)

    for col in ("row_id", "constant_col", "notes"):
        assert col not in featured.columns
        assert col in report.dropped_columns


def test_target_column_survives_untouched(regression_df):
    schema, cleaned = _cleaned(regression_df, "target")
    featured, _ = engineer_features(cleaned, schema)
    assert "target" in featured.columns
