import numpy as np
import pandas as pd

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


def test_identifier_and_constant_columns_are_dropped(mixed_type_df):
    schema, cleaned = _cleaned(mixed_type_df, "price")
    featured, report = engineer_features(cleaned, schema)

    for col in ("row_id", "constant_col"):
        assert col not in featured.columns
        assert col in report.dropped_columns


def test_text_column_is_tfidf_vectorized_when_enough_rows(mixed_type_df):
    schema, cleaned = _cleaned(mixed_type_df, "price")
    featured, report = engineer_features(cleaned, schema, vectorize_text=True)

    assert "notes" in report.text_vectorized
    assert "notes" not in featured.columns
    assert any(col.startswith("notes_tfidf_") for col in featured.columns)
    assert "notes" not in report.dropped_columns


def test_text_column_is_dropped_when_vectorization_disabled(mixed_type_df):
    schema, cleaned = _cleaned(mixed_type_df, "price")
    featured, report = engineer_features(cleaned, schema, vectorize_text=False)

    assert "notes" not in featured.columns
    assert "notes" in report.dropped_columns
    assert not any(col.startswith("notes_tfidf_") for col in featured.columns)


def test_target_column_survives_untouched(regression_df):
    schema, cleaned = _cleaned(regression_df, "target")
    featured, _ = engineer_features(cleaned, schema)
    assert "target" in featured.columns


def _high_cardinality_df():
    rng = np.random.default_rng(9)
    n = 200
    codes = [f"C{i % 30}" for i in range(n)]
    target = rng.normal(size=n)
    return pd.DataFrame({"code": codes, "target": target})


def test_high_cardinality_uses_frequency_encoding_by_default():
    df = _high_cardinality_df()
    schema, cleaned = _cleaned(df, "target")
    assert "code" in schema.categorical_high_cols  # sanity check on the fixture's role inference

    featured, report = engineer_features(cleaned, schema)

    assert "code" in report.frequency_encoded
    assert "code_freq" in featured.columns
    assert "code" not in featured.columns
    assert report.deferred_target_encoding == []


def test_high_cardinality_target_encoding_is_deferred_to_modeling():
    df = _high_cardinality_df()
    schema, cleaned = _cleaned(df, "target")

    featured, report = engineer_features(cleaned, schema, high_cardinality_encoding="target")

    assert report.deferred_target_encoding == ["code"]
    assert report.frequency_encoded == []
    assert "code" in featured.columns  # left raw -- modeling.py encodes it after the train/test split
    assert featured["code"].dtype == object


def test_target_encoding_falls_back_to_frequency_without_a_target():
    df = _high_cardinality_df()
    schema, cleaned = _cleaned(df, target=None)

    featured, report = engineer_features(cleaned, schema, high_cardinality_encoding="target")

    assert report.deferred_target_encoding == []
    assert "code" in report.frequency_encoded
