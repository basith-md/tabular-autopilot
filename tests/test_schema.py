from tabular_autopilot.schema import ColumnRole, infer_schema


def test_infers_expected_roles(mixed_type_df):
    schema = infer_schema(mixed_type_df, target="price")

    assert schema.columns["price"].role == ColumnRole.TARGET
    assert schema.columns["row_id"].role == ColumnRole.IDENTIFIER
    assert schema.columns["rooms"].role == ColumnRole.NUMERIC
    assert schema.columns["city"].role == ColumnRole.CATEGORICAL_LOW
    assert schema.columns["notes"].role == ColumnRole.TEXT
    assert schema.columns["constant_col"].role == ColumnRole.CONSTANT
    assert schema.columns["signup_date"].role == ColumnRole.DATETIME
    assert schema.columns["lat"].role == ColumnRole.GEO_LAT
    assert schema.columns["lon"].role == ColumnRole.GEO_LON

    assert schema.has_geo
    assert schema.geo_lat_col == "lat"
    assert schema.geo_lon_col == "lon"
    assert "row_id" not in schema.feature_cols
    assert "constant_col" not in schema.feature_cols
    assert "signup_date" not in schema.feature_cols


def test_task_detection_regression(regression_df):
    schema = infer_schema(regression_df, target="target")
    assert schema.task == "regression"


def test_task_detection_classification(classification_df):
    schema = infer_schema(classification_df, target="label")
    assert schema.task == "classification"


def test_no_target_is_fine(mixed_type_df):
    schema = infer_schema(mixed_type_df, target=None)
    assert schema.target is None
    assert schema.task is None
