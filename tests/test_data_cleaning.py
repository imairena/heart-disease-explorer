import numpy as np
import pandas as pd

from data_cleaning import clean_data, TARGET_COL, CATEGORICAL_COLS


def _make_raw_df() -> pd.DataFrame:
    """Create a tiny raw-like DataFrame with typical UCI issues."""
    # Two rows with mixed issues:
    # - '?' and -9 as missing
    # - 0 values for chol/trestbps (invalid)
    data = {
        "age": [55, "?", 60],
        "sex": [1, 0, 1],
        "cp": [1, 4, -9],
        "trestbps": [140, 0, 130],
        "chol": [250, 0, -9],
        "fbs": [0, 1, 0],
        "restecg": [0, 2, "?"],
        "thalach": [150, 130, 120],
        "exang": [0, 1, 0],
        "oldpeak": [1.0, -9.0, 2.3],
        "slope": [2, "?", 3],
        "ca": [0, 2, "?"],
        "thal": [3, 7, -9],
        TARGET_COL: [0, 1, 4],
    }
    return pd.DataFrame(data)


def test_clean_data_binary_target_and_no_missing_markers():
    raw = _make_raw_df()
    cleaned = clean_data(raw)

    # After cleaning there should be no '?' or -9 placeholders anywhere
    for col in cleaned.columns:
        assert "?" not in cleaned[col].astype(str).values
        assert not (cleaned[col] == -9).any()
        assert not (cleaned[col] == -9.0).any()

    # Target should be strictly binary 0/1
    unique_targets = set(cleaned[TARGET_COL].unique())
    assert unique_targets.issubset({0, 1})


def test_clean_data_invalid_zero_values_are_imputed():
    raw = _make_raw_df()
    cleaned = clean_data(raw)

    # No zero chol or trestbps values should remain
    assert not (cleaned["chol"] == 0).any()
    assert not (cleaned["trestbps"] == 0).any()


def test_clean_data_categorical_columns_are_int_and_non_null():
    raw = _make_raw_df()
    cleaned = clean_data(raw)

    for col in CATEGORICAL_COLS:
        assert col in cleaned.columns
        # No missing values
        assert not cleaned[col].isna().any()
        # Stored as integers
        assert np.issubdtype(cleaned[col].dtype, np.integer)

