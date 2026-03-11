"""Pre-train checks: run before DVC pipeline to validate data integrity."""

from pathlib import Path

import pandas as pd
import pytest

RAW_CSV = Path("data/raw/creditcard.csv")
TRAIN_CSV = Path("data/prepared/train.csv")
TEST_CSV = Path("data/prepared/test.csv")

RAW_REQUIRED_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount", "Class"]
PREPARED_REQUIRED_COLUMNS = [f"V{i}" for i in range(1, 29)] + [
    "hour_of_day",
    "Amount_scaled",
    "Class",
]


# ---------------------------------------------------------------------------
# Raw data checks
# ---------------------------------------------------------------------------


class TestRawData:
    @pytest.fixture(scope="class")
    def raw_df(self):
        if not RAW_CSV.exists():
            pytest.skip("Raw data not available (run dvc pull)")
        return pd.read_csv(RAW_CSV)

    def test_raw_file_exists(self):
        assert RAW_CSV.exists(), f"Raw dataset not found at {RAW_CSV}"

    def test_required_columns_present(self, raw_df):
        missing = set(RAW_REQUIRED_COLUMNS) - set(raw_df.columns)
        assert not missing, f"Missing columns in raw data: {missing}"

    def test_target_column_binary(self, raw_df):
        values = set(raw_df["Class"].unique())
        assert values <= {0, 1}, f"Class column has unexpected values: {values}"

    def test_no_empty_dataframe(self, raw_df):
        assert len(raw_df) > 0, "Raw dataset is empty"

    def test_fraud_cases_exist(self, raw_df):
        assert raw_df["Class"].sum() > 0, "No fraud cases (Class=1) found in raw data"

    def test_amount_non_negative(self, raw_df):
        assert (raw_df["Amount"] >= 0).all(), "Negative values found in Amount column"


# ---------------------------------------------------------------------------
# Prepared data checks
# ---------------------------------------------------------------------------


class TestPreparedData:
    @pytest.fixture(scope="class")
    def train_df(self):
        if not TRAIN_CSV.exists():
            pytest.skip("Prepared data not available (run dvc repro)")
        return pd.read_csv(TRAIN_CSV)

    @pytest.fixture(scope="class")
    def test_df(self):
        if not TEST_CSV.exists():
            pytest.skip("Prepared data not available (run dvc repro)")
        return pd.read_csv(TEST_CSV)

    def test_prepared_files_exist(self):
        assert TRAIN_CSV.exists(), f"Missing {TRAIN_CSV}"
        assert TEST_CSV.exists(), f"Missing {TEST_CSV}"

    def test_required_columns_train(self, train_df):
        missing = set(PREPARED_REQUIRED_COLUMNS) - set(train_df.columns)
        assert not missing, f"Missing columns in train.csv: {missing}"

    def test_required_columns_test(self, test_df):
        missing = set(PREPARED_REQUIRED_COLUMNS) - set(test_df.columns)
        assert not missing, f"Missing columns in test.csv: {missing}"

    def test_no_nulls_train(self, train_df):
        null_counts = train_df.isnull().sum()
        nulls = null_counts[null_counts > 0]
        assert nulls.empty, f"Null values found in train.csv:\n{nulls}"

    def test_no_nulls_test(self, test_df):
        null_counts = test_df.isnull().sum()
        nulls = null_counts[null_counts > 0]
        assert nulls.empty, f"Null values found in test.csv:\n{nulls}"

    def test_train_larger_than_test(self, train_df, test_df):
        assert len(train_df) > len(test_df), "train.csv should be larger than test.csv"

    def test_hour_of_day_range(self, train_df):
        assert (
            train_df["hour_of_day"].between(0, 23).all()
        ), "hour_of_day contains values outside [0, 23]"

    def test_both_classes_in_train(self, train_df):
        assert set(train_df["Class"].unique()) == {
            0,
            1,
        }, "train.csv must contain both fraud (1) and non-fraud (0) samples"

    def test_column_schema_match(self, train_df, test_df):
        assert list(train_df.columns) == list(
            test_df.columns
        ), "train.csv and test.csv have different column schemas"
