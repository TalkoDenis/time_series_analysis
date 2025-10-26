import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from scripts.validate_content import validate_content
from scripts.validate_structure import validate_structure


def test_validate_structure_success(sample_raw_df):
    try:
        result_df = validate_structure(sample_raw_df)
        assert_frame_equal(result_df, sample_raw_df)
    except Exception as e:
        pytest.fail(f"validate_structure() had fallen! Error: {e}")


def test_validate_structure_empty_df():
    empty_df = pd.DataFrame()

    with pytest.raises(Exception, match="DataFrame is empty!"):
        validate_structure(empty_df)


def test_validate_structure_too_few_columns():
    one_col_df = pd.DataFrame({"date": ["2025-01-01"]})

    with pytest.raises(Exception, match="Expected at least 2 columns"):
        validate_structure(one_col_df)


def test_validate_content_success(sample_prophet_df):
    try:
        validate_content(sample_prophet_df)
    except Exception as e:
        pytest.fail(f"validate_content() had fallen! Error: {e}")


def test_validate_content_bad_date(sample_prophet_df):
    bad_df = sample_prophet_df.copy()
    bad_df.loc[0, "ds"] = "Is not a date!"

    with pytest.raises(Exception, match="Invalid data in date column"):
        validate_content(bad_df)


def test_validate_content_bad_value(sample_prophet_df):
    bad_df = sample_prophet_df.copy()
    bad_df["y"] = bad_df["y"].astype(object)
    bad_df.loc[0, "y"] = "Is not a number"

    with pytest.raises(Exception, match="Invalid data in value column"):
        validate_content(bad_df)


def test_validate_content_sorts_data(sample_prophet_df):
    unsorted_df = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2025-01-03", "2025-01-01", "2025-01-02"]),
            "y": [300, 100, 200],
        }
    )

    sorted_df = validate_content(unsorted_df)

    assert sorted_df["ds"].iloc[0] == pd.to_datetime("2025-01-01")
    assert list(sorted_df.index) == [0, 1, 2]
