import pandas as pd
from pandas.testing import assert_frame_equal

from scripts.rename_columns import rename_columns


def test_rename_columns():
    test_data = pd.DataFrame(
        {"date": ["2025-01-01", "2025-01-02"], "value": [100, 200]}
    )

    expected_data = pd.DataFrame(
        {"ds": ["2025-01-01", "2025-01-02"], "y": [100, 200]}
    )

    result_data = rename_columns(test_data)

    assert_frame_equal(result_data, expected_data)
