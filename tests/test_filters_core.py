import datetime as dt

import pandas as pd

from filters.core import (
    CategoricalCondition,
    DatetimeColumnCondition,
    DatetimeIndexCondition,
    NumericRangeCondition,
    TIME_INDEX_OPTION,
    build_mask,
    infer_filterable_columns,
)


def test_infer_filterable_columns_detects_types():
    df = pd.DataFrame(
        {
            "num": [1, 2, 3],
            "cat": ["a", "b", "c"],
            "dt": pd.date_range("2022-01-01", periods=3),
        }
    )
    df.index = pd.date_range("2022-02-01", periods=3)

    info = infer_filterable_columns(df)

    assert info["is_datetime_index"] is True
    assert "dt" in info["datetime"]
    assert "num" in info["numeric"]
    assert "cat" in info["categorical"]
    assert info["options"][0] == TIME_INDEX_OPTION


def test_build_mask_datetime_index_condition_handles_bounds_and_tz():
    idx = pd.date_range("2022-01-01", periods=4, freq="D", tz="UTC")
    df = pd.DataFrame({"value": [10, 20, 30, 40]}, index=idx)

    cond = DatetimeIndexCondition(
        logic="AND",
        low=dt.datetime(2022, 1, 2, 0, 0),
        up=dt.datetime(2022, 1, 3, 12, 0),
    )

    mask = build_mask(df, [cond])
    assert mask.tolist() == [False, True, True, False]


def test_build_mask_datetime_column_condition():
    df = pd.DataFrame(
        {
            "ts": [
                "2022-01-01 00:00:00",
                "2022-01-02 10:00:00",
                "2022-01-03 00:00:00",
            ],
            "value": [1, 2, 3],
        }
    )

    cond = DatetimeColumnCondition(
        logic="AND",
        column="ts",
        low=dt.datetime(2022, 1, 2),
        up=dt.datetime(2022, 1, 2, 12, 0),
    )

    mask = build_mask(df, [cond])
    assert mask.tolist() == [False, True, False]


def test_build_mask_numeric_range_condition():
    df = pd.DataFrame({"num": [1, 2, 3, 4], "num_str": ["1", "2", "3", "4"]})

    cond = NumericRangeCondition(
        logic="AND",
        column="num_str",
        low=1.5,
        up=3.5,
    )

    mask = build_mask(df, [cond])
    assert mask.tolist() == [False, True, True, False]


def test_build_mask_categorical_operations_and_logic():
    df = pd.DataFrame(
        {
            "cat": ["apple", "banana", "pear", "plum"],
            "flag": ["yes", "no", "yes", "no"],
        }
    )

    cond1 = CategoricalCondition(
        logic="AND",
        column="cat",
        op="contains",
        values=["ap"],
        case_sensitive=False,
        na_match=False,
    )
    cond2 = CategoricalCondition(
        logic="OR",
        column="flag",
        op="is one of",
        values=["yes"],
        case_sensitive=False,
        na_match=False,
    )

    mask = build_mask(df, [cond1, cond2])
    # cond1 matches "apple"; cond2 OR adds rows with flag == yes
    assert mask.tolist() == [True, False, True, False]


def test_build_mask_regex_and_na_handling():
    df = pd.DataFrame(
        {
            "cat": pd.Series(["alpha", "beta", "gamma", pd.NA], dtype="string"),
        }
    )

    cond_regex = CategoricalCondition(
        logic="AND",
        column="cat",
        op="regex",
        values=["^a"],
        case_sensitive=False,
        na_match=False,
    )
    cond_na = CategoricalCondition(
        logic="OR",
        column="cat",
        op="is one of",
        values=["delta"],
        case_sensitive=False,
        na_match=True,
    )

    mask = build_mask(df, [cond_regex, cond_na])
    # Regex matches "alpha"; OR condition with na_match=True keeps string NA
    assert mask.tolist() == [True, False, False, True]
