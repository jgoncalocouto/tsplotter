from __future__ import annotations

import datetime as dt
from typing import List

import pandas as pd
import streamlit as st

from .core import (
    CategoricalCondition,
    Condition,
    DatetimeColumnCondition,
    DatetimeIndexCondition,
    NumericRangeCondition,
    TIME_INDEX_OPTION,
    infer_filterable_columns,
)


def _parse_csv_list(s: str) -> List[str]:
    return [v.strip() for v in s.split(",") if v.strip()]


def _default_year_from_series(series: pd.Series, fallback: int) -> int:
    try:
        ts = pd.to_datetime(series, errors="coerce").dropna()
        if len(ts) == 0:
            return fallback
        return int(ts.max().year)
    except Exception:
        return fallback


def render_filter_controls(
    df: pd.DataFrame,
    key_prefix: str = "flt",
    allow_time_index: bool = True,
) -> List[Condition]:
    """Render Streamlit controls for filters and return selected conditions."""
    info = infer_filterable_columns(df, include_index=allow_time_index)

    is_dt_index = info["is_datetime_index"]
    options = info["options"]
    dt_cols = set(info["datetime"])
    num_cols = set(info["numeric"])
    cat_cols = set(info["categorical"])

    if is_dt_index and len(df.index) > 0:
        default_year = int(pd.Timestamp(df.index.max()).year)
    else:
        default_year = dt.date.today().year

    n_filters = st.number_input(
        "Number of filtering conditions",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
        key=f"{key_prefix}_n",
        help="Each condition matches rows; you can chain with AND/OR. Empty numeric bounds mean −∞ / +∞.",
    )

    conditions: List[Condition] = []

    for i in range(int(n_filters)):
        with st.expander(f"Condition {i+1}", expanded=True):
            logic_op = st.selectbox(
                "Logical operator vs previous",
                ["AND", "OR"],
                index=0,
                key=f"{key_prefix}_logic_{i}",
            )
            var_i = st.selectbox(
                "Variable",
                options=options,
                index=0,
                key=f"{key_prefix}_var_{i}",
            )

            if var_i == TIME_INDEX_OPTION:
                use_low = st.checkbox(
                    "Set lower datetime bound (≥)",
                    value=False,
                    key=f"{key_prefix}_use_lowdt_idx_{i}",
                )
                if use_low:
                    low_date = st.date_input(
                        "Lower date",
                        value=dt.date(default_year, 1, 1),
                        key=f"{key_prefix}_low_date_idx_{i}",
                    )
                    low_time = st.time_input(
                        "Lower time",
                        value=dt.time(0, 0),
                        key=f"{key_prefix}_low_time_idx_{i}",
                    )
                    low_val = dt.datetime.combine(low_date, low_time)
                else:
                    low_val = None

                use_up = st.checkbox(
                    "Set upper datetime bound (≤)",
                    value=False,
                    key=f"{key_prefix}_use_updt_idx_{i}",
                )
                if use_up:
                    up_date = st.date_input(
                        "Upper date",
                        value=dt.date(default_year, 12, 31),
                        key=f"{key_prefix}_up_date_idx_{i}",
                    )
                    up_time = st.time_input(
                        "Upper time",
                        value=dt.time(23, 59, 59),
                        key=f"{key_prefix}_up_time_idx_{i}",
                    )
                    up_val = dt.datetime.combine(up_date, up_time)
                else:
                    up_val = None

                conditions.append(
                    DatetimeIndexCondition(
                        logic=logic_op,
                        low=low_val,
                        up=up_val,
                    )
                )

            elif var_i in dt_cols:
                col = df[var_i]
                col_year = _default_year_from_series(col, default_year)

                use_low = st.checkbox(
                    "Set lower datetime bound (≥)",
                    value=False,
                    key=f"{key_prefix}_use_lowdt_col_{i}",
                )
                if use_low:
                    low_date = st.date_input(
                        "Lower date",
                        value=dt.date(col_year, 1, 1),
                        key=f"{key_prefix}_low_date_col_{i}",
                    )
                    low_time = st.time_input(
                        "Lower time",
                        value=dt.time(0, 0),
                        key=f"{key_prefix}_low_time_col_{i}",
                    )
                    low_val = dt.datetime.combine(low_date, low_time)
                else:
                    low_val = None

                use_up = st.checkbox(
                    "Set upper datetime bound (≤)",
                    value=False,
                    key=f"{key_prefix}_use_updt_col_{i}",
                )
                if use_up:
                    up_date = st.date_input(
                        "Upper date",
                        value=dt.date(col_year, 12, 31),
                        key=f"{key_prefix}_up_date_col_{i}",
                    )
                    up_time = st.time_input(
                        "Upper time",
                        value=dt.time(23, 59, 59),
                        key=f"{key_prefix}_up_time_col_{i}",
                    )
                    up_val = dt.datetime.combine(up_date, up_time)
                else:
                    up_val = None

                conditions.append(
                    DatetimeColumnCondition(
                        logic=logic_op,
                        column=var_i,
                        low=low_val,
                        up=up_val,
                    )
                )

            elif var_i in num_cols:
                low_txt = st.text_input(
                    "Lower bound (≥) — empty = -inf",
                    key=f"{key_prefix}_low_num_{i}",
                )
                up_txt = st.text_input(
                    "Upper bound (≤) — empty = +inf",
                    key=f"{key_prefix}_up_num_{i}",
                )

                def _float_or_none(s: str):
                    s = s.strip()
                    if not s:
                        return None
                    try:
                        return float(s)
                    except Exception:
                        return None

                conditions.append(
                    NumericRangeCondition(
                        logic=logic_op,
                        column=var_i,
                        low=_float_or_none(low_txt),
                        up=_float_or_none(up_txt),
                    )
                )

            elif var_i in cat_cols:
                op = st.selectbox(
                    "Operation",
                    [
                        "is one of",
                        "is not one of",
                        "contains",
                        "not contains",
                        "starts with",
                        "ends with",
                        "regex",
                    ],
                    key=f"{key_prefix}_op_{i}",
                )
                case_sens = st.checkbox(
                    "Case sensitive",
                    value=False,
                    key=f"{key_prefix}_case_{i}",
                )
                na_match = st.checkbox(
                    "Treat NA as match",
                    value=False,
                    key=f"{key_prefix}_na_{i}",
                )

                ser_str = df[var_i].astype(str)
                uniq = pd.unique(ser_str.dropna())[:2000]

                if op in ["is one of", "is not one of"] and len(uniq) <= 100:
                    vals = st.multiselect(
                        "Values",
                        options=sorted(map(str, uniq)),
                        key=f"{key_prefix}_vals_{i}",
                    )
                elif op in ["is one of", "is not one of"]:
                    csv = st.text_input(
                        "Comma-separated values",
                        key=f"{key_prefix}_vals_csv_{i}",
                    )
                    vals = _parse_csv_list(csv)
                else:
                    patt = st.text_input(
                        "Pattern / text",
                        key=f"{key_prefix}_patt_{i}",
                    )
                    vals = [patt]

                conditions.append(
                    CategoricalCondition(
                        logic=logic_op,
                        column=var_i,
                        op=op,
                        values=vals,
                        case_sensitive=case_sens,
                        na_match=na_match,
                    )
                )

    return conditions
