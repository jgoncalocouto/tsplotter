from __future__ import annotations

import datetime as dt
import math
import re
from typing import Callable, List

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


def _datetime_defaults(values) -> tuple[dt.datetime, dt.datetime]:
    """Use the actual data extent, preserving wall time for timezone-aware data."""
    values = pd.to_datetime(values, errors="coerce")
    valid = values[values.notna()]
    if len(valid) == 0:
        today = dt.date.today()
        return dt.datetime.combine(today, dt.time.min), dt.datetime.combine(today, dt.time.max)
    # Round outward so time widgets' second precision includes subsecond data.
    low = pd.Timestamp(valid.min()).tz_localize(None).floor("s")
    high = pd.Timestamp(valid.max()).tz_localize(None).ceil("s")
    return (
        low.to_pydatetime().replace(tzinfo=None),
        high.to_pydatetime().replace(tzinfo=None),
    )


def _add_condition(key_prefix: str) -> None:
    count_key = f"{key_prefix}_n"
    st.session_state[count_key] = min(10, st.session_state.get(count_key, 0) + 1)


def _clear_conditions(key_prefix: str) -> None:
    for key in list(st.session_state):
        if key.startswith(f"{key_prefix}_") and key not in {
            f"{key_prefix}_add", f"{key_prefix}_clear"
        }:
            del st.session_state[key]
    st.session_state[f"{key_prefix}_n"] = 0


def _numeric_bound(text: str, label: str) -> tuple[float | None, str | None]:
    if not text.strip():
        return None, None
    try:
        value = float(text)
        if not math.isfinite(value):
            raise ValueError
        return value, None
    except ValueError:
        return None, f"Enter a valid number for {label}, or leave it empty for no limit."


def _datetime_controls(values, key_prefix: str, kind: str, i: int):
    low_default, up_default = _datetime_defaults(values)
    low, up = None, None
    left, right = st.columns(2)
    with left:
        if st.checkbox("Set start time", key=f"{key_prefix}_use_lowdt_{kind}_{i}"):
            low_date = st.date_input(
                "Start date", value=low_default.date(), key=f"{key_prefix}_low_date_{kind}_{i}"
            )
            low_time = st.time_input(
                "Start time", value=low_default.time(), step=1,
                key=f"{key_prefix}_low_time_{kind}_{i}"
            )
            low = dt.datetime.combine(low_date, low_time)
    with right:
        if st.checkbox("Set end time", key=f"{key_prefix}_use_updt_{kind}_{i}"):
            up_date = st.date_input(
                "End date", value=up_default.date(), key=f"{key_prefix}_up_date_{kind}_{i}"
            )
            up_time = st.time_input(
                "End time", value=up_default.time(), step=1,
                key=f"{key_prefix}_up_time_{kind}_{i}"
            )
            up = dt.datetime.combine(up_date, up_time)
    return low, up


def render_filter_controls(
    df: pd.DataFrame,
    key_prefix: str = "flt",
    allow_time_index: bool = True,
    on_invalid: Callable[[], None] | None = None,
) -> List[Condition]:
    """Render filters; stop downstream charts until invalid conditions are fixed."""
    info = infer_filterable_columns(df, include_index=allow_time_index)
    options = info["options"]
    dt_cols = set(info["datetime"])
    num_cols = set(info["numeric"])
    cat_cols = set(info["categorical"])

    if not options:
        st.caption("There are no columns available to filter.")
        return []

    count_key = f"{key_prefix}_n"
    if count_key not in st.session_state:
        st.session_state[count_key] = 0
    n_filters = int(st.session_state[count_key])
    add_col, clear_col, count_col = st.columns([1, 1, 2])
    with add_col:
        st.button(
            "Add condition", key=f"{key_prefix}_add", on_click=_add_condition,
            args=(key_prefix,), disabled=n_filters >= 10, use_container_width=True,
        )
    with clear_col:
        st.button(
            "Clear filters", key=f"{key_prefix}_clear", on_click=_clear_conditions,
            args=(key_prefix,), disabled=n_filters == 0, use_container_width=True,
        )
    with count_col:
        st.caption(f"{n_filters} of 10 conditions" if n_filters else "All rows included")
    if not n_filters:
        st.caption("Add a condition to narrow the data. Empty bounds keep the full range.")

    conditions: List[Condition] = []
    invalid = False
    for i in range(n_filters):
        with st.container(border=True):
            st.markdown(f"**Condition {i + 1}**")
            logic_op = "AND"
            if i:
                logic_op = st.selectbox(
                    "Combine with previous conditions", ["AND", "OR"],
                    key=f"{key_prefix}_logic_{i}",
                    help="AND keeps rows matching both; OR keeps rows matching either.",
                )
            var_i = st.selectbox("Column", options=options, key=f"{key_prefix}_var_{i}")

            if var_i == TIME_INDEX_OPTION or var_i in dt_cols:
                is_index = var_i == TIME_INDEX_OPTION
                values = df.index if is_index else df[var_i]
                low, up = _datetime_controls(values, key_prefix, "idx" if is_index else "col", i)
                if low is not None and up is not None and low > up:
                    st.error("Start time must be before or equal to end time.")
                    invalid = True
                if is_index:
                    conditions.append(DatetimeIndexCondition(logic=logic_op, low=low, up=up))
                else:
                    conditions.append(DatetimeColumnCondition(logic=logic_op, column=var_i, low=low, up=up))

            elif var_i in num_cols:
                left, right = st.columns(2)
                with left:
                    low_txt = st.text_input(
                        "Minimum (inclusive)", key=f"{key_prefix}_low_num_{i}",
                        placeholder="No minimum",
                    )
                with right:
                    up_txt = st.text_input(
                        "Maximum (inclusive)", key=f"{key_prefix}_up_num_{i}",
                        placeholder="No maximum",
                    )
                low, low_error = _numeric_bound(low_txt, "minimum")
                up, up_error = _numeric_bound(up_txt, "maximum")
                for error in (low_error, up_error):
                    if error:
                        st.error(error)
                        invalid = True
                if low is not None and up is not None and low > up:
                    st.error("Minimum must be less than or equal to maximum.")
                    invalid = True
                conditions.append(NumericRangeCondition(logic=logic_op, column=var_i, low=low, up=up))

            elif var_i in cat_cols:
                op = st.selectbox(
                    "Match", ["is one of", "is not one of", "contains", "not contains",
                              "starts with", "ends with", "regex"],
                    key=f"{key_prefix}_op_{i}",
                )
                uniq = pd.unique(df[var_i].dropna().astype(str))[:2000]
                if op in {"is one of", "is not one of"} and len(uniq) <= 100:
                    vals = st.multiselect("Values", options=sorted(uniq), key=f"{key_prefix}_vals_{i}")
                elif op in {"is one of", "is not one of"}:
                    csv = st.text_input("Comma-separated values", key=f"{key_prefix}_vals_csv_{i}")
                    vals = _parse_csv_list(csv)
                else:
                    pattern = st.text_input("Pattern / text", key=f"{key_prefix}_patt_{i}")
                    vals = [pattern]
                    if op == "regex":
                        try:
                            re.compile(pattern)
                        except re.error:
                            st.error("This regular expression is invalid. Check brackets and special characters.")
                            invalid = True
                case_sens = False
                if op not in {"is one of", "is not one of"}:
                    case_sens = st.checkbox("Case sensitive", key=f"{key_prefix}_case_{i}")
                na_match = st.checkbox("Include missing values", key=f"{key_prefix}_na_{i}")
                conditions.append(CategoricalCondition(
                    logic=logic_op, column=var_i, op=op, values=vals,
                    case_sensitive=case_sens, na_match=na_match,
                ))

    if invalid:
        st.info("Fix the highlighted conditions or clear filters to continue.")
        if on_invalid is not None:
            on_invalid()
        st.stop()
    return conditions
