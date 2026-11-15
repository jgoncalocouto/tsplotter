"""Behavior checks for filter validation and recovery in the actual Streamlit UI."""

import datetime as dt

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def _app(data: str):
    return AppTest.from_string(
        "import pandas as pd\n"
        "import streamlit as st\n"
        "from filters.ui import render_filter_controls\n"
        "from filters.core import build_mask\n"
        f"df = {data}\n"
        "conditions = render_filter_controls(df)\n"
        "mask = build_mask(df, conditions)\n"
        "st.success(f'Results: {int(mask.sum())} rows')\n"
    ).run(timeout=15)


def test_add_clear_and_first_condition_logic():
    app = _app("pd.DataFrame({'value': [1, 2, 3]})")
    assert not app.exception
    assert app.button(key="flt_clear").disabled
    app.button(key="flt_add").click().run()
    assert len(app.selectbox) == 1  # First condition has no ignored AND/OR selector.
    app.button(key="flt_add").click().run()
    assert len(app.selectbox) == 3
    app.button(key="flt_clear").click().run()
    assert not app.exception
    assert len(app.selectbox) == 0
    assert app.success[0].value == "Results: 3 rows"


def test_invalid_numeric_and_reversed_bounds_block_results_until_fixed():
    app = _app("pd.DataFrame({'value': [1, 2, 3]})")
    app.button(key="flt_add").click().run()
    app.text_input(key="flt_low_num_0").input("oops").run()
    assert "valid number" in app.error[0].value
    assert not app.success
    app.text_input(key="flt_low_num_0").input("3")
    app.text_input(key="flt_up_num_0").input("1").run()
    assert "Minimum must" in app.error[0].value
    assert not app.success
    app.text_input(key="flt_low_num_0").input("1").run()
    assert not app.exception
    assert not app.error
    assert app.success[0].value == "Results: 1 rows"


def test_invalid_regex_can_be_repaired_without_crashing():
    app = _app("pd.DataFrame({'label': ['alpha', 'beta']})")
    app.button(key="flt_add").click().run()
    app.selectbox(key="flt_op_0").select("regex").run()
    app.text_input(key="flt_patt_0").input("[").run()
    assert not app.exception
    assert "regular expression is invalid" in app.error[0].value
    assert not app.success
    app.text_input(key="flt_patt_0").input("^alpha$").run()
    assert not app.exception
    assert app.success[0].value == "Results: 1 rows"


def test_datetime_defaults_cover_actual_multiyear_timezone_aware_data():
    app = _app(
        "pd.DataFrame({'value': [1, 2]}, index=pd.DatetimeIndex("
        "['2020-02-03 08:09:10', '2024-05-06 17:18:19'], tz='Europe/London'))"
    )
    app.button(key="flt_add").click().run()
    app.checkbox(key="flt_use_lowdt_idx_0").check().run()
    app.checkbox(key="flt_use_updt_idx_0").check().run()
    assert not app.exception
    assert app.date_input(key="flt_low_date_idx_0").value == dt.date(2020, 2, 3)
    assert app.time_input(key="flt_low_time_idx_0").value == dt.time(8, 9, 10)
    assert app.date_input(key="flt_up_date_idx_0").value == dt.date(2024, 5, 6)
    assert app.time_input(key="flt_up_time_idx_0").value == dt.time(17, 18, 19)
    assert app.success[0].value == "Results: 2 rows"


def test_no_columns_has_safe_empty_state():
    app = _app("pd.DataFrame()")
    assert not app.exception
    assert "no columns" in app.caption[0].value


def test_default_time_bounds_include_subsecond_samples():
    app = _app(
        "pd.DataFrame({'value': [1, 2]}, index=pd.DatetimeIndex("
        "['2024-05-06 17:18:19.123456789', '2024-05-06 17:18:20.999999999']))"
    )
    app.button(key="flt_add").click().run()
    app.checkbox(key="flt_use_lowdt_idx_0").check().run()
    app.checkbox(key="flt_use_updt_idx_0").check().run()
    assert not app.exception
    assert app.success[0].value == "Results: 2 rows"
