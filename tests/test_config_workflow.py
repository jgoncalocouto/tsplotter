"""Round-trip real configuration downloads through the app's explicit load workflow."""

import datetime as dt
import json
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
MAIN = Path(__file__).resolve().parents[1] / "main.py"


def _app(*, with_data=True, frame=None):
    source = (
        "import io\n"
        "import pandas as pd\n"
        "import streamlit as st\n"
        "from contextlib import ExitStack\n"
        "from pathlib import Path\n"
        "from unittest.mock import patch\n"
        "original_upload = st.file_uploader\n"
        "original_download = st.download_button\n"
        "def inject_upload(*args, **kwargs):\n"
        "    payload = st.session_state.get('_test_uploaded_config')\n"
        "    if kwargs.get('key') == 'config_upload' and payload is not None:\n"
        "        uploaded = io.BytesIO(payload)\n"
        "        uploaded.name = 'plot-configuration.json'\n"
        "        uploaded.size = len(payload)\n"
        "        return uploaded\n"
        "    return original_upload(*args, **kwargs)\n"
        "def capture_download(*args, **kwargs):\n"
        "    if kwargs.get('key') == 'config_download':\n"
        "        st.session_state['_test_saved_config'] = kwargs['data']\n"
        "    return original_download(*args, **kwargs)\n"
        "with ExitStack() as stack:\n"
        "    stack.enter_context(patch.object(st, 'file_uploader', inject_upload))\n"
        "    stack.enter_context(patch.object(st, 'download_button', capture_download))\n"
    )
    if frame is not None:
        source += f"    stack.enter_context(patch('workspace_ui.example_dataset', return_value={frame}))\n"
    source += f"    exec(compile(Path({str(MAIN)!r}).read_text(encoding='utf-8'), {str(MAIN)!r}, 'exec'))\n"
    app = AppTest.from_string(source)
    if with_data:
        app.session_state["data_source"] = "Example dataset"
    app.run(timeout=30)
    assert not app.exception
    return app


def _set(app, widget_type, key, value):
    getattr(app, widget_type)(key=key).set_value(value).run(timeout=30)
    assert not app.exception


def _save(app):
    app.run(timeout=30)
    assert not app.exception
    payload = app.session_state["_test_saved_config"]
    assert isinstance(payload, bytes)
    assert json.loads(payload)["app"] == "tsplotter"
    return payload


def _load(app, payload):
    app.session_state["_test_uploaded_config"] = payload
    app.run(timeout=30)
    assert not app.exception
    app.button(key="config_load").click().run(timeout=30)
    assert not app.exception


CHART_CASES = [
    ("Trends", [
        ("number_input", "mlc_n_subplots", 2),
        ("multiselect", "mlc_cols_0", ["humidity"]),
        ("multiselect", "mlc_cols_1", ["power_kw"]),
        ("selectbox", "mlc_shape", "spline"),
        ("slider", "mlc_width", 4),
    ]),
    ("Distribution", [
        ("multiselect", "hist_cols_main", ["humidity", "power_kw"]),
        ("selectbox", "hist_binmode", "Target #bins"),
        ("slider", "hist_bins_main", 70),
    ]),
    ("Scatter", [
        ("selectbox", "sc_x", "humidity"),
        ("selectbox", "sc_y", "power_kw"),
        ("selectbox", "sc_group", "temperature"),
        ("selectbox", "sc_color_bin_mode", "Number of bins"),
        ("number_input", "sc_color_bin_count", 4),
        ("slider", "sc_mop", .55),
    ]),
    ("Grouped scatter", [
        ("selectbox", "sc2_x", "humidity"),
        ("selectbox", "sc2_y", "power_kw"),
        ("selectbox", "sc2_g1", "temperature"),
        ("selectbox", "sc2_g2", "humidity"),
        ("selectbox", "sc2_color_bin_mode", "Number of bins"),
        ("number_input", "sc2_color_bin_count", 3),
        ("selectbox", "sc2_marker_bin_mode", "Bin width"),
        ("number_input", "sc2_marker_bin_width", 4.0),
    ]),
    ("Rolling window", [
        ("selectbox", "rw_y", "humidity"),
        ("number_input", "rw_window", 25),
        ("checkbox", "rw_include_current", False),
    ]),
    ("Derivatives", [
        ("selectbox", "der_y", "power_kw"),
        ("selectbox", "der_tunit", "minutes"),
        ("number_input", "der_smooth", 3),
        ("slider", "der_lw", 5),
    ]),
]


@pytest.mark.parametrize("chart,settings", CHART_CASES)
def test_saved_configuration_restores_each_chart_in_a_new_session(chart, settings):
    source = _app()
    _set(source, "radio", "plot_type", chart)
    for widget_type, key, value in settings:
        _set(source, widget_type, key, value)
    payload = _save(source)

    restored = _app()
    _load(restored, payload)
    assert restored.radio(key="plot_type").value == chart
    for widget_type, key, value in settings:
        assert getattr(restored, widget_type)(key=key).value == value
    assert len(restored.get("plotly_chart")) == 1


def test_saved_configuration_includes_settings_from_inactive_charts():
    source = _app()
    _set(source, "multiselect", "mlc_cols_0", ["humidity"])
    _set(source, "radio", "plot_type", "Distribution")
    _set(source, "multiselect", "hist_cols_main", ["power_kw"])
    restored = _app()
    _load(restored, _save(source))
    assert restored.radio(key="plot_type").value == "Distribution"
    assert restored.multiselect(key="hist_cols_main").value == ["power_kw"]
    _set(restored, "radio", "plot_type", "Trends")
    assert restored.multiselect(key="mlc_cols_0").value == ["humidity"]


def test_date_time_and_numeric_filters_round_trip_with_time_settings():
    source = _app()
    _set(source, "selectbox", "setup_out_tz", "Europe/Lisbon")
    source.button(key="flt2_add").click().run(timeout=30)
    _set(source, "checkbox", "flt2_use_lowdt_idx_0", True)
    _set(source, "date_input", "flt2_low_date_idx_0", dt.date(2025, 1, 2))
    _set(source, "time_input", "flt2_low_time_idx_0", dt.time(6, 30))
    _set(source, "checkbox", "flt2_use_updt_idx_0", True)
    _set(source, "date_input", "flt2_up_date_idx_0", dt.date(2025, 1, 4))
    _set(source, "time_input", "flt2_up_time_idx_0", dt.time(18, 45))
    source.button(key="flt2_add").click().run(timeout=30)
    _set(source, "selectbox", "flt2_var_1", "temperature")
    _set(source, "text_input", "flt2_low_num_1", "21")
    _set(source, "text_input", "flt2_up_num_1", "23.5")
    expected_rows = source.metric[0].value
    assert 0 < int(expected_rows.replace(",", "")) < 720

    restored = _app()
    _load(restored, _save(source))
    assert restored.selectbox(key="setup_out_tz").value == "Europe/Lisbon"
    assert restored.date_input(key="flt2_low_date_idx_0").value == dt.date(2025, 1, 2)
    assert restored.time_input(key="flt2_low_time_idx_0").value == dt.time(6, 30)
    assert restored.date_input(key="flt2_up_date_idx_0").value == dt.date(2025, 1, 4)
    assert restored.time_input(key="flt2_up_time_idx_0").value == dt.time(18, 45)
    assert restored.selectbox(key="flt2_var_1").value == "temperature"
    assert restored.text_input(key="flt2_low_num_1").value == "21"
    assert restored.text_input(key="flt2_up_num_1").value == "23.5"
    assert restored.metric[0].value == expected_rows


def test_configuration_can_be_loaded_before_a_dataset_is_chosen():
    source = _app()
    _set(source, "radio", "plot_type", "Derivatives")
    _set(source, "selectbox", "der_y", "power_kw")
    _set(source, "number_input", "der_smooth", 7)
    waiting = _app(with_data=False)
    _load(waiting, _save(source))
    assert waiting.session_state["raw_df"] is None
    assert not waiting.get("plotly_chart")
    _set(waiting, "radio", "data_source", "Example dataset")
    assert waiting.radio(key="plot_type").value == "Derivatives"
    assert waiting.selectbox(key="der_y").value == "power_kw"
    assert waiting.number_input(key="der_smooth").value == 7
    assert len(waiting.get("plotly_chart")) == 1


def test_uploading_a_configuration_only_changes_the_view_after_load_is_clicked():
    source = _app()
    _set(source, "radio", "plot_type", "Derivatives")
    _set(source, "selectbox", "der_y", "power_kw")
    app = _app()
    _set(app, "multiselect", "mlc_cols_0", ["humidity"])
    app.session_state["_test_uploaded_config"] = _save(source)
    app.run(timeout=30)
    assert not app.exception
    assert app.radio(key="plot_type").value == "Trends"
    assert app.multiselect(key="mlc_cols_0").value == ["humidity"]
    app.button(key="config_load").click().run(timeout=30)
    assert not app.exception
    assert app.radio(key="plot_type").value == "Derivatives"
    assert app.selectbox(key="der_y").value == "power_kw"


def test_loading_a_different_schema_reports_missing_columns_without_crashing():
    source = _app()
    _set(source, "radio", "plot_type", "Grouped scatter")
    _set(source, "selectbox", "sc2_g1", "temperature")
    _set(source, "selectbox", "sc2_g2", "humidity")
    source.button(key="flt2_add").click().run(timeout=30)
    _set(source, "selectbox", "flt2_var_0", "power_kw")
    _set(source, "text_input", "flt2_low_num_0", "2")
    replacement = (
        "pd.DataFrame({'timestamp': pd.date_range('2025-01-01', periods=20, freq='h'), "
        "'value': range(20), 'response': range(100, 120)})"
    )
    restored = _app(frame=replacement)
    _load(restored, _save(source))
    assert len(restored.get("plotly_chart")) == 1
    assert restored.metric[0].value == "20"
    notices = " ".join(message.value for message in restored.warning)
    assert any(column in notices for column in ("temperature", "humidity", "power_kw"))


@pytest.mark.parametrize("kind", ["malformed", "unsupported_version"])
def test_invalid_configuration_leaves_current_plot_settings_unchanged(kind):
    app = _app()
    _set(app, "multiselect", "mlc_cols_0", ["humidity"])
    _set(app, "slider", "mlc_width", 6)
    if kind == "malformed":
        payload = b"{this is not JSON"
    else:
        document = json.loads(_save(app))
        document["version"] = 999
        payload = json.dumps(document).encode("utf-8")
    _load(app, payload)
    assert app.error
    assert app.radio(key="plot_type").value == "Trends"
    assert app.multiselect(key="mlc_cols_0").value == ["humidity"]
    assert app.slider(key="mlc_width").value == 6
    assert len(app.get("plotly_chart")) == 1


def test_loading_defaults_clears_prior_overrides_in_other_charts_and_filters():
    defaults = _save(_app())
    app = _app()
    _set(app, "multiselect", "mlc_cols_0", ["power_kw"])
    _set(app, "slider", "mlc_width", 7)
    _set(app, "radio", "plot_type", "Derivatives")
    _set(app, "number_input", "der_smooth", 9)
    app.button(key="flt2_add").click().run(timeout=30)
    _set(app, "selectbox", "flt2_var_0", "temperature")
    _set(app, "text_input", "flt2_low_num_0", "23")
    _load(app, defaults)
    assert app.radio(key="plot_type").value == "Trends"
    assert app.multiselect(key="mlc_cols_0").value == ["temperature"]
    assert app.slider(key="mlc_width").value == 2
    assert app.metric[0].value == "720"
    assert app.button(key="flt2_clear").disabled
    _set(app, "radio", "plot_type", "Derivatives")
    assert app.number_input(key="der_smooth").value == 1


def test_loading_group_bins_replaces_stale_group_source_markers():
    source = _app()
    _set(source, "radio", "plot_type", "Grouped scatter")
    for widget_type, key, value in CHART_CASES[3][1]:
        _set(source, widget_type, key, value)
    app = _app()
    _set(app, "radio", "plot_type", "Grouped scatter")
    _set(app, "selectbox", "sc2_g1", "power_kw")
    _set(app, "selectbox", "sc2_g2", "temperature")
    _set(app, "number_input", "sc2_color_bin_count", 8)
    _set(app, "number_input", "sc2_marker_bin_count", 9)
    _load(app, _save(source))
    assert app.selectbox(key="sc2_g1").value == "temperature"
    assert app.selectbox(key="sc2_g2").value == "humidity"
    assert app.selectbox(key="sc2_color_bin_mode").value == "Number of bins"
    assert app.number_input(key="sc2_color_bin_count").value == 3
    assert app.selectbox(key="sc2_marker_bin_mode").value == "Bin width"
    assert app.number_input(key="sc2_marker_bin_width").value == 4.0


@pytest.mark.parametrize("invalid_view", ["filter", "bin_width"])
def test_valid_configuration_can_recover_an_invalid_current_view(invalid_view):
    valid_payload = _save(_app())
    app = _app()
    if invalid_view == "filter":
        app.button(key="flt2_add").click().run(timeout=30)
        _set(app, "selectbox", "flt2_var_0", "temperature")
        _set(app, "text_input", "flt2_low_num_0", "not a number")
    else:
        _set(app, "radio", "plot_type", "Grouped scatter")
        _set(app, "selectbox", "sc2_g1", "temperature")
        _set(app, "selectbox", "sc2_color_bin_mode", "Bin width")
        _set(app, "number_input", "sc2_color_bin_width", 0.0)
    assert app.error
    assert not app.get("plotly_chart")
    assert app.button(key="config_load")
    _load(app, valid_payload)
    assert not app.error
    assert app.radio(key="plot_type").value == "Trends"
    assert app.metric[0].value == "720"
    assert len(app.get("plotly_chart")) == 1


@pytest.mark.parametrize("source_count,target_count", [(200, 10), (10, 200)])
def test_categorical_membership_survives_a_change_between_csv_and_multiselect(source_count, target_count):
    def frame(cardinality):
        return (
            f"pd.DataFrame({{'timestamp': pd.date_range('2025-01-01', periods={cardinality * 2}, freq='min'), "
            f"'value': range({cardinality * 2}), "
            f"'category': ['g' + str(i % {cardinality}) for i in range({cardinality * 2})]}})"
        )

    source = _app(frame=frame(source_count))
    source.button(key="flt2_add").click().run(timeout=30)
    _set(source, "selectbox", "flt2_var_0", "category")
    if source_count > 100:
        _set(source, "text_input", "flt2_vals_csv_0", "g1, g3")
    else:
        _set(source, "multiselect", "flt2_vals_0", ["g1", "g3"])
    restored = _app(frame=frame(target_count))
    _load(restored, _save(source))
    assert restored.metric[0].value == "4"
    assert restored.selectbox(key="flt2_var_0").value == "category"
    if target_count > 100:
        assert restored.text_input(key="flt2_vals_csv_0").value == "g1, g3"
    else:
        assert restored.multiselect(key="flt2_vals_0").value == ["g1", "g3"]
    assert len(restored.get("plotly_chart")) == 1


def test_missing_saved_time_column_falls_back_to_row_numbers_with_feedback():
    saved = _save(_app())
    replacement = (
        "pd.DataFrame({'recorded_at': pd.date_range('2025-01-01', periods=20, freq='h'), "
        "'temperature': range(20)})"
    )
    restored = _app(frame=replacement)
    _load(restored, saved)
    assert restored.radio(key="setup_time_mode").value == "Relative time"
    assert restored.selectbox(key="setup_index").value == "(row number 0..N-1)"
    assert restored.metric[0].value == "20"
    assert len(restored.get("plotly_chart")) == 1
    assert any("timestamp" in warning.value and "unavailable" in warning.value
               for warning in restored.warning)


def test_missing_inactive_datetime_column_preserves_a_valid_saved_relative_axis():
    original = (
        "pd.DataFrame({'timestamp': pd.date_range('2025-01-01', periods=20, freq='h'), "
        "'temperature': range(20), 'elapsed': [i * .5 for i in range(20)]})"
    )
    source = _app(frame=original)
    _set(source, "radio", "setup_time_mode", "Relative time")
    _set(source, "selectbox", "setup_index", "elapsed")
    payload = _save(source)
    settings = json.loads(payload)["settings"]
    assert settings["setup_dt_col"] == "timestamp"
    assert settings["setup_index"] == "elapsed"
    replacement = "pd.DataFrame({'temperature': range(20), 'elapsed': [i * .5 for i in range(20)]})"
    restored = _app(frame=replacement)
    _load(restored, payload)
    assert restored.radio(key="setup_time_mode").value == "Relative time"
    assert restored.selectbox(key="setup_index").value == "elapsed"
    assert restored.dataframe[0].value.index.name == "elapsed"
    assert restored.dataframe[0].value.index.tolist() == [i * .5 for i in range(20)]
    assert len(restored.get("plotly_chart")) == 1


def test_saving_waits_for_loaded_time_settings_to_apply_pending_chart_and_filters():
    source = _app()
    _set(source, "selectbox", "setup_profile", "Custom strptime")
    _set(source, "radio", "plot_type", "Rolling window")
    _set(source, "selectbox", "rw_y", "humidity")
    _set(source, "number_input", "rw_window", 25)
    source.button(key="flt2_add").click().run(timeout=30)
    _set(source, "selectbox", "flt2_var_0", "temperature")
    _set(source, "text_input", "flt2_low_num_0", "21")
    payload = _save(source)
    replacement = (
        "pd.DataFrame({'timestamp': ['01/02/2025 08:00', '02/02/2025 08:00', '03/02/2025 08:00'], "
        "'temperature': [20., 22., 24.], 'humidity': [50., 55., 60.], 'power_kw': [2., 3., 4.]})"
    )
    restored = _app(frame=replacement)
    _load(restored, payload)
    assert "config_view_pending" in restored.session_state
    assert not restored.get("plotly_chart")
    download = next(element for element in restored.get("download_button") if element.key == "config_download")
    assert download.proto.disabled
    _set(restored, "text_input", "setup_format", "%d/%m/%Y %H:%M")
    assert "config_view_pending" not in restored.session_state
    assert restored.radio(key="plot_type").value == "Rolling window"
    assert restored.selectbox(key="rw_y").value == "humidity"
    assert restored.number_input(key="rw_window").value == 25
    assert restored.text_input(key="flt2_low_num_0").value == "21"
    assert restored.metric[0].value == "2"
    download = next(element for element in restored.get("download_button") if element.key == "config_download")
    assert not download.proto.disabled
    saved = json.loads(_save(restored))["settings"]
    assert saved["plot_type"] == "Rolling window"
    assert saved["rw_window"] == 25
    assert saved["flt2_low_num_0"] == "21"
    assert saved["setup_format"] == "%d/%m/%Y %H:%M"


def test_inactive_invalid_bin_width_does_not_block_saving_another_group_mode():
    frame = (
        "pd.DataFrame({'timestamp': pd.date_range('2025-01-01', periods=10, freq='h'), "
        "'temperature': range(20, 30), 'humidity': range(50, 60)})"
    )
    app = _app(frame=frame)
    _set(app, "radio", "plot_type", "Grouped scatter")
    _set(app, "selectbox", "sc2_g1", "temperature")
    for valid_mode in ("Number of bins", "Individual values"):
        _set(app, "selectbox", "sc2_color_bin_mode", "Bin width")
        _set(app, "number_input", "sc2_color_bin_width", 0.0)
        assert app.error
        assert not app.get("plotly_chart")
        download = next(element for element in app.get("download_button") if element.key == "config_download")
        assert download.proto.disabled
        _set(app, "selectbox", "sc2_color_bin_mode", valid_mode)
        assert not app.error
        assert len(app.get("plotly_chart")) == 1
        download = next(element for element in app.get("download_button") if element.key == "config_download")
        assert not download.proto.disabled
        saved = json.loads(_save(app))["settings"]
        assert saved["sc2_color_bin_mode"] == valid_mode
        assert "sc2_color_bin_width" not in saved
