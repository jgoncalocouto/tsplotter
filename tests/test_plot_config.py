"""Verify portable plot settings and the configuration trust boundary."""
import datetime as dt
import json

import pytest

from plot_config import (
    MAX_CONFIG_BYTES, decode_config, encode_config, is_config_key, setting_spec,
)


def config_bytes(settings, **header):
    return json.dumps({"app": "tsplotter", "version": 1, "settings": settings, **header}).encode()


def test_round_trip_plot_time_filter_and_independent_group_settings():
    settings = {
        "plot_type": "Grouped scatter", "setup_time_mode": "Absolute time",
        "setup_dt_col": "time", "setup_out_tz": "UTC", "sc2_x": "temperature",
        "sc2_y": "humidity", "sc2_g1": "temperature", "sc2_g2": "temperature",
        "sc2_color_bin_mode": "Number of bins", "sc2_color_bin_count": 5,
        "sc2_marker_bin_mode": "Bin width", "sc2_marker_bin_width": 2.5,
        "sc2_msize": 9, "sc2_mop": .65, "sc2_palette": "D3",
        "mlc_cols_0": ["temperature", "humidity"], "mlc_n_subplots": 2,
        "hist_binmode": "Fixed width", "hist_width_input": .125,
        "rw_window": 100, "der_smooth": 3, "overview_cols": ["temperature"],
        "flt2_n": 2, "flt2_var_0": "(time index)", "flt2_use_lowdt_idx_0": True,
        "flt2_low_date_idx_0": dt.date(2025, 1, 31), "flt2_low_time_idx_0": dt.time(12, 34, 56),
        "flt2_logic_1": "OR", "flt2_var_1": "shift", "flt2_op_1": "is one of",
        "flt2_vals_1": ["Nuit", "日"], "flt2_na_1": False,
    }
    original = settings.copy()
    payload = encode_config(settings)
    document = json.loads(payload)
    assert document["app"] == "tsplotter"
    assert document["version"] == 1
    assert document["settings"]["flt2_low_date_idx_0"] == "2025-01-31"
    assert document["settings"]["flt2_low_time_idx_0"] == "12:34:56"
    result = decode_config(payload)
    assert result.settings == settings
    assert result.warnings == []
    assert settings == original


def test_export_never_serializes_data_files_cache_or_trigger_widget_state():
    sentinel = object()
    state = {
        "plot_type": "Scatter", "raw_df": sentinel, "upload_meta": sentinel,
        "data_source": "Example dataset", "dataset_upload": sentinel, "dataset_id": "private",
        "sc_download": True, "sc2_download": False, "flt2_add": True, "flt2_clear": False,
        "sc2_color_bin_source": "temperature", "config_file": sentinel,
        "setup_untrusted": sentinel, "sc_unrecognized": sentinel,
    }
    assert decode_config(encode_config(state)).settings == {"plot_type": "Scatter"}


def test_unknown_settings_are_ignored_with_warnings_and_cannot_restore_data():
    result = decode_config(config_bytes({"plot_type": "Trends", "raw_df": [1, 2], "sc_download": True}))
    assert result.settings == {"plot_type": "Trends"}
    assert len(result.warnings) == 2


@pytest.mark.parametrize("key", [
    "mlc_cols_6", "mlc_cols_-1", "flt2_var_10", "flt2_var_0_extra", "sc_color_bin_source",
    "sc2_download", "flt2_clear", "raw_df", "data_source", "setup_anything",
])
def test_only_explicit_writable_keys_are_supported(key):
    assert not is_config_key(key)
    assert setting_spec(key) is None


@pytest.mark.parametrize("key,value", [
    ("mlc_n_subplots", 0), ("mlc_n_subplots", 7), ("mlc_n_subplots", True),
    ("mlc_n_subplots", 2.0), ("mlc_markers", "false"), ("mlc_width", 9),
    ("sc2_marker_bin_count", 101), ("sc2_color_bin_width", 0),
    ("sc2_marker_bin_width", -1), ("sc2_mop", .1), ("rw_window", 1),
    ("der_smooth", 10000), ("hist_bins_main", 201), ("hist_width_input", 1e-13),
    ("plot_type", "Unsupported"), ("sc2_palette", "Random"), ("sc2_color_bin_mode", "Quantiles"),
    ("overview_cols", "temperature"), ("mlc_cols_0", [1]), ("flt2_vals_0", [None]),
    ("flt2_n", 11), ("flt2_logic_0", "NOT"), ("flt2_low_num_0", 12),
    ("flt2_use_lowdt_idx_0", 1), ("flt2_low_date_idx_0", "2025-02-30"),
    ("flt2_low_date_idx_0", "2025-01-01T12:00:00"),
    ("flt2_low_time_idx_0", "25:00:00"), ("flt2_low_time_idx_0", "12:00:00+00:00"),
])
def test_invalid_known_setting_is_rejected_before_any_settings_are_returned(key, value):
    with pytest.raises(ValueError, match=key):
        decode_config(config_bytes({"plot_type": "Scatter", key: value}))


@pytest.mark.parametrize("data", [b"not json", b"", b"\xff", b"[]", b"null", b"42"])
def test_malformed_or_nonobject_files_are_rejected(data):
    with pytest.raises(ValueError):
        decode_config(data)


@pytest.mark.parametrize("version", [0, 2, True, "1", None])
def test_unsupported_version_is_rejected(version):
    with pytest.raises(ValueError, match="version"):
        decode_config(config_bytes({}, version=version))


def test_other_app_or_missing_settings_is_rejected():
    with pytest.raises(ValueError, match="TSPlotter"):
        decode_config(config_bytes({}, app="other-app"))
    with pytest.raises(ValueError, match="missing"):
        decode_config('{"app":"tsplotter","version":1}')
    with pytest.raises(ValueError, match="JSON object"):
        decode_config(config_bytes([1, 2]))


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_json_numbers_are_rejected_even_for_unknown_fields(constant):
    with pytest.raises(ValueError, match="finite"):
        decode_config('{"app":"tsplotter","version":1,"settings":{"future":' + constant + '}}')


def test_duplicate_fields_are_rejected_instead_of_loading_ambiguous_values():
    with pytest.raises(ValueError, match="Duplicate"):
        decode_config('{"app":"tsplotter","version":1,"settings":{"sc_msize":5,"sc_msize":8}}')


def test_size_limits_apply_to_import_and_export():
    with pytest.raises(ValueError, match="too large"):
        decode_config(b" " * (MAX_CONFIG_BYTES + 1))
    with pytest.raises(ValueError, match="too large"):
        encode_config({"flt2_vals_0": ["x" * 9999] * 30})


def test_utf8_bom_and_empty_settings_are_portable():
    assert decode_config(b"\xef\xbb\xbf" + config_bytes({})).settings == {}
    assert decode_config(encode_config({})).settings == {}


def test_capture_does_not_alias_selected_column_lists():
    original = {"mlc_cols_0": ["temperature"]}
    loaded = decode_config(encode_config(original))
    loaded.settings["mlc_cols_0"].append("humidity")
    assert original["mlc_cols_0"] == ["temperature"]


def test_huge_json_integer_is_a_validation_error_without_float_overflow():
    with pytest.raises(ValueError, match="finite"):
        decode_config(config_bytes({"sc2_color_bin_width": 10**400}))
