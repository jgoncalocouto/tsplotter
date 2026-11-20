"""Save/load controls and dataset-aware application of plotting configurations."""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from filters.core import TIME_INDEX_OPTION as FILTER_TIME_INDEX, infer_filterable_columns
from plot_config import decode_config, encode_config, is_config_key


ROW_INDEX = "(row number 0..N-1)"
TIME_INDEX = "(use existing time index)"
BIN_SOURCES = {
    "sc_color": "sc_group", "sc2_color": "sc2_g1", "sc2_marker": "sc2_g2",
}


def preserve_configuration_state() -> None:
    """Keep settings for views or optional controls that are not currently shown."""
    for key in list(st.session_state):
        if is_config_key(key) or key in {f"{prefix}_bin_source" for prefix in BIN_SOURCES}:
            st.session_state[key] = st.session_state[key]


def queue_configuration(data: bytes) -> None:
    """Validate a complete upload before changing any current plotting settings."""
    try:
        loaded = decode_config(data)
    except ValueError as error:
        st.session_state.config_error = f"Could not load configuration: {error} Current settings were kept."
        return
    st.session_state.config_pending = loaded.settings
    st.session_state.config_warnings = loaded.warnings
    st.session_state.pop("config_error", None)
    st.session_state.config_status = "Configuration loaded. Choose a dataset to apply it."


def _warn(message: str) -> None:
    warnings = st.session_state.setdefault("config_warnings", [])
    if message not in warnings:
        warnings.append(message)


def apply_pending_time_settings(raw: pd.DataFrame) -> None:
    """Restore time settings before any of their widgets are created."""
    if "config_pending" not in st.session_state:
        return
    settings = dict(st.session_state.pop("config_pending"))
    for key in list(st.session_state):
        if is_config_key(key) or key in {f"{prefix}_bin_source" for prefix in BIN_SOURCES}:
            del st.session_state[key]

    valid_time = set(raw.columns) | {""}
    if isinstance(raw.index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        valid_time.add(TIME_INDEX)
    if "setup_dt_col" in settings and settings["setup_dt_col"] not in valid_time:
        missing = settings.pop("setup_dt_col")
        if settings.get("setup_time_mode") == "Absolute time":
            _warn(f"Saved time column '{missing}' is unavailable. Relative time with row numbers is being used; you can choose another time column.")
            settings["setup_time_mode"] = "Relative time"
            settings["setup_index"] = ROW_INDEX
        else:
            _warn(f"Skipped unavailable inactive time column '{missing}'. The saved relative axis is unchanged.")
    if "setup_index" in settings and settings["setup_index"] not in set(raw.columns) | {ROW_INDEX}:
        missing = settings["setup_index"]
        settings["setup_index"] = ROW_INDEX
        _warn(f"Saved relative X-axis column '{missing}' is unavailable and has been reset to row numbers.")
    for key in list(settings):
        if key.startswith("setup_"):
            st.session_state[key] = settings.pop(key)
    st.session_state.config_view_pending = settings
    st.session_state.config_status = "Time settings loaded. The chart settings will apply when the time axis is usable."


def _compatible_filters(settings: dict[str, Any], frame: pd.DataFrame) -> None:
    """Drop unavailable conditions and rebuild contiguous filter indices."""
    keys = [key for key in settings if key.startswith("flt2_")]
    saved = {key: settings.pop(key) for key in keys}
    count = saved.get("flt2_n", 0)
    info = infer_filterable_columns(frame, include_index=True)
    options = set(info["options"])
    kept = 0
    for index in range(count):
        fields = {match.group(1): value for key, value in saved.items()
                  if (match := re.fullmatch(r"flt2_(.+)_" + str(index), key))}
        column = fields.get("var")
        if column not in options:
            _warn(f"Skipped saved filter {index + 1}: column '{column}' is unavailable for filtering.")
            continue
        if column == FILTER_TIME_INDEX:
            kind = "idx"
            allowed = {"use_lowdt_idx", "use_updt_idx", "low_date_idx", "up_date_idx", "low_time_idx", "up_time_idx"}
        elif column in info["datetime"]:
            kind = "col"
            allowed = {"use_lowdt_col", "use_updt_col", "low_date_col", "up_date_col", "low_time_col", "up_time_col"}
        elif column in info["numeric"]:
            kind = "numeric"
            allowed = {"low_num", "up_num"}
        else:
            kind = "categorical"
            allowed = {"op", "vals", "vals_csv", "patt", "case", "na"}
        if not allowed.intersection(fields):
            _warn(f"Skipped saved filter {index + 1}: '{column}' has a different data type in this dataset.")
            continue
        if kind == "categorical":
            unique = pd.unique(frame[column].dropna().astype(str))[:2000]
            membership = fields.get("op", "is one of") in {"is one of", "is not one of"}
            if membership:
                values = fields.get("vals")
                if values is None:
                    values = [item.strip() for item in fields.get("vals_csv", "").split(",") if item.strip()]
                if len(unique) <= 100:
                    available = set(unique)
                    valid = [value for value in values if value in available]
                    if values and not valid:
                        _warn(f"Skipped saved filter {index + 1}: none of its values exist in '{column}'.")
                        continue
                    if len(valid) < len(values):
                        _warn(f"Unavailable values were removed from the saved filter on '{column}'.")
                    fields["vals"] = valid
                    fields.pop("vals_csv", None)
                else:
                    fields["vals_csv"] = ", ".join(values)
                    fields.pop("vals", None)
        for field, value in fields.items():
            if field in allowed | {"logic", "var"}:
                settings[f"flt2_{field}_{kept}"] = value
        kept += 1
    settings["flt2_n"] = kept


def apply_pending_view_settings(frame: pd.DataFrame) -> None:
    """Restore chart/filter settings using only compatible columns in this table."""
    if "config_view_pending" not in st.session_state:
        return
    settings = dict(st.session_state.pop("config_view_pending"))
    columns = set(frame.columns)
    numeric = set(frame.select_dtypes(include=[np.number]).columns)
    selector_names = {
        "sc_x": "Scatter X", "sc_y": "Scatter Y", "sc_group": "Scatter color",
        "sc2_x": "Grouped scatter X", "sc2_y": "Grouped scatter Y",
        "sc2_g1": "Grouped scatter color", "sc2_g2": "Grouped scatter marker",
        "rw_y": "Rolling signal", "der_y": "Derivative signal",
    }
    for key, label in selector_names.items():
        if key not in settings:
            continue
        valid = numeric if key in {"rw_y", "der_y"} else columns
        if key in {"sc_group", "sc2_g1", "sc2_g2"}:
            valid = valid | {"(none)"}
        if settings[key] not in valid:
            _warn(f"Saved {label} column '{settings[key]}' is unavailable. Choose a replacement in the chart controls.")
            settings.pop(key)
    for key in ["hist_cols_main", "overview_cols", *[f"mlc_cols_{i}" for i in range(6)]]:
        if key in settings:
            missing = [column for column in settings[key] if column not in numeric]
            if missing:
                _warn(f"Unavailable numeric columns were skipped: {', '.join(missing)}.")
            settings[key] = [column for column in settings[key] if column in numeric]

    for prefix, selector in BIN_SOURCES.items():
        column = settings.get(selector, "(none)")
        if column in numeric and not pd.api.types.is_bool_dtype(frame[column]):
            st.session_state[f"{prefix}_bin_source"] = column
        else:
            for suffix in ("mode", "count", "width"):
                key = f"{prefix}_bin_{suffix}"
                if key in settings:
                    if settings[key] not in {"Individual values", 5}:
                        _warn("Saved numeric bin settings were skipped for a missing or nonnumeric grouping column.")
                    settings.pop(key)
            st.session_state[f"{prefix}_bin_source"] = column
    _compatible_filters(settings, frame)
    for key, value in settings.items():
        st.session_state[key] = value
    st.session_state.config_status = "Configuration applied."


def _snapshot_settings() -> dict[str, Any]:
    settings = {key: st.session_state[key] for key in st.session_state if is_config_key(key)}
    # An invalid old width must not block saving after the user switches away
    # from width mode. Only the parameters used by the saved mode are needed.
    for prefix in BIN_SOURCES:
        if settings.get(f"{prefix}_bin_mode") != "Bin width":
            settings.pop(f"{prefix}_bin_width", None)
        if settings.get(f"{prefix}_bin_mode") != "Number of bins":
            settings.pop(f"{prefix}_bin_count", None)
    return settings


def render_configuration_panel() -> None:
    """Render late so downloads contain the just-created widget defaults too."""
    with st.sidebar.expander("Plot configuration"):
        st.caption("Save chart choices, display options, bins, time parsing, and filters. The dataset is loaded separately.")
        uploaded = st.file_uploader("Configuration file", type=["json"], key="config_upload")
        st.button("Load configuration", key="config_load", disabled=uploaded is None,
                  on_click=queue_configuration, args=(uploaded.getvalue(),) if uploaded is not None else (b"",),
                  use_container_width=True)
        if st.session_state.get("config_error"):
            st.error(st.session_state.config_error)
        if st.session_state.get("config_status"):
            st.success(st.session_state.config_status)
        for warning in st.session_state.get("config_warnings", []):
            st.warning(warning)
        ready = (st.session_state.get("raw_df") is not None
                 and "config_pending" not in st.session_state and "config_view_pending" not in st.session_state)
        try:
            data = encode_config(_snapshot_settings()) if ready else b""
        except ValueError as error:
            st.info(f"Fix these settings before saving: {error}")
            data, ready = b"", False
        st.download_button("Save configuration", data=data, file_name="tsplotter-config.json",
                           mime="application/json", key="config_download", disabled=not ready,
                           use_container_width=True)
        if not ready and not st.session_state.get("config_error"):
            if "config_view_pending" in st.session_state:
                st.caption("Choose a usable time axis to apply the remaining chart settings before saving.")
            else:
                st.caption("Choose a dataset and a valid plot setup to save a configuration.")
