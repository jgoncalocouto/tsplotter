"""Portable, versioned plot settings without data or Streamlit session internals."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import datetime as dt
import json
import math
from numbers import Integral, Real
import re
from typing import Any


CONFIG_APP = "tsplotter"
CONFIG_VERSION = 1
MAX_CONFIG_BYTES = 256 * 1024
MAX_SETTINGS = 512


@dataclass(frozen=True)
class SettingSpec:
    kind: str
    minimum: int | float | None = None
    maximum: int | float | None = None
    choices: tuple[str, ...] | None = None
    exclusive_minimum: bool = False
    max_length: int = 10000
    max_items: int = 2000


@dataclass(frozen=True)
class LoadedConfig:
    settings: dict[str, Any]
    warnings: list[str]


def _enum(*values: str) -> SettingSpec:
    return SettingSpec("string", choices=values)


TEXT = SettingSpec("string")
BOOLEAN = SettingSpec("bool")
STRINGS = SettingSpec("strings")
DATE = SettingSpec("date")
TIME = SettingSpec("time")
PALETTES = ("Plotly", "D3", "Set1", "Set2", "Set3", "Pastel", "Bold", "Dark24", "Light24")
TIMEZONES = ("UTC", "Europe/Lisbon", "Europe/London", "US/Eastern", "US/Pacific", "Asia/Tokyo")
BIN_MODES = ("Individual values", "Number of bins", "Bin width")

SETTING_SPECS: dict[str, SettingSpec] = {
    "plot_type": _enum("Trends", "Distribution", "Scatter", "Grouped scatter", "Rolling window", "Derivatives"),
    "overview_cols": STRINGS,
    "setup_time_mode": _enum("Absolute time", "Relative time"),
    "setup_index": TEXT,
    "setup_dt_col": TEXT,
    "setup_profile": _enum("Auto-detect", "DMY (dd/mm/yyyy)", "MDY (mm/dd/yyyy)", "YMD (yyyy-mm-dd)",
                           "Unix epoch (seconds)", "Unix epoch (milliseconds)", "Custom strptime"),
    "setup_format": TEXT,
    "setup_tz_mode": _enum("Treat as naive (no tz)", "Assume timezone and localize", "Input already has timezone"),
    "setup_assume_tz": _enum(*TIMEZONES, "Custom..."),
    "setup_out_tz": _enum("(keep)", *TIMEZONES, "Custom..."),
    "setup_custom_tz": TEXT,
    "setup_target_tz": TEXT,
    "mlc_n_subplots": SettingSpec("int", 1, 6),
    "mlc_shape": _enum("linear", "spline", "hv", "vh", "hvh", "vhv"),
    "mlc_width": SettingSpec("int", 1, 8),
    "mlc_ytype": _enum("linear", "log"),
    "mlc_maxpts": SettingSpec("int", 1000, 100000),
    "hist_cols_main": STRINGS,
    "hist_binmode": _enum("Auto (width)", "Fixed width", "Target #bins"),
    "hist_width_input": SettingSpec("number", 1e-12),
    "hist_bins_main": SettingSpec("int", 10, 200),
    "sc_x": TEXT,
    "sc_y": TEXT,
    "sc_group": TEXT,
    "sc_cmap_cont": _enum("Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo", "Blues", "Greens", "Greys", "Oranges", "Purples", "Reds"),
    "sc_cmap_disc": _enum(*PALETTES),
    "sc2_x": TEXT,
    "sc2_y": TEXT,
    "sc2_g1": TEXT,
    "sc2_g2": TEXT,
    "sc2_palette": _enum(*PALETTES),
    "sc2_marker_preset": _enum("Core Shapes", "Crosshair Set", "Directional", "Starburst", "Minimal Dots"),
    "sc2_marker_custom": TEXT,
    "rw_y": TEXT,
    "rw_window": SettingSpec("int", 2, 200000),
    "rw_msize": SettingSpec("int", 3, 16),
    "rw_mop": SettingSpec("number", .2, 1),
    "rw_maxpts": SettingSpec("int", 1000, 200000),
    "der_y": TEXT,
    "der_tunit": _enum("seconds", "minutes", "hours", "days", "none (sample index)"),
    "der_xmode": _enum("index step (1)", "use numeric index values"),
    "der_lw": SettingSpec("int", 1, 8),
    "der_smooth": SettingSpec("int", 1, 9999),
    "der_maxpts": SettingSpec("int", 1000, 200000),
    "flt2_n": SettingSpec("int", 0, 10),
}
for _key in (
    "mlc_markers", "mlc_rangeslider", "mlc_unified", "mlc_spikes",
    "rw_include_current", "rw_full", "rw_band", "rw_unified", "rw_spikes", "rw_rangeslider",
    "der_markers", "der_unified", "der_rangeslider", "der_spikes",
):
    SETTING_SPECS[_key] = BOOLEAN
for _prefix in ("sc", "sc2"):
    SETTING_SPECS[f"{_prefix}_maxpts"] = SettingSpec("int", 1000, 200000)
    SETTING_SPECS[f"{_prefix}_msize"] = SettingSpec("int", 3, 20)
    SETTING_SPECS[f"{_prefix}_mop"] = SettingSpec("number", .2, 1)
for _prefix in ("sc_color", "sc2_color", "sc2_marker"):
    SETTING_SPECS[f"{_prefix}_bin_mode"] = _enum(*BIN_MODES)
    SETTING_SPECS[f"{_prefix}_bin_count"] = SettingSpec("int", 1, 100)
    SETTING_SPECS[f"{_prefix}_bin_width"] = SettingSpec("number", 0, exclusive_minimum=True)

_DYNAMIC_SPECS = (
    (re.compile(r"mlc_cols_[0-5]"), STRINGS),
    (re.compile(r"flt2_logic_[0-9]"), _enum("AND", "OR")),
    (re.compile(r"flt2_var_[0-9]"), TEXT),
    (re.compile(r"flt2_use_(?:low|up)dt_(?:idx|col)_[0-9]"), BOOLEAN),
    (re.compile(r"flt2_(?:low|up)_date_(?:idx|col)_[0-9]"), DATE),
    (re.compile(r"flt2_(?:low|up)_time_(?:idx|col)_[0-9]"), TIME),
    (re.compile(r"flt2_(?:low|up)_num_[0-9]"), TEXT),
    (re.compile(r"flt2_op_[0-9]"), _enum("is one of", "is not one of", "contains", "not contains", "starts with", "ends with", "regex")),
    (re.compile(r"flt2_vals_[0-9]"), STRINGS),
    (re.compile(r"flt2_(?:vals_csv|patt)_[0-9]"), TEXT),
    (re.compile(r"flt2_(?:case|na)_[0-9]"), BOOLEAN),
)


def setting_spec(key: str) -> SettingSpec | None:
    """Return a setting schema only for supported, writable configuration keys."""
    if not isinstance(key, str):
        return None
    if key in SETTING_SPECS:
        return SETTING_SPECS[key]
    return next((spec for pattern, spec in _DYNAMIC_SPECS if pattern.fullmatch(key)), None)


def is_config_key(key: str) -> bool:
    return setting_spec(key) is not None


def _validate_value(key: str, value: Any, spec: SettingSpec) -> Any:
    error = f"Invalid setting '{key}'"
    if spec.kind == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{error}: expected true or false.")
        return value
    if spec.kind in {"int", "number"}:
        expected = Integral if spec.kind == "int" else Real
        if isinstance(value, bool) or not isinstance(value, expected):
            raise ValueError(f"{error}: expected a {'whole number' if spec.kind == 'int' else 'number'}.")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"{error}: numbers must be finite.")
        if spec.minimum is not None and (value < spec.minimum or (spec.exclusive_minimum and value == spec.minimum)):
            comparison = "greater than" if spec.exclusive_minimum else "at least"
            raise ValueError(f"{error}: use a value {comparison} {spec.minimum}.")
        if spec.maximum is not None and value > spec.maximum:
            raise ValueError(f"{error}: use a value no greater than {spec.maximum}.")
        return int(value) if spec.kind == "int" else float(value)
    if spec.kind == "strings":
        if not isinstance(value, list) or len(value) > spec.max_items:
            raise ValueError(f"{error}: expected a list of at most {spec.max_items} names or values.")
        return [_validate_value(key, item, TEXT) for item in value]
    if spec.kind == "string":
        if not isinstance(value, str) or len(value) > spec.max_length:
            raise ValueError(f"{error}: expected text of at most {spec.max_length} characters.")
        if spec.choices is not None and value not in spec.choices:
            raise ValueError(f"{error}: choose one of {', '.join(spec.choices)}.")
        return value
    if spec.kind == "date":
        if type(value) is dt.date:
            return value
        if isinstance(value, str):
            try:
                return dt.date.fromisoformat(value)
            except ValueError:
                pass
        raise ValueError(f"{error}: expected an ISO date such as 2025-01-31.")
    if spec.kind == "time":
        if isinstance(value, str):
            try:
                value = dt.time.fromisoformat(value)
            except ValueError:
                pass
        if isinstance(value, dt.time) and value.tzinfo is None:
            return value
        raise ValueError(f"{error}: expected a local ISO time such as 14:30:00.")
    raise ValueError(f"{error}: unsupported setting type.")


def validate_settings(settings: Mapping[str, Any]) -> LoadedConfig:
    """Validate all known settings without modifying the supplied mapping."""
    if not isinstance(settings, Mapping):
        raise ValueError("Configuration settings must be a JSON object.")
    if len(settings) > MAX_SETTINGS:
        raise ValueError(f"A configuration can contain at most {MAX_SETTINGS} settings.")
    clean: dict[str, Any] = {}
    warnings = []
    for key, value in settings.items():
        spec = setting_spec(key)
        if spec is None:
            warnings.append(f"Ignored unsupported setting: {str(key)[:100]}.")
            continue
        clean[key] = _validate_value(key, value, spec)
    return LoadedConfig(clean, warnings)


def encode_config(state: Mapping[str, Any]) -> bytes:
    """Export allowlisted settings, excluding datasets and widget trigger state."""
    # Filter before validation so unrelated session values are never serialized.
    settings = {key: state[key] for key in state if is_config_key(key)}
    clean = validate_settings(settings).settings
    document = {"app": CONFIG_APP, "version": CONFIG_VERSION, "settings": clean}
    def encode_special(value):
        if type(value) is dt.date or isinstance(value, dt.time):
            return value.isoformat()
        raise TypeError("Configuration contains an unsupported value.")
    data = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False,
                      default=encode_special).encode("utf-8")
    if len(data) > MAX_CONFIG_BYTES:
        raise ValueError("This configuration is too large. Save fewer selected values.")
    return data


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate configuration field: {key[:100]}.")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Configuration numbers must be finite.")


def _json_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Configuration numbers must be finite.")
    return number


def decode_config(data: bytes | str) -> LoadedConfig:
    """Read a validated settings file; never evaluate code or restore file/data state."""
    if not isinstance(data, (bytes, str)):
        raise ValueError("Choose a JSON configuration file.")
    try:
        raw = data.encode("utf-8") if isinstance(data, str) else data
    except UnicodeError as error:
        raise ValueError("The configuration must use UTF-8 text.") from error
    if len(raw) > MAX_CONFIG_BYTES:
        raise ValueError("This configuration is too large. The limit is 256 KB.")
    try:
        document = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_json_object,
                              parse_constant=_reject_constant, parse_float=_json_float)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("This is not a valid UTF-8 JSON configuration file.") from error
    if not isinstance(document, dict):
        raise ValueError("A configuration must be a JSON object.")
    if document.get("app") != CONFIG_APP:
        raise ValueError("This file is not a TSPlotter configuration.")
    version = document.get("version")
    if type(version) is not int or version != CONFIG_VERSION:
        raise ValueError(f"Unsupported configuration version. This app supports version {CONFIG_VERSION}.")
    if "settings" not in document:
        raise ValueError("The configuration is missing its settings object.")
    return validate_settings(document["settings"])
