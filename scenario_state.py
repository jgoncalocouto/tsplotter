"""Utilities for serialising and restoring Streamlit session widget state."""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd


SCENARIO_WIDGET_PREFIXES: Sequence[str] = (
    "overview_",
    "mlc_",
    "hist_",
    "sc_",
    "sc2_",
    "rw_",
    "der_",
)


def json_safe_value(value: Any) -> Any:
    """Convert complex values to JSON-serialisable equivalents."""

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, _dt.datetime)):
        return value.isoformat()
    if isinstance(value, (_dt.date,)):
        return value.isoformat()
    if isinstance(value, (_dt.time,)):
        return value.isoformat()
    if isinstance(value, list):
        return [json_safe_value(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe_value(v) for v in value]
    if isinstance(value, set):
        return [json_safe_value(v) for v in sorted(value)]
    if isinstance(value, dict):
        return {str(k): json_safe_value(v) for k, v in value.items()}
    return value


def collect_widget_state(
    session_state: Mapping[str, Any],
    prefixes: Sequence[str] = SCENARIO_WIDGET_PREFIXES,
) -> Dict[str, Any]:
    """Filter widget entries that should be persisted into a session preset."""

    saved: Dict[str, Any] = {}
    for key, value in session_state.items():
        if any(key.startswith(prefix) for prefix in prefixes):
            saved[key] = json_safe_value(value)
    return saved


def apply_widget_state(
    widget_state: Mapping[str, Any],
    session_state: MutableMapping[str, Any],
) -> None:
    """Populate ``session_state`` with persisted widget values."""

    for key, value in widget_state.items():
        session_state[key] = value

