from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

TIME_INDEX_OPTION = "(time index)"


@dataclass
class DatetimeIndexCondition:
    logic: str
    low: Optional[dt.datetime] = None
    up: Optional[dt.datetime] = None


@dataclass
class DatetimeColumnCondition:
    logic: str
    column: str
    low: Optional[dt.datetime] = None
    up: Optional[dt.datetime] = None


@dataclass
class NumericRangeCondition:
    logic: str
    column: str
    low: Optional[float] = None
    up: Optional[float] = None


@dataclass
class CategoricalCondition:
    logic: str
    column: str
    op: str
    values: Sequence[str]
    case_sensitive: bool = False
    na_match: bool = False


Condition = Union[
    DatetimeIndexCondition,
    DatetimeColumnCondition,
    NumericRangeCondition,
    CategoricalCondition,
]


def infer_filterable_columns(df: pd.DataFrame, include_index: bool = True) -> dict:
    """Return the inferred filterable columns grouped by semantic type."""
    is_dt_index = include_index and isinstance(df.index, pd.DatetimeIndex)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    dt_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    filter_vars: List[str] = []
    if is_dt_index:
        filter_vars.append(TIME_INDEX_OPTION)
    filter_vars.extend(dt_cols)
    filter_vars.extend(num_cols)
    filter_vars.extend(cat_cols)

    return {
        "is_datetime_index": is_dt_index,
        "datetime": dt_cols,
        "numeric": num_cols,
        "categorical": cat_cols,
        "options": filter_vars,
    }


def _align_to_tz(ts: Optional[dt.datetime], tz) -> Optional[pd.Timestamp]:
    if ts is None:
        return None
    ts = pd.Timestamp(ts)
    if tz is not None:
        return ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)
    else:
        return ts.tz_localize(None) if ts.tzinfo is not None else ts


def _categorical_noop(cond: CategoricalCondition) -> bool:
    op = cond.op
    vals = list(cond.values)
    if op in {"is one of", "is not one of"}:
        return len(vals) == 0
    if op in {"contains", "not contains", "starts with", "ends with", "regex"}:
        return len(vals) == 0 or (len(vals) == 1 and vals[0] == "")
    return False


def build_mask(df: pd.DataFrame, conditions: Iterable[Condition]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    first = True

    for cond in conditions:
        if isinstance(cond, DatetimeIndexCondition):
            if cond.low is None and cond.up is None:
                continue
            series = df.index
            tz = getattr(series, "tz", None)
            low = _align_to_tz(cond.low, tz)
            up = _align_to_tz(cond.up, tz)
            cond_mask = pd.Series(True, index=df.index)
            if low is not None:
                cond_mask &= series >= low
            if up is not None:
                cond_mask &= series <= up

        elif isinstance(cond, DatetimeColumnCondition):
            if cond.low is None and cond.up is None:
                continue
            series = pd.to_datetime(df[cond.column], errors="coerce")
            tz = getattr(series.dt, "tz", None)
            low = _align_to_tz(cond.low, tz)
            up = _align_to_tz(cond.up, tz)
            cond_mask = pd.Series(True, index=df.index)
            if low is not None:
                cond_mask &= series >= low
            if up is not None:
                cond_mask &= series <= up

        elif isinstance(cond, NumericRangeCondition):
            if cond.low is None and cond.up is None:
                continue
            series = df[cond.column]
            if not pd.api.types.is_numeric_dtype(series):
                series = pd.to_numeric(series, errors="coerce")
            cond_mask = pd.Series(True, index=df.index)
            if cond.low is not None:
                cond_mask &= series >= cond.low
            if cond.up is not None:
                cond_mask &= series <= cond.up

        else:  # categorical
            if _categorical_noop(cond):
                continue
            series = df[cond.column].astype("string")
            op = cond.op
            vals = list(cond.values)
            case = cond.case_sensitive
            na_match = cond.na_match
            cond_mask = pd.Series(False, index=df.index)

            if op in {"is one of", "is not one of"}:
                base = series.isin(vals)
                cond_mask = base.fillna(na_match)
                if op == "is not one of":
                    cond_mask = ~cond_mask

            elif op in {"contains", "not contains"}:
                patt = vals[0]
                if not case:
                    series_cmp = series.str.lower()
                    patt = patt.lower()
                else:
                    series_cmp = series
                base = series_cmp.str.contains(patt, regex=False, na=na_match)
                cond_mask = base if op == "contains" else ~base

            elif op in {"starts with", "ends with"}:
                patt = vals[0]
                if not case:
                    series_cmp = series.str.lower()
                    patt = patt.lower()
                else:
                    series_cmp = series
                if op == "starts with":
                    cond_mask = series_cmp.str.startswith(patt, na=na_match)
                else:
                    cond_mask = series_cmp.str.endswith(patt, na=na_match)

            else:  # regex
                patt = vals[0]
                cond_mask = series.str.contains(patt, regex=True, case=case, na=na_match)

            if na_match:
                cond_mask = cond_mask | series.isna()

        if first:
            mask = cond_mask
            first = False
        else:
            if cond.logic == "AND":
                mask = mask & cond_mask
            else:
                mask = mask | cond_mask

    return mask


def _dt_to_iso(value: Optional[dt.datetime]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    return value.isoformat()


def _iso_to_dt(value: Optional[str]) -> Optional[dt.datetime]:
    if value in (None, ""):
        return None
    return pd.Timestamp(value).to_pydatetime()


def condition_to_dict(cond: Condition) -> Dict[str, Any]:
    """Serialize a condition dataclass into a JSON-friendly dict."""
    if isinstance(cond, DatetimeIndexCondition):
        return {
            "type": "datetime_index",
            "logic": cond.logic,
            "low": _dt_to_iso(cond.low),
            "up": _dt_to_iso(cond.up),
        }
    if isinstance(cond, DatetimeColumnCondition):
        return {
            "type": "datetime_column",
            "logic": cond.logic,
            "column": cond.column,
            "low": _dt_to_iso(cond.low),
            "up": _dt_to_iso(cond.up),
        }
    if isinstance(cond, NumericRangeCondition):
        return {
            "type": "numeric_range",
            "logic": cond.logic,
            "column": cond.column,
            "low": float(cond.low) if cond.low is not None else None,
            "up": float(cond.up) if cond.up is not None else None,
        }
    if isinstance(cond, CategoricalCondition):
        return {
            "type": "categorical",
            "logic": cond.logic,
            "column": cond.column,
            "op": cond.op,
            "values": list(cond.values),
            "case_sensitive": bool(cond.case_sensitive),
            "na_match": bool(cond.na_match),
        }
    raise TypeError(f"Unsupported condition type: {type(cond)!r}")


def condition_from_dict(data: Dict[str, Any]) -> Condition:
    """Deserialize a condition dictionary back into the appropriate dataclass."""
    ctype = data.get("type")
    logic = data.get("logic", "AND")

    if ctype == "datetime_index":
        return DatetimeIndexCondition(
            logic=logic,
            low=_iso_to_dt(data.get("low")),
            up=_iso_to_dt(data.get("up")),
        )
    if ctype == "datetime_column":
        return DatetimeColumnCondition(
            logic=logic,
            column=data["column"],
            low=_iso_to_dt(data.get("low")),
            up=_iso_to_dt(data.get("up")),
        )
    if ctype == "numeric_range":
        low = data.get("low")
        up = data.get("up")
        return NumericRangeCondition(
            logic=logic,
            column=data["column"],
            low=float(low) if low is not None else None,
            up=float(up) if up is not None else None,
        )
    if ctype == "categorical":
        return CategoricalCondition(
            logic=logic,
            column=data["column"],
            op=data.get("op", "is one of"),
            values=list(data.get("values", [])),
            case_sensitive=bool(data.get("case_sensitive", False)),
            na_match=bool(data.get("na_match", False)),
        )
    raise ValueError(f"Unknown condition type: {ctype!r}")
