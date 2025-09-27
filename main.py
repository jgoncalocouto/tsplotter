# -*- coding: utf-8 -*-
# Time-Series Plot Helper (Streamlit + Plotly) — 2025-09-10
# Run with: streamlit run main_tsplotter.py
from __future__ import annotations
import io
import json
import os
import tempfile
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import datetime as _dt

from matloader_v73 import read_mat_v73

# Plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.validators.scatter.marker import SymbolValidator
except Exception:
    px = None
    go = None
    SymbolValidator = None

ROW_INDEX_OPTION = "(row number 0..N-1)"
TIME_INDEX_OPTION = "(use existing time index)"

# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(page_title="Time-Series Plot Helper", layout="wide")
# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="display:flex;align-items:center;gap:.5rem;margin:0">
  🕒 Time-Series Plot Helper
</h1>
<p style="color:#6b7280;margin:.25rem 0 0">
  Upload a dataset, parse time (<em>absolute</em> or <em>relative</em>), filter rows, and explore via
  multi-subplot line charts, histograms, and grouped scatters.
</p>
""", unsafe_allow_html=True)

with st.expander("Quick start", expanded=False):
    st.markdown("""
1) **Upload** CSV/TSV, Excel, Parquet, JSON, or MATLAB .mat (v7.3)
2) Pick **Time mode**: Absolute (timestamps) or Relative (row/column index)  
3) Apply **Filters** (date & numeric ranges)  
4) Explore **Plots** and **Download** the data behind each figure
""")



# ----------------------------------
# Cached readers / builders
# ----------------------------------

@st.cache_data(show_spinner=False)
def _read_table_from_bytes(
    name: str, data: bytes, sheet: Optional[str]
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    lname = name.lower()
    bio = io.BytesIO(data)

    if lname.endswith(".mat"):
        tmp = tempfile.NamedTemporaryFile(suffix=".mat", delete=False)
        try:
            tmp.write(data)
            tmp.flush()
        finally:
            tmp.close()
        try:
            df, meta = read_mat_v73(tmp.name)
            extra: Dict[str, Any] = {"format": "mat_v73", "source_name": name}
            extra.update(meta)
            return df, extra
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    if lname.endswith((".csv", ".tsv")):
        # Multiple separators and encodings to be robust to Windows exports
        seps = [",", ";", "\t", "|"]
        encs = ["utf-8", "utf-8-sig", "cp1252", "latin1", "utf-16"]
        last_err = None
        for sep in seps:
            for enc in encs:
                bio.seek(0)
                try:
                    return (
                        pd.read_csv(
                            bio,
                            sep=sep if lname.endswith(".csv") else ("\t" if sep == "\t" else sep),
                            encoding=enc,
                        ),
                        None,
                    )
                except Exception as e:
                    last_err = e
        raise RuntimeError(f"CSV/TSV parse failed. Last error: {last_err}")

    if lname.endswith(".parquet"):
        bio.seek(0)
        return pd.read_parquet(bio), None

    if lname.endswith((".xls", ".xlsx")):
        bio.seek(0)
        df = pd.read_excel(bio, sheet_name=sheet) if sheet else pd.read_excel(bio)
        return df, None

    if lname.endswith(".json"):
        bio.seek(0)
        try:
            return pd.read_json(bio, lines=True), None
        except Exception:
            bio.seek(0)
            obj = json.load(bio)
            try:
                return pd.json_normalize(obj), None
            except Exception:
                return pd.DataFrame(obj), None

    raise ValueError("Unsupported file type. Use CSV, Excel, Parquet, JSON, or MATLAB .mat (v7.3).")

def add_buy_me_coffee(url: str):
    html = f"""
    <style>
      /* Give the page some bottom padding so the button doesn't overlap content */
      .block-container {{ padding-bottom: 160px; }}

      /* Floating circular button */
      #fixed-coffee {{
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 9999;
      }}
      #fixed-coffee a {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 72px;
        height: 72px;
        border-radius: 999px;
        background: #ff813f;
        color: #ffffff !important;
        text-decoration: none;
        box-shadow: 0 8px 20px rgba(0,0,0,.25);
        font-size: 36px;
        line-height: 1;
        transition: transform .15s ease, box-shadow .15s ease, opacity .15s ease;
        opacity: 0.95;
      }}
      #fixed-coffee a:hover {{
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(0,0,0,.28);
        opacity: 1;
      }}

      /* Smaller on mobile + move up so it clears Streamlit's bottom toolbar */
      @media (max-width: 640px) {{
        #fixed-coffee a {{ width: 56px; height: 56px; font-size: 28px; }}
        #fixed-coffee {{ bottom: 88px; right: 16px; }}
      }}
    </style>

    <div id="fixed-coffee">
      <a href="{url}" target="_blank" rel="noopener noreferrer" title="Buy me a coffee" aria-label="Buy me a coffee">☕</a>
    </div>
    """
    import streamlit as st
    st.markdown(html, unsafe_allow_html=True)


def _align_to_index_tz(ts, index_dtindex):
    """Return a pandas.Timestamp aligned to the index tz (or naive if index is naive)."""
    if ts is None:
        return None
    ts = pd.Timestamp(ts)
    idx_tz = getattr(index_dtindex, "tz", None)
    if idx_tz is not None:
        # Index is tz-aware → localize naive bounds or convert aware bounds
        return ts.tz_localize(idx_tz) if ts.tzinfo is None else ts.tz_convert(idx_tz)
    else:
        # Index is naive → drop tz from aware bounds (keep wall time)
        return ts.tz_localize(None) if ts.tzinfo is not None else ts

def _guess_datetime_col(df: pd.DataFrame) -> Optional[str]:
    cands = [c for c in df.columns if any(k in c.lower() for k in ("date", "time", "timestamp", "datetime", "ds"))]
    if cands:
        return cands[0]
    first = df.columns[0]
    try:
        pd.to_datetime(df[first], errors="raise")
        return first
    except Exception:
        return None


def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    return df


def _maybe_downsample_for_plot(df: pd.DataFrame, max_points: int = 30_000) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    stride = max(1, len(df) // max_points)
    return df.iloc[::stride]


def _parse_dt_series(
    series: pd.Series,
    parse_profile: str,
    custom_fmt: Optional[str],
    tz_mode: str,
    assume_tz: Optional[str],
    out_tz: Optional[str],
) -> pd.Series:
    s = series
    if parse_profile == "Unix epoch (seconds)":
        dt = pd.to_datetime(s, errors="coerce", unit="s", utc=False)
    elif parse_profile == "Unix epoch (milliseconds)":
        dt = pd.to_datetime(s, errors="coerce", unit="ms", utc=False)
    elif parse_profile == "DMY (dd/mm/yyyy)":
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    elif parse_profile == "MDY (mm/dd/yyyy)":
        dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
    elif parse_profile == "YMD (yyyy-mm-dd)":
        dt = pd.to_datetime(s, errors="coerce")
    elif parse_profile == "Custom strptime":
        dt = pd.to_datetime(s, errors="coerce", format=custom_fmt)
    else:  # Auto
        dt = pd.to_datetime(s, errors="coerce")

    if tz_mode == "Assume timezone and localize" and assume_tz:
        try:
            dt = dt.dt.tz_localize(assume_tz, nonexistent="shift_forward", ambiguous="NaT")
        except Exception:
            pass

    if out_tz and out_tz != "(keep)":
        try:
            if dt.dt.tz is None:
                dt = dt.dt.tz_localize("UTC").dt.tz_convert(out_tz)
            else:
                dt = dt.dt.tz_convert(out_tz)
        except Exception:
            pass
    return dt


@st.cache_data(show_spinner=False)
def _build_prepared_df(
    raw_df: pd.DataFrame,
    time_mode: str,
    index_col: Optional[str],
    dt_col: Optional[str],
    parse_profile: Optional[str],
    custom_fmt: Optional[str],
    tz_mode: Optional[str],
    assume_tz: Optional[str],
    out_tz: Optional[str],
) -> pd.DataFrame:
    if time_mode == "Relative time":
        tmp = raw_df.copy()
        if index_col is None:
            raise ValueError("Relative mode requires an index choice.")
        if index_col == ROW_INDEX_OPTION:
            tmp.index = np.arange(len(tmp), dtype=float)  # 0..N-1
            tmp.index.name = "row"
            return _downcast_numeric(tmp)
        coerced = pd.to_numeric(tmp[index_col], errors="coerce")
        if coerced.notna().sum() >= int(0.9 * len(coerced)):
            tmp[index_col] = coerced.fillna(method="ffill").fillna(0)
        tmp = tmp.set_index(index_col)
        if not tmp.index.is_monotonic_increasing:
            try:
                tmp = tmp.sort_index()
            except Exception:
                pass
        tmp.index.name = index_col
        return _downcast_numeric(tmp)

    if not dt_col:
        raise ValueError("Absolute mode requires a datetime column.")
    if dt_col == TIME_INDEX_OPTION:
        tmp = raw_df.copy()
        if not isinstance(tmp.index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
            tmp.index = pd.to_datetime(tmp.index, errors="coerce")
        if hasattr(tmp.index, "notna"):
            mask = tmp.index.notna()
            tmp = tmp.loc[mask]
        try:
            tmp = tmp.sort_index()
        except Exception:
            pass
        tmp.index.name = raw_df.index.name or "Time"
        return _downcast_numeric(tmp)
    tmp = raw_df.copy()
    tmp[dt_col] = _parse_dt_series(tmp[dt_col], parse_profile or "Auto-detect", custom_fmt, tz_mode or "", assume_tz, out_tz)
    tmp = tmp.dropna(subset=[dt_col])
    tmp = tmp.set_index(dt_col)
    try:
        tmp = tmp.sort_index()
    except Exception:
        pass
    tmp.index.name = dt_col
    return _downcast_numeric(tmp)

# ----------------------------------
# Session state
# ----------------------------------
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "upload_meta" not in st.session_state:
    st.session_state.upload_meta = None

# ----------------------------------
# Sidebar - Upload (cached)
# ----------------------------------
st.sidebar.header("📥 1) Upload")
up = st.sidebar.file_uploader(
    "Upload data file",
    type=["csv", "tsv", "xls", "xlsx", "parquet", "json", "mat"],
    accept_multiple_files=False,
)

sheet = None
if up and up.name.lower().endswith((".xls", ".xlsx")):
    sheet = st.sidebar.text_input("Excel sheet (optional)") or None

if up:
    try:
        name = up.name
        data = up.getvalue()
        df, meta = _read_table_from_bytes(name, data, sheet)
        st.session_state.raw_df = df
        st.session_state.upload_meta = meta
    except Exception as e:
        st.session_state.upload_meta = None
        st.sidebar.error(f"Read failed: {e}")

raw_df = st.session_state.raw_df
upload_meta = st.session_state.upload_meta
if raw_df is None:
    st.stop()

if upload_meta and upload_meta.get("format") == "mat_v73":
    lines = []
    source_name = upload_meta.get("source_name")
    if source_name:
        lines.append(f"File: `{source_name}`")
    kind = upload_meta.get("kind")
    if kind:
        lines.append(f"Layout: `{kind}`")
    n_cols = upload_meta.get("n_cols")
    if n_cols is None:
        n_cols = upload_meta.get("cols")
    if n_cols is not None:
        try:
            n_cols_int = int(n_cols)
        except Exception:
            n_cols_int = n_cols
        lines.append(f"Columns detected: **{n_cols_int}**")
    time_ds = upload_meta.get("time_dataset")
    if time_ds:
        unit = upload_meta.get("time_unit")
        if unit:
            lines.append(f"Time vector: `{time_ds}` ({unit})")
        else:
            lines.append(f"Time vector: `{time_ds}`")
    index_name = upload_meta.get("index_name")
    if index_name:
        lines.append(f"Index label: `{index_name}`")
    h5path = upload_meta.get("h5path")
    if h5path:
        lines.append(f"Source group: `{h5path}`")
    rows = upload_meta.get("rows")
    if rows is not None:
        try:
            rows_int = int(rows)
        except Exception:
            rows_int = rows
        lines.append(f"Rows detected: **{rows_int}**")

    info_md = "**MAT v7.3 import**"
    if lines:
        info_md += "\n" + "\n".join(f"- {line}" for line in lines)
    st.sidebar.info(info_md)


# ----------------------------------
# Sidebar - Time mode & parsing
# ----------------------------------
st.sidebar.header("🕒 2) Time mode & parsing")
time_mode = st.sidebar.selectbox("Time mode", options=["Absolute time", "Relative time"], index=0)

abs_controls_active = (time_mode == "Absolute time")
rel_controls_active = (time_mode == "Relative time")

index_col = None
dt_col = None
parse_profile = None
custom_fmt = None
tz_mode = None
assume_tz = None
out_tz = None

if rel_controls_active:
    index_choices = [ROW_INDEX_OPTION] + list(raw_df.columns)
    index_col = st.sidebar.selectbox(
        "Index (x-axis) column",
        options=index_choices,
        help="Pick a column to be the x-axis, or choose row number 0..N-1.",
    )

if abs_controls_active:
    _guess = _guess_datetime_col(raw_df) or ""
    _dt_opts = [""]
    if isinstance(raw_df.index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        _dt_opts.append(TIME_INDEX_OPTION)
    _dt_opts.extend(list(raw_df.columns))
    if _guess and _guess in raw_df.columns:
        _dt_idx = _dt_opts.index(_guess)
    elif TIME_INDEX_OPTION in _dt_opts:
        _dt_idx = _dt_opts.index(TIME_INDEX_OPTION)
    else:
        _dt_idx = 0

    dt_col = st.sidebar.selectbox(
        "Datetime column",
        options=_dt_opts,
        index=_dt_idx,
        help="Column to interpret as time.",
    )

    parse_profile = st.sidebar.selectbox(
        "Input layout",
        options=[
            "Auto-detect",
            "DMY (dd/mm/yyyy)",
            "MDY (mm/dd/yyyy)",
            "YMD (yyyy-mm-dd)",
            "Unix epoch (seconds)",
            "Unix epoch (milliseconds)",
            "Custom strptime",
        ],
        index=0,
    )
    if parse_profile == "Custom strptime":
        custom_fmt = st.sidebar.text_input("strptime format", value="%Y-%m-%d %H:%M:%S") or None

    tz_mode = st.sidebar.selectbox(
        "Timezone handling",
        options=[
            "Treat as naive (no tz)",
            "Assume timezone and localize",
            "Input already has timezone",
        ],
        index=0,
    )
    _common_tzs = ["UTC", "Europe/Lisbon", "Europe/London", "US/Eastern", "US/Pacific", "Asia/Tokyo"]
    if tz_mode == "Assume timezone and localize":
        assume_tz = st.sidebar.selectbox("Assumed tz", options=_common_tzs + ["Custom..."], index=1)
        if assume_tz == "Custom...":
            assume_tz = st.sidebar.text_input("IANA tz (e.g., Europe/Paris)") or None

    out_tz = st.sidebar.selectbox(
        "Convert output to",
        options=["(keep)"] + _common_tzs + ["Custom..."],
        index=1,
    )
    if out_tz == "Custom...":
        out_tz = st.sidebar.text_input("Target tz (IANA)") or None

# Guard: Absolute time requires dt column
if time_mode == "Absolute time" and not dt_col:
    st.warning("Pick a Datetime column in the sidebar to continue with Absolute time.")
    st.stop()

# ----------------------------------
# Build prepared dataframe (CACHED)
# ----------------------------------
try:
    prepared_df = _build_prepared_df(
        raw_df=raw_df,
        time_mode=time_mode,
        index_col=index_col,
        dt_col=dt_col,
        parse_profile=parse_profile,
        custom_fmt=custom_fmt,
        tz_mode=tz_mode,
        assume_tz=assume_tz,
        out_tz=out_tz,
    )
except Exception as e:
    st.error(str(e))
    st.stop()

st.sidebar.success(f"Parsed {len(prepared_df):,} rows. Index: {prepared_df.index.name}")


# ----------------------------------
#  FILTERS 
# ----------------------------------
st.header("Filters")
_is_dt_index = isinstance(prepared_df.index, pd.DatetimeIndex)
_TIME_VAR_OPTION = "(time index)"

# Detect columns by type
_num_cols = prepared_df.select_dtypes(include=[np.number]).columns.tolist()
_dt_cols = prepared_df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
_cat_cols = prepared_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# Build variable list: (time index?) + datetime columns + numeric + categorical/bool
_filter_vars = ([_TIME_VAR_OPTION] if _is_dt_index else []) + _dt_cols + _num_cols + _cat_cols

# Default year for date pickers: latest from time index if present, else current year
if _is_dt_index and len(prepared_df.index) > 0:
    _default_year = int(pd.Timestamp(prepared_df.index.max()).year)
else:
    _default_year = _dt.date.today().year

# ——— helpers ———
def _align_to_tz(ts, tz):
    """Align a bound to a given tz (or strip tz if tz is None)."""
    if ts is None:
        return None
    ts = pd.Timestamp(ts)
    if tz is not None:
        return ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)
    else:
        return ts.tz_localize(None) if ts.tzinfo is not None else ts

def _parse_csv_list(s: str):
    vals = [v.strip() for v in s.split(",") if v.strip()]
    return vals

# UI
n_filters = st.number_input(
    "Number of filtering conditions",
    min_value=0, max_value=10, value=0, step=1, key="flt2_n",
    help="Each condition matches rows; you can chain with AND/OR. Empty numeric bounds mean −∞ / +∞."
)

_conditions = []
for i in range(int(n_filters)):
    with st.expander(f"Condition {i+1}", expanded=True):
        logic_op = st.selectbox("Logical operator vs previous", ["AND", "OR"], index=0, key=f"flt2_logic_{i}")
        var_i = st.selectbox("Variable", options=_filter_vars, index=0, key=f"flt2_var_{i}")

        # Decide branch by variable type
        if var_i == _TIME_VAR_OPTION:
            # Datetime bounds for the index
            use_low = st.checkbox("Set lower datetime bound (≥)", value=False, key=f"flt2_use_lowdt_idx_{i}")
            if use_low:
                low_date = st.date_input("Lower date", value=_dt.date(_default_year, 1, 1), key=f"flt2_low_date_idx_{i}")
                low_time = st.time_input("Lower time", value=_dt.time(0, 0), key=f"flt2_low_time_idx_{i}")
                low_val = _dt.datetime.combine(low_date, low_time)
            else:
                low_val = None

            use_up = st.checkbox("Set upper datetime bound (≤)", value=False, key=f"flt2_use_updt_idx_{i}")
            if use_up:
                up_date = st.date_input("Upper date", value=_dt.date(_default_year, 12, 31), key=f"flt2_up_date_idx_{i}")
                up_time = st.time_input("Upper time", value=_dt.time(23, 59, 59), key=f"flt2_up_time_idx_{i}")
                up_val = _dt.datetime.combine(up_date, up_time)
            else:
                up_val = None

            _conditions.append({"logic": logic_op, "kind": "dt_index", "low": low_val, "up": up_val})

        elif var_i in _dt_cols:
            # Datetime bounds for a datetime column
            # Default year from this column if possible
            col = prepared_df[var_i]
            try:
                _col_year = int(pd.to_datetime(col, errors="coerce").max().year)
            except Exception:
                _col_year = _default_year

            use_low = st.checkbox("Set lower datetime bound (≥)", value=False, key=f"flt2_use_lowdt_col_{i}")
            if use_low:
                low_date = st.date_input("Lower date", value=_dt.date(_col_year, 1, 1), key=f"flt2_low_date_col_{i}")
                low_time = st.time_input("Lower time", value=_dt.time(0, 0), key=f"flt2_low_time_col_{i}")
                low_val = _dt.datetime.combine(low_date, low_time)
            else:
                low_val = None

            use_up = st.checkbox("Set upper datetime bound (≤)", value=False, key=f"flt2_use_updt_col_{i}")
            if use_up:
                up_date = st.date_input("Upper date", value=_dt.date(_col_year, 12, 31), key=f"flt2_up_date_col_{i}")
                up_time = st.time_input("Upper time", value=_dt.time(23, 59, 59), key=f"flt2_up_time_col_{i}")
                up_val = _dt.datetime.combine(up_date, up_time)
            else:
                up_val = None

            _conditions.append({"logic": logic_op, "kind": "dt_col", "var": var_i, "low": low_val, "up": up_val})

        elif var_i in _num_cols:
            # Numeric range
            low_txt = st.text_input("Lower bound (≥) — empty = -inf", key=f"flt2_low_num_{i}")
            up_txt  = st.text_input("Upper bound (≤) — empty = +inf", key=f"flt2_up_num_{i}")

            def _f_or_none(s):
                s = s.strip()
                if not s: return None
                try: return float(s)
                except Exception: return None

            _conditions.append({"logic": logic_op, "kind": "num", "var": var_i, "low": _f_or_none(low_txt), "up": _f_or_none(up_txt)})

        else:
            # Categorical / text / bool
            op = st.selectbox(
                "Operator",
                ["is one of", "is not one of", "contains", "not contains", "starts with", "ends with", "regex"],
                index=0, key=f"flt2_op_cat_{i}"
            )
            case_sens = st.checkbox("Case sensitive", value=False, key=f"flt2_case_{i}")
            na_match = st.checkbox("Treat NA as match", value=False, key=f"flt2_na_{i}")

            # If low-cardinality, show multiselect; else CSV text
            ser_str = prepared_df[var_i].astype(str)
            uniq = pd.unique(ser_str.dropna())[:2000]  # cap
            if op in ["is one of", "is not one of"] and len(uniq) <= 100:
                vals = st.multiselect("Values", options=sorted(map(str, uniq)), key=f"flt2_vals_{i}")
            elif op in ["is one of", "is not one of"]:
                csv = st.text_input("Comma-separated values", key=f"flt2_vals_csv_{i}")
                vals = _parse_csv_list(csv)
            else:
                patt = st.text_input("Pattern / text", key=f"flt2_patt_{i}")
                vals = [patt]

            _conditions.append({
                "logic": logic_op, "kind": "cat", "var": var_i,
                "op": op, "vals": vals, "case": case_sens, "na_match": na_match
            })

# Build mask
_mask = pd.Series(True, index=prepared_df.index)
_first = True

for cond in _conditions:
    # skip empty/no-op
    if cond["kind"] in ("dt_index", "dt_col"):
        if cond.get("low") is None and cond.get("up") is None:
            continue
    elif cond["kind"] == "num":
        if cond.get("low") is None and cond.get("up") is None:
            continue
    elif cond["kind"] == "cat":
        if cond.get("op") in ["is one of", "is not one of"] and not cond.get("vals"):
            continue
        if cond.get("op") in ["contains", "not contains", "starts with", "ends with", "regex"] and not cond.get("vals", [""])[0]:
            continue

    # per-kind masks
    if cond["kind"] == "dt_index":
        series = prepared_df.index
        tz = series.tz
        low = _align_to_tz(cond["low"], tz)
        up  = _align_to_tz(cond["up"], tz)
        c_mask = pd.Series(True, index=prepared_df.index)
        if low is not None: c_mask &= (series >= low)
        if up  is not None: c_mask &= (series <= up)

    elif cond["kind"] == "dt_col":
        series = pd.to_datetime(prepared_df[cond["var"]], errors="coerce")
        tz = getattr(series.dt, "tz", None)
        low = _align_to_tz(cond["low"], tz)
        up  = _align_to_tz(cond["up"], tz)
        c_mask = pd.Series(True, index=prepared_df.index)
        if low is not None: c_mask &= (series >= low)
        if up  is not None: c_mask &= (series <= up)

    elif cond["kind"] == "num":
        series = prepared_df[cond["var"]]
        if not pd.api.types.is_numeric_dtype(series):
            series = pd.to_numeric(series, errors="coerce")
        c_mask = pd.Series(True, index=prepared_df.index)
        low, up = cond["low"], cond["up"]
        if low is not None: c_mask &= (series >= low)
        if up  is not None: c_mask &= (series <= up)

    else:  # "cat"
        series = prepared_df[cond["var"]].astype(str)
        op = cond["op"]; vals = cond["vals"]; case = cond["case"]; na_match = cond["na_match"]
        c_mask = pd.Series(False, index=prepared_df.index)

        if op in ["is one of", "is not one of"]:
            base = series.isin(vals)
            c_mask = base.fillna(na_match)
            if op == "is not one of":
                c_mask = ~c_mask

        elif op in ["contains", "not contains"]:
            patt = vals[0]
            if not case:
                series_cmp = series.str.lower()
                patt = patt.lower()
            else:
                series_cmp = series
            base = series_cmp.str.contains(patt, regex=False, na=na_match)
            c_mask = base if op == "contains" else ~base

        elif op in ["starts with", "ends with"]:
            patt = vals[0]
            if not case:
                series_cmp = series.str.lower(); patt = patt.lower()
            else:
                series_cmp = series
            base = series_cmp.str.startswith(patt, na=na_match) if op == "starts with" else series_cmp.str.endswith(patt, na=na_match)
            c_mask = base

        else:  # regex
            patt = vals[0]
            base = series.str.contains(patt, regex=True, case=case, na=na_match)
            c_mask = base

    # chain
    if _first:
        _mask = c_mask
        _first = False
    else:
        _mask = (_mask & c_mask) if (cond["logic"] == "AND") else (_mask | c_mask)

filtered_df = prepared_df.loc[_mask].copy()
_removed = len(prepared_df) - len(filtered_df)
st.caption(f"Filters applied. Rows kept: **{len(filtered_df):,}** (removed {max(_removed,0):,}).")

# downstream data
work = filtered_df.dropna(how="all").copy()



# ----------------------------------
# Overview
# ----------------------------------
st.header("📊 Overview")
st.caption("Quick stats and a preview so you know what you’re plotting.")

with st.container(border=True):
    
    _overview_numeric = work.select_dtypes(include=[np.number]).columns.tolist()
    _default_overview = _overview_numeric[: min(6, len(_overview_numeric))] if _overview_numeric else []
    
    with st.expander("Summary Statistics",expanded=True):
    
        ov_cols = st.multiselect(
            "Columns for summary",
            options=_overview_numeric,
            default=_default_overview,
            key="overview_cols",
            help="Pick the variables to summarize below."
        )
        
        if ov_cols:
            summary_df = pd.DataFrame({
                "min": work[ov_cols].min(),
                "max": work[ov_cols].max(),
                "mean": work[ov_cols].mean(),
                "median": work[ov_cols].median(),
                "std": work[ov_cols].std(),
                "count": work[ov_cols].count(),
                "count_zeros": (work[ov_cols] == 0).sum(),
            })
            st.dataframe(summary_df)
        else:
            st.info("Select at least one numeric column for the summary table.")
            
    
    with st.expander("Full Dataframe",expanded=True):
        st.dataframe(prepared_df)

# ----------------------------------
# Plots
# ----------------------------------
st.header("📈 Plots")

if px is None or go is None:
    st.error("Plotly is required. Install with: `pip install plotly`")
else:

    # ---------- LINE CHART (multi-subplots, synced X) ----------
    with st.expander("Line Chart (multi-subplots, synced X)", expanded=True):
        st.caption("Stack up to 6 synced subplots. Great for related indicators on a shared time axis.")
        from plotly.subplots import make_subplots
    
        with st.expander("Line chart options", expanded=False):
            numeric_cols = work.select_dtypes(include=[np.number]).columns.tolist()
    
            n_subplots = st.number_input(
                "Number of subplots",
                min_value=1,
                max_value=6,
                value=1,
                step=1,
                key="mlc_n_subplots",
                help="Stack up to 6 subplots that share the same X axis.",
            )
    
            show_markers = st.checkbox("Show markers", value=False, key="mlc_markers")
            line_shape = st.selectbox(
                "Line shape",
                options=["linear", "spline", "hv", "vh", "hvh", "vhv"],
                index=0,
                key="mlc_shape",
            )
            line_width = st.slider("Line width", 1, 8, 2, key="mlc_width")
            yaxis_type = st.selectbox("Y-axis scale (all subplots)", ["linear", "log"], index=0, key="mlc_ytype")
            range_slider = st.checkbox("Show range slider (bottom x-axis)", value=True, key="mlc_rangeslider")
            unified_hover = st.checkbox("Unified hover (x)", value=True, key="mlc_unified")
            show_spikes = st.checkbox("Show spike lines", value=True, key="mlc_spikes")
            max_points = st.slider(
                "Max points per series (downsampling)",
                min_value=1_000, max_value=100_000, value=30_000, step=1_000,
                key="mlc_maxpts"
            )
    
        # Per-subplot column selectors
        subplot_selections = []
        for i in range(int(n_subplots)):
            default_i = [numeric_cols[i]] if i < len(numeric_cols) else []
            cols_i = st.multiselect(
                f"Subplot {i+1} — columns",
                options=numeric_cols,
                default=default_i,
                key=f"mlc_cols_{i}",
                help="Pick one or more numeric series for this subplot."
            )
            subplot_selections.append(cols_i)
    
        # Do we have anything to plot?
        if any(len(sel) > 0 for sel in subplot_selections):
            # Union of all selected columns (for a single downsample pass)
            union_cols = sorted({c for sel in subplot_selections for c in sel})
            plot_df = work[union_cols].copy()
            plot_df = _maybe_downsample_for_plot(plot_df, max_points=max_points)
    
            # X axis values and label
            if isinstance(plot_df.index, pd.DatetimeIndex):
                x_vals = plot_df.index
                x_name = plot_df.index.name or "time"
            else:
                x_vals = plot_df.index.astype(float) if pd.api.types.is_numeric_dtype(plot_df.index) else plot_df.index.astype(str)
                x_name = plot_df.index.name or "index"
    
            # Build subplots
            fig = make_subplots(
                rows=int(n_subplots),
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                subplot_titles=[f"Subplot {r+1}" for r in range(int(n_subplots))],
            )
    
            # Add traces per subplot
            for r, cols in enumerate(subplot_selections, start=1):
                for c in cols:
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=plot_df[c],
                            mode="lines+markers" if show_markers else "lines",
                            name=f"{c} (S{r})",
                            line=dict(shape=line_shape, width=line_width),
                            hovertemplate="%{x}<br>%{y}<extra>%{fullData.name}</extra>",
                        ),
                        row=r, col=1
                    )
                # Y-axis scale per subplot (global choice applied)
                fig.update_yaxes(type=yaxis_type, row=r, col=1)
    
            # Global layout
            fig.update_layout(
                hovermode="x unified" if unified_hover else "closest",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                margin=dict(l=40, r=20, t=40, b=40),
                showlegend=True,
            )
    
            # Spikes on all rows
            for r in range(1, int(n_subplots) + 1):
                fig.update_xaxes(showspikes=show_spikes, spikemode="across", spikesnap="cursor", row=r, col=1)
    
            # Range slider only on bottom axis
            fig.update_xaxes(rangeslider=dict(visible=range_slider), row=int(n_subplots), col=1)
            # Label bottom x axis
            fig.update_xaxes(title=x_name, row=int(n_subplots), col=1)
    
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    
            # Download the data exactly as plotted
            with st.expander("Download (multi-subplot line data)"):
                out_df = pd.DataFrame(index=plot_df.index)
                for r, cols in enumerate(subplot_selections, start=1):
                    for c in cols:
                        out_df[f"{c} [subplot {r}]"] = plot_df[c]
                out_df = out_df.reset_index().rename(columns={"index": x_name})
                st.download_button(
                    label="Download plotted data (CSV)",
                    data=out_df.to_csv(index=False).encode("utf-8"),
                    file_name="multi_subplots_line_data.csv",
                    mime="text/csv",
                    help="Exactly the data used across all subplots after downsampling."
                )
        else:
            st.info("Select at least one series across the subplots to draw the chart.")


    # ---------- HISTOGRAM ----------
    with st.expander("Histogram (value distribution of selected columns)", expanded=True):
        st.caption("Visualize value distributions. Binning supports Auto (FD/Scott/Sturges), Fixed width, or Target bins.")
        with st.expander("Histogram options", expanded=False):
            numeric_cols = work.select_dtypes(include=[np.number]).columns.tolist()
            default_hist = numeric_cols[:1] if numeric_cols else []
            hist_cols = st.multiselect(
                "Columns for histogram",
                options=numeric_cols,
                default=default_hist,
                key="hist_cols_main",
            )

            bin_mode = st.selectbox(
                "Binning mode",
                options=["Auto (width)", "Fixed width", "Target #bins"],
                index=0,
                help="Auto uses Freedman-Diaconis with Scott/Sturges fallbacks. Fixed width lets you pick bin size. Target #bins matches the old behavior.",
                key="hist_binmode",
            )

            fixed_width = None
            target_bins = None
            if bin_mode == "Fixed width":
                fixed_width = st.number_input(
                    "Bin width",
                    min_value=1e-12,
                    value=1.0,
                    step=1.0,
                    format="%.6g",
                    help="Size of each bin along x.",
                    key="hist_width_input",
                )
            elif bin_mode == "Target #bins":
                target_bins = st.slider(
                    "Number of bins",
                    min_value=10, max_value=200, value=50,
                    key="hist_bins_main",
                )

        if hist_cols:
            vals_all = pd.concat([work[c].dropna().astype(float) for c in hist_cols], axis=0)
            if len(vals_all) < 2 or vals_all.min() == vals_all.max():
                st.info("Not enough variance to build histogram bins.")
            else:
                vmin, vmax = float(vals_all.min()), float(vals_all.max())

                def _compute_auto_width(x: pd.Series) -> float:
                    x = x.to_numpy()
                    x = x[np.isfinite(x)]
                    n = x.size
                    if n < 2:
                        return max((vmax - vmin), 1.0)
                    iqr = np.subtract(*np.percentile(x, [75, 25]))
                    if np.isfinite(iqr) and iqr > 0:
                        h = 2.0 * iqr * (n ** (-1.0/3.0))  # Freedman-Diaconis
                    else:
                        sigma = np.nanstd(x, ddof=1)
                        if np.isfinite(sigma) and sigma > 0:
                            h = 3.5 * sigma * (n ** (-1.0/3.0))  # Scott
                        else:
                            k = max(1, int(np.ceil(np.log2(n) + 1)))  # Sturges
                            span = max(vmax - vmin, np.finfo(float).eps)
                            h = span / k
                    return max(h, np.finfo(float).eps)

                if bin_mode == "Auto (width)":
                    width = _compute_auto_width(vals_all)
                elif bin_mode == "Fixed width":
                    width = float(fixed_width)
                else:
                    k = int(target_bins)
                    span = max(vmax - vmin, np.finfo(float).eps)
                    width = span / k

                n_bins = max(1, int(np.ceil((vmax - vmin) / width)))
                edges = vmin + np.arange(n_bins + 1) * width
                if edges[-1] < vmax:
                    edges = np.append(edges, edges[-1] + width)
                    n_bins += 1
                left_edges = edges[:-1]
                right_edges = edges[1:]
                bin_widths = right_edges - left_edges
                centers = left_edges + bin_widths / 2.0
                hover_labels = [
                    f"[{l:.6g}, {r:.6g}{']' if i == len(left_edges) - 1 else ')'}"
                    for i, (l, r) in enumerate(zip(left_edges, right_edges))
                ]

                hist_counts_dict = {}
                fig_hist = go.Figure()
                for c in hist_cols:
                    arr = work[c].dropna().astype(float).to_numpy()
                    hist_counts, _ = np.histogram(arr, bins=edges)
                    hist_counts_dict[c] = hist_counts
                    fig_hist.add_trace(
                        go.Bar(
                            x=centers,
                            y=hist_counts,
                            width=bin_widths,
                            name=str(c),
                            text=hover_labels,
                            hovertemplate="bin=%{text}<br>count=%{y}<extra>%{fullData.name}</extra>",
                        )
                    )

                fig_hist.update_layout(
                    barmode="overlay",
                    margin=dict(l=40, r=20, t=10, b=40),
                    hovermode="x",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                    xaxis_title="Value",
                    yaxis_title="Count",
                    showlegend=True,
                )
                fig_hist.update_xaxes(showline=True, zeroline=False)

                st.plotly_chart(fig_hist, use_container_width=True, theme="streamlit")

                with st.expander("Download (histogram data)"):
                    meta_line = f"# mode={bin_mode}, width={width:.6g}, bins={n_bins}, range=[{vmin:.6g}, {vmax:.6g}]"
                    hist_df = pd.DataFrame(
                        {
                            "bin_left": left_edges,
                            "bin_right": right_edges,
                            "bin_center": centers,
                        }
                    )
                    for key, values in hist_counts_dict.items():
                        hist_df[key] = values
                    hist_df = hist_df[["bin_left", "bin_right", "bin_center", *hist_counts_dict.keys()]]
                    csv_bytes = (meta_line + "\n" + hist_df.to_csv(index=False)).encode("utf-8")
                    st.download_button(
                        label="Download histogram (CSV)",
                        data=csv_bytes,
                        file_name="histogram_counts.csv",
                        mime="text/csv",
                        help="Bin centers and counts per selected series. First line documents mode/width/bin count.",
                    )
                st.caption(f"Computed width: {width:.6g} · Resulting bins: {n_bins} · Range: [{vmin:.6g}, {vmax:.6g}]")
        else:
            st.info("Pick columns in Histogram options to generate the histogram.")

    # ---------- SCATTER (grouped optional) ----------
    with st.expander("Scatter (grouped optional)", expanded=True):
        st.caption("Explore relationships between variables. Optional grouping switches between discrete palette or colorbar.")
        with st.expander("Scatter options", expanded=False):
            all_cols = list(work.columns)
            numeric_cols = work.select_dtypes(include=[np.number]).columns.tolist()
            default_x = numeric_cols[0] if numeric_cols else (all_cols[0] if all_cols else None)
            default_y = numeric_cols[1] if len(numeric_cols) > 1 else (all_cols[1] if len(all_cols) > 1 else None)

            x_idx = all_cols.index(default_x) if (default_x in all_cols) else (0 if all_cols else 0)
            y_idx = all_cols.index(default_y) if (default_y in all_cols) else (0 if all_cols else 0)

            x_col = st.selectbox("X variable", options=all_cols, index=x_idx, key="sc_x")
            y_col = st.selectbox("Y variable", options=all_cols, index=y_idx, key="sc_y")
            group_col = st.selectbox("Group (optional)", options=["(none)"] + all_cols, index=0, key="sc_group")

            CONT_CMAPS = {
                "Viridis": px.colors.sequential.Viridis,
                "Plasma": px.colors.sequential.Plasma,
                "Inferno": px.colors.sequential.Inferno,
                "Magma": px.colors.sequential.Magma,
                "Cividis": px.colors.sequential.Cividis,
                "Turbo": px.colors.sequential.Turbo,
                "Blues": px.colors.sequential.Blues,
                "Greens": px.colors.sequential.Greens,
                "Greys": px.colors.sequential.Greys,
                "Oranges": px.colors.sequential.Oranges,
                "Purples": px.colors.sequential.Purples,
                "Reds": px.colors.sequential.Reds,
            }
            DISC_PALETTES = {
                "Plotly": px.colors.qualitative.Plotly,
                "D3": px.colors.qualitative.D3,
                "Set1": px.colors.qualitative.Set1,
                "Set2": px.colors.qualitative.Set2,
                "Set3": px.colors.qualitative.Set3,
                "Pastel": px.colors.qualitative.Pastel,
                "Bold": px.colors.qualitative.Bold,
                "Dark24": px.colors.qualitative.Dark24,
                "Light24": px.colors.qualitative.Light24,
            }
            cont_cmap_name = st.selectbox("Continuous colormap (numeric group)", options=list(CONT_CMAPS.keys()), index=0, key="sc_cmap_cont")
            disc_palette_name = st.selectbox("Discrete palette (categorical group)", options=list(DISC_PALETTES.keys()), index=0, key="sc_cmap_disc")

            max_points_sc = st.slider("Max points (downsampling)", min_value=1_000, max_value=200_000, value=50_000, step=1_000, key="sc_maxpts")
            marker_size = st.slider("Marker size", min_value=3, max_value=20, value=7, key="sc_msize")
            marker_opacity = st.slider("Marker opacity", min_value=0.2, max_value=1.0, value=0.8, step=0.05, key="sc_mop")

        needed_cols = [c for c in [x_col, y_col, (None if group_col == "(none)" else group_col)] if c]
        if not needed_cols:
            st.info("Choose X and Y to draw the scatter.")
        else:
            plot_sc = work[needed_cols].dropna().copy()
            if len(plot_sc) > max_points_sc:
                stride = max(1, len(plot_sc) // max_points_sc)
                plot_sc = plot_sc.iloc[::stride]

            color_arg = None if group_col == "(none)" else group_col

            if color_arg is None:
                fig_sc = px.scatter(
                    plot_sc,
                    x=x_col,
                    y=y_col,
                    opacity=marker_opacity,
                )
            else:
                is_numeric_group = pd.api.types.is_numeric_dtype(plot_sc[color_arg])
                if is_numeric_group:
                    fig_sc = px.scatter(
                        plot_sc,
                        x=x_col,
                        y=y_col,
                        color=color_arg,
                        color_continuous_scale=CONT_CMAPS[cont_cmap_name],
                        opacity=marker_opacity,
                    )
                    fig_sc.update_layout(coloraxis_colorbar=dict(title=color_arg))
                else:
                    fig_sc = px.scatter(
                        plot_sc,
                        x=x_col,
                        y=y_col,
                        color=color_arg,
                        color_discrete_sequence=DISC_PALETTES[disc_palette_name],
                        opacity=marker_opacity,
                    )
                    fig_sc.update_layout(
                        legend=dict(orientation="v", y=0.5, yanchor="middle", x=1.02, xanchor="left")
                    )

            fig_sc.update_traces(marker=dict(size=marker_size))
            if color_arg is None:
                fig_sc.update_traces(name=f"{y_col} vs {x_col}", showlegend=True)
                fig_sc.update_layout(legend=dict(orientation="v", y=0.5, yanchor="middle", x=1.02, xanchor="left"))

            fig_sc.update_layout(
                margin=dict(l=40, r=20, t=10, b=40),
                showlegend=True,
            )
            st.plotly_chart(fig_sc, use_container_width=True, theme="streamlit")

            with st.expander("Download (scatter data)"):
                st.download_button(
                    label="Download scatter data (CSV)",
                    data=plot_sc.to_csv(index=False).encode("utf-8"),
                    file_name="scatter_data.csv",
                    mime="text/csv",
                    help="Exact data used to draw the scatter (after local downsampling)."
                )
                
    # ---------- SCATTER (color by g1, marker by g2; two-part legend) ----------
    with st.expander("Scatter (g1 → color, g2 → marker)", expanded=True):
        st.caption("Two independent encodings: colors = g1 categories, markers = g2 categories. Legend shows both mappings.")
        from collections import Counter
    
        with st.expander("Scatter (g1/g2) options", expanded=False):
            all_cols = list(work.columns)
            numeric_cols = work.select_dtypes(include=[np.number]).columns.tolist()
    
            # Defaults: first two numeric columns for X/Y
            default_x = numeric_cols[0] if numeric_cols else (all_cols[0] if all_cols else None)
            default_y = numeric_cols[1] if len(numeric_cols) > 1 else (all_cols[1] if len(all_cols) > 1 else None)
    
            x_col = st.selectbox("X variable", options=all_cols, index=(all_cols.index(default_x) if default_x in all_cols else 0), key="sc2_x")
            y_col = st.selectbox("Y variable", options=all_cols, index=(all_cols.index(default_y) if default_y in all_cols else max(0, len(all_cols)-1)), key="sc2_y")
    
            g1_col = st.selectbox("Group by (color) — g1", options=["(none)"] + all_cols, index=0, key="sc2_g1")
            g2_col = st.selectbox("Group by (marker) — g2", options=["(none)"] + all_cols, index=0, key="sc2_g2")
    
            # Discrete palettes (same set you used before)
            DISC_PALETTES = {
                "Plotly": px.colors.qualitative.Plotly,
                "D3": px.colors.qualitative.D3,
                "Set1": px.colors.qualitative.Set1,
                "Set2": px.colors.qualitative.Set2,
                "Set3": px.colors.qualitative.Set3,
                "Pastel": px.colors.qualitative.Pastel,
                "Bold": px.colors.qualitative.Bold,
                "Dark24": px.colors.qualitative.Dark24,
                "Light24": px.colors.qualitative.Light24,
            }
            disc_palette_name = st.selectbox("Discrete palette (for g1 colors)", options=list(DISC_PALETTES.keys()), index=0, key="sc2_palette")

            MARKER_PRESETS = {
                "Core Shapes": [
                    "circle",
                    "square",
                    "diamond",
                    "triangle-up",
                    "triangle-down",
                    "triangle-left",
                    "triangle-right",
                ],
                "Crosshair Set": [
                    "circle-open",
                    "square-open",
                    "diamond-open",
                    "x",
                    "cross",
                    "triangle-up-open",
                    "triangle-down-open",
                ],
                "Directional": [
                    "triangle-up",
                    "triangle-right",
                    "triangle-down",
                    "triangle-left",
                    "arrow-up",
                    "arrow-right",
                    "arrow-down",
                    "arrow-left",
                ],
                "Starburst": [
                    "star",
                    "hexagon",
                    "pentagon",
                    "bowtie",
                    "hourglass",
                    "circle-cross",
                    "square-x",
                ],
                "Minimal Dots": [
                    "circle",
                    "circle-open",
                    "circle-dot",
                    "circle-open-dot",
                    "circle-cross",
                    "circle-open-cross",
                ],
            }
            marker_preset_name = st.selectbox(
                "Marker preset (for g2 symbols)",
                options=list(MARKER_PRESETS.keys()),
                index=0,
                key="sc2_marker_preset",
            )
            custom_marker_input = st.text_input(
                "Custom marker list (comma separated, overrides preset)",
                value="",
                placeholder="circle, square-open, diamond, x",
                key="sc2_marker_custom",
            )

            max_points_sc = st.slider("Max points (downsampling)", min_value=1_000, max_value=200_000, value=50_000, step=1_000, key="sc2_maxpts")
            marker_size = st.slider("Marker size", min_value=3, max_value=20, value=7, key="sc2_msize")
            marker_opacity = st.slider("Marker opacity", min_value=0.2, max_value=1.0, value=0.8, step=0.05, key="sc2_mop")

        # Resolve marker list (custom text overrides preset if valid)
        marker_sequence = MARKER_PRESETS[marker_preset_name]
        marker_sequence_label = f"'{marker_preset_name}' preset"
        custom_marker_tokens: list[str] = []
        invalid_marker_tokens: list[str] = []
        if custom_marker_input:
            custom_marker_tokens = [tok.strip() for tok in custom_marker_input.split(",") if tok.strip()]
            if custom_marker_tokens:
                validator = SymbolValidator() if SymbolValidator is not None else None
                resolved_tokens: list[str] = []
                for tok in custom_marker_tokens:
                    if validator is None:
                        resolved_tokens.append(tok)
                        continue
                    try:
                        resolved_tokens.append(validator.validate_coerce(tok))
                    except Exception:
                        invalid_marker_tokens.append(tok)
                if invalid_marker_tokens:
                    skipped = ", ".join(sorted(set(invalid_marker_tokens)))
                    st.warning(f"Unrecognized marker symbols skipped: {skipped}")
                if resolved_tokens:
                    marker_sequence = resolved_tokens
                    marker_sequence_label = "custom marker list"
                else:
                    st.warning("No valid markers detected in the custom list. Reverting to the selected preset.")

        palette = DISC_PALETTES[disc_palette_name]

        summary_lines = [
            f"- **X**: `{x_col}`",
            f"- **Y**: `{y_col}`",
            f"- **Color grouping (g1)**: `{g1_col}`" if g1_col != "(none)" else "- **Color grouping (g1)**: _disabled_",
            f"- **Marker grouping (g2)**: `{g2_col}`" if g2_col != "(none)" else "- **Marker grouping (g2)**: _disabled_",
            f"- **Color palette**: `{disc_palette_name}` ({len(palette)} colors)",
            f"- **Marker source**: {marker_sequence_label} ({len(marker_sequence)} symbols)",
            f"- **Markers**: `{', '.join(marker_sequence)}`" if marker_sequence else "- **Markers**: _none_",
        ]
        st.info("\n".join(summary_lines))

        # Build base DF
        needed = [x_col, y_col]
        if g1_col != "(none)":
            needed.append(g1_col)
        if g2_col != "(none)":
            needed.append(g2_col)
    
        if not all(needed):
            st.info("Choose X and Y (and optionally g1, g2) to draw the scatter.")
        else:
            plot_sc = work[needed].dropna().copy()
    
            # Treat group columns as categorical strings so we get a discrete legend
            if g1_col != "(none)":
                plot_sc[g1_col] = plot_sc[g1_col].astype(str)
            if g2_col != "(none)":
                plot_sc[g2_col] = plot_sc[g2_col].astype(str)
    
            # Downsample
            if len(plot_sc) > max_points_sc:
                stride = max(1, len(plot_sc) // max_points_sc)
                plot_sc = plot_sc.iloc[::stride]
    
            # Prepare categories & mappings
            g1_cats = list(plot_sc[g1_col].unique()) if g1_col != "(none)" else ["All"]
            g2_cats = list(plot_sc[g2_col].unique()) if g2_col != "(none)" else ["All"]
    
            if len(g1_cats) > len(palette):
                st.warning(f"g1 has {len(g1_cats)} categories, but palette has {len(palette)} colors. Cycling colors.")
            if len(g2_cats) > len(marker_sequence):
                st.warning(
                    f"g2 has {len(g2_cats)} categories, but the {marker_sequence_label} has {len(marker_sequence)} symbols. Cycling symbols."
                )

            color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(g1_cats)}
            symbol_map = {cat: marker_sequence[i % len(marker_sequence)] for i, cat in enumerate(g2_cats)}
    
            # Build figure with actual data traces (no legend to avoid clutter)
            fig = go.Figure()
    
            # Add one trace per (g1, g2) combination
            if g1_col == "(none)" and g2_col == "(none)":
                # Simple scatter
                fig.add_trace(go.Scattergl(
                    x=plot_sc[x_col], y=plot_sc[y_col],
                    mode="markers",
                    marker=dict(size=marker_size, opacity=marker_opacity),
                    name="points",
                    showlegend=False,
                ))
            else:
                # Group and add traces
                if g1_col == "(none)":   # only g2 grouping
                    for g2v, df_sub in plot_sc.groupby(g2_col):
                        fig.add_trace(go.Scattergl(
                            x=df_sub[x_col], y=df_sub[y_col],
                            mode="markers",
                            marker=dict(size=marker_size, opacity=marker_opacity, symbol=symbol_map[g2v]),
                            name=f"{g2_col}={g2v}",
                            showlegend=False,
                        ))
                elif g2_col == "(none)":  # only g1 grouping
                    for g1v, df_sub in plot_sc.groupby(g1_col):
                        fig.add_trace(go.Scattergl(
                            x=df_sub[x_col], y=df_sub[y_col],
                            mode="markers",
                            marker=dict(size=marker_size, opacity=marker_opacity, color=color_map[g1v]),
                            name=f"{g1_col}={g1v}",
                            showlegend=False,
                        ))
                else:
                    # both g1 & g2
                    for (g1v, g2v), df_sub in plot_sc.groupby([g1_col, g2_col]):
                        fig.add_trace(go.Scattergl(
                            x=df_sub[x_col], y=df_sub[y_col],
                            mode="markers",
                            marker=dict(
                                size=marker_size,
                                opacity=marker_opacity,
                                color=color_map[g1v],
                                symbol=symbol_map[g2v],
                            ),
                            name=f"{g1_col}={g1v}, {g2_col}={g2v}",
                            showlegend=False,  # keep legend clean; we'll add guides below
                        ))
    
            # --- Legend guides: Colors (g1) ---
            if g1_col != "(none)":
                for i, g1v in enumerate(g1_cats):
                    fig.add_trace(go.Scatter(
                        x=[None], y=[None],  # invisible point for legend only
                        mode="markers",
                        marker=dict(color=color_map[g1v], size=10),
                        name=str(g1v),
                        legendgroup=f"color_{g1_col}",
                        legendgrouptitle_text=f"Color: {g1_col}" if i == 0 else None,
                        showlegend=True,
                    ))
    
            # --- Legend guides: Markers (g2) ---
            if g2_col != "(none)":
                for i, g2v in enumerate(g2_cats):
                    fig.add_trace(go.Scatter(
                        x=[None], y=[None],
                        mode="markers",
                        marker=dict(symbol=symbol_map[g2v], size=10, line=dict(width=1)),
                        name=str(g2v),
                        legendgroup=f"marker_{g2_col}",
                        legendgrouptitle_text=f"Marker: {g2_col}" if i == 0 else None,
                        showlegend=True,
                    ))
    
            fig.update_layout(
                margin=dict(l=40, r=20, t=10, b=40),
                legend=dict(orientation="v", y=0.5, yanchor="middle", x=1.02, xanchor="left"),
                legend_tracegroupgap=10,  # visual gap between groups
            )
    
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    
            with st.expander("Download (scatter g1/g2 data)"):
                st.download_button(
                    label="Download scatter data (CSV)",
                    data=plot_sc.to_csv(index=False).encode("utf-8"),
                    file_name="scatter_g1_g2_data.csv",
                    mime="text/csv",
                    help="Exact data used to draw the scatter (after local downsampling)."
                )
  
    # ---------- SCATTER + ROLLING WINDOW (past n rows: min / mean / max) ----------
    with st.expander("Scatter + Rolling Window (min / mean / max)", expanded=True):
        st.caption("Scatter the selected series vs the current index and overlay trailing rolling min/mean/max computed over the past n rows.")
    
        with st.expander("Rolling options", expanded=False):
            numeric_cols = work.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                st.info("No numeric columns available.")
            else:
                y_default = numeric_cols[0]
                y_col = st.selectbox(
                    "Y variable (scatter target)",
                    options=numeric_cols,
                    index=numeric_cols.index(y_default),
                    key="rw_y",
                )
    
                # Trailing window settings (past n rows)
                window = st.number_input(
                    "Rolling window (past n rows)",
                    min_value=2, max_value=200_000, value=50, step=1,
                    help="Trailing window across past n rows.",
                    key="rw_window",
                )
    
                include_current = st.checkbox(
                    "Include current row in window",
                    value=True,
                    help="ON: window includes current row + previous n−1 rows. OFF: strictly previous n rows (current excluded).",
                    key="rw_include_current",
                )
    
                require_full = st.checkbox(
                    "Require full window (min_periods = window)",
                    value=False, key="rw_full"
                )
                min_periods = int(window if require_full else 1)
    
                show_band = st.checkbox("Shade band between min and max", value=True, key="rw_band")
                unified_hover = st.checkbox("Unified hover (x)", value=True, key="rw_unified")
                show_spikes = st.checkbox("Show spike lines", value=True, key="rw_spikes")
                range_slider = st.checkbox(
                    "Show range slider (x-axis)",
                    value=isinstance(work.index, pd.DatetimeIndex),
                    key="rw_rangeslider",
                )
    
                marker_size = st.slider("Scatter marker size", 3, 16, 7, key="rw_msize")
                marker_opacity = st.slider("Scatter marker opacity", 0.2, 1.0, 0.7, 0.05, key="rw_mop")
                max_points = st.slider("Max points (downsampling)", 1_000, 200_000, 50_000, 1_000, key="rw_maxpts")
    
        if numeric_cols:
            # Series + trailing rolling stats (past n rows)
            s = pd.to_numeric(work[y_col], errors="coerce")
    
            # Exclude current by shifting if requested
            base = s if include_current else s.shift(1)
    
            roll = base.rolling(window=window, min_periods=min_periods)  # trailing (center=False)
            s_min  = roll.min()
            s_mean = roll.mean()
            s_max  = roll.max()
    
            plot_df = pd.DataFrame(
                {
                    y_col: s,
                    f"{y_col}__roll_min": s_min,
                    f"{y_col}__roll_mean": s_mean,
                    f"{y_col}__roll_max": s_max,
                },
                index=work.index,
            ).dropna(how="all")
    
            # Downsample for plotting
            plot_df_ds = _maybe_downsample_for_plot(plot_df, max_points=max_points)
    
            # X axis values/title
            if isinstance(plot_df_ds.index, pd.DatetimeIndex):
                x_vals = plot_df_ds.index
                x_name = plot_df_ds.index.name or "time"
            else:
                x_vals = (
                    plot_df_ds.index.astype(float)
                    if pd.api.types.is_numeric_dtype(plot_df_ds.index)
                    else plot_df_ds.index.astype(str)
                )
                x_name = plot_df_ds.index.name or "index"
    
            # Figure
            fig = go.Figure()
    
            # 1) Scatter of raw values
            fig.add_trace(
                go.Scattergl(
                    x=x_vals,
                    y=plot_df_ds[y_col],
                    mode="markers",
                    name=y_col,
                    marker=dict(size=marker_size, opacity=marker_opacity),
                    hovertemplate="%{x}<br>%{y}<extra>"+y_col+"</extra>",
                )
            )
    
            # 2) Rolling lines (min / mean / max)
            y_min  = plot_df_ds[f"{y_col}__roll_min"]
            y_mean = plot_df_ds[f"{y_col}__roll_mean"]
            y_max  = plot_df_ds[f"{y_col}__roll_max"]
    
            if show_band:
                # add min first, then max with fill to show band
                fig.add_trace(
                    go.Scatter(
                        x=x_vals, y=y_min,
                        mode="lines",
                        name="rolling min",
                        line=dict(width=1),
                        hovertemplate="%{x}<br>%{y}<extra>rolling min</extra>",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_vals, y=y_max,
                        mode="lines",
                        name="rolling max",
                        line=dict(width=1),
                        fill="tonexty",
                        fillcolor="rgba(100,100,100,0.15)",
                        hovertemplate="%{x}<br>%{y}<extra>rolling max</extra>",
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=x_vals, y=y_min,
                        mode="lines",
                        name="rolling min",
                        line=dict(width=1),
                        hovertemplate="%{x}<br>%{y}<extra>rolling min</extra>",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_vals, y=y_max,
                        mode="lines",
                        name="rolling max",
                        line=dict(width=1),
                        hovertemplate="%{x}<br>%{y}<extra>rolling max</extra>",
                    )
                )
    
            # Mean on top
            fig.add_trace(
                go.Scatter(
                    x=x_vals, y=y_mean,
                    mode="lines",
                    name="rolling mean",
                    line=dict(width=2),
                    hovertemplate="%{x}<br>%{y}<extra>rolling mean</extra>",
                )
            )
    
            # Layout
            fig.update_layout(
                margin=dict(l=40, r=20, t=10, b=40),
                hovermode="x unified" if unified_hover else "closest",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                showlegend=True,
            )
            fig.update_xaxes(
                rangeslider=dict(visible=range_slider),
                showspikes=show_spikes,
                spikemode="across",
                spikesnap="cursor",
                showline=True,
                zeroline=False,
                title=x_name,
            )
    
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    
            # Download data exactly as plotted
            with st.expander("Download (rolling scatter data)"):
                out = plot_df_ds.reset_index().rename(columns={"index": x_name})
                st.download_button(
                    "Download CSV",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name=f"rolling_scatter_{y_col}.csv",
                    mime="text/csv",
                    help="Includes raw values and rolling min/mean/max (after local downsampling).",
                )
                
    # ---------- LINE + DERIVATIVES (value, 1st, 2nd) with trailing alignment ----------
    with st.expander("Line + Derivatives (value, 1st, 2nd)", expanded=True):
        from plotly.subplots import make_subplots
    
        # Define available numeric columns once (used outside options block too)
        numeric_cols = work.select_dtypes(include=[np.number]).columns.tolist()
    
        with st.expander("Derivative options", expanded=False):
            if not numeric_cols:
                st.info("No numeric columns available.")
            else:
                y_default = numeric_cols[0]
                y_col = st.selectbox(
                    "Series (Y)",
                    options=numeric_cols,
                    index=numeric_cols.index(y_default),
                    key="der_y",
                    help="Pick the series to analyze; derivatives are computed with respect to the current index."
                )
    
                if isinstance(work.index, pd.DatetimeIndex):
                    # Add "none (sample index)" so derivatives ignore time spacing
                    t_unit = st.selectbox(
                        "Time unit (for derivatives)",
                        options=["seconds", "minutes", "hours", "days", "none (sample index)"],
                        index=0,
                        key="der_tunit"
                    )
                else:
                    x_mode = st.selectbox(
                        "X spacing (for derivatives)",
                        options=["index step (1)", "use numeric index values"],
                        index=0,
                        key="der_xmode",
                        help="If your index is numeric and meaningful, choose 'use numeric index values'."
                    )
    
                show_markers   = st.checkbox("Show markers", value=True, key="der_markers")
                line_width     = st.slider("Line width", 1, 8, 2, key="der_lw")
                unified_hover  = st.checkbox("Unified hover (x)", value=True, key="der_unified")
                range_slider   = st.checkbox("Show range slider (x-axis)",
                                             value=isinstance(work.index, pd.DatetimeIndex), key="der_rangeslider")
                show_spikes    = st.checkbox("Show spike lines", value=True, key="der_spikes")
    
                # Optional smoothing for derivatives only (preserves leading NaNs)
                smooth_w = st.number_input(
                    "Smooth derivatives (rolling mean window; 1 = off)",
                    min_value=1, max_value=9999, value=1, step=1, key="der_smooth",
                    help="Applies a rolling mean to dY/dX and d²Y/dX² for readability (original series unchanged)."
                )
    
                max_points = st.slider(
                    "Max points (downsampling)",
                    1_000, 200_000, 50_000, 1_000, key="der_maxpts",
                    help="For speed, the plotted data may be stride-downsampled."
                )
    
        if numeric_cols:
            # Base series (drop NA and sort)
            df = work[[y_col]].dropna().copy()
            try:
                df = df.sort_index()
            except Exception:
                pass
    
            if len(df) < 3:
                st.info("Need at least 3 points to compute a stable second derivative.")
            else:
                y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype="float64")
    
                # Build x in consistent units for derivatives
                n = len(df)
                if isinstance(df.index, pd.DatetimeIndex):
                    if t_unit == "none (sample index)":
                        # Uniform step by sample index
                        x = np.arange(n, dtype="float64")
                        x_label = "sample index (uniform)"
                        der1_label = f"d/dn({y_col})"
                        der2_label = f"d²/dn²({y_col})"
                    else:
                        # Time-based spacing
                        t_sec = df.index.asi8.astype("float64") / 1e9  # ns -> seconds
                        unit_div = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0, "days": 86400.0}[t_unit]
                        x = t_sec / unit_div
                        x_label = f"time ({t_unit})"
                        der1_label = f"d/dt({y_col})"
                        der2_label = f"d²/dt²({y_col})"
                else:
                    if x_mode == "use numeric index values" and pd.api.types.is_numeric_dtype(df.index):
                        x = df.index.to_numpy(dtype="float64")
                        x_label = df.index.name or "index"
                        der1_label = f"d/dx({y_col})"
                        der2_label = f"d²/dx²({y_col})"
                    else:
                        x = np.arange(n, dtype="float64")
                        x_label = "sample index (uniform)"
                        der1_label = f"d/dn({y_col})"
                        der2_label = f"d²/dn²({y_col})"
    
                # Guard: degenerate/invalid x → fall back to unit steps
                if (not np.isfinite(x).all()) or np.allclose(x.max(), x.min()):
                    x = np.arange(n, dtype="float64")
                    x_label = "sample index (uniform)"
                    der1_label = f"d/dn({y_col})"
                    der2_label = f"d²/dn²({y_col})"
    
                # Trailing derivatives with alignment:
                # d1[t] = (y[t]-y[t-1]) / (x[t]-x[t-1])    for t>=1 ; d1[0]=NaN
                # d2[t] = (d1[t]-d1[t-1]) / (x[t]-x[t-1])  for t>=2 ; d2[0:2]=NaN
                d1 = np.full(n, np.nan)
                d2 = np.full(n, np.nan)
    
                dy = np.diff(y)
                dx = np.diff(x)
                with np.errstate(divide="ignore", invalid="ignore"):
                    d1_vals = np.where(dx != 0, dy / dx, np.nan)
                d1[1:] = d1_vals
    
                num = d1[2:] - d1[1:-1]
                ddx = dx[1:]
                with np.errstate(divide="ignore", invalid="ignore"):
                    d2_vals = np.where(ddx != 0, num / ddx, np.nan)
                d2[2:] = d2_vals
    
                # Optional smoothing (keep leading NaNs)
                if int(smooth_w) > 1:
                    w = int(smooth_w)
                    d1_sm = pd.Series(d1).rolling(w, min_periods=1).mean().to_numpy()
                    d2_sm = pd.Series(d2).rolling(w, min_periods=1).mean().to_numpy()
                    d1_sm[:1] = np.nan
                    d2_sm[:2] = np.nan
                    d1, d2 = d1_sm, d2_sm
    
                # Assemble plotted dataframe
                plot_df = pd.DataFrame(
                    {
                        y_col: y,
                        der1_label: d1,
                        der2_label: d2,
                    },
                    index=df.index,
                )
    
                # Downsample for plotting (preserves NaNs)
                plot_df_ds = _maybe_downsample_for_plot(plot_df, max_points=max_points)
    
                # X for plotting (original index)
                if isinstance(plot_df_ds.index, pd.DatetimeIndex):
                    x_vals = plot_df_ds.index
                    x_name = plot_df_ds.index.name or "time"
                else:
                    x_vals = (
                        plot_df_ds.index.astype(float)
                        if pd.api.types.is_numeric_dtype(plot_df_ds.index)
                        else plot_df_ds.index.astype(str)
                    )
                    x_name = plot_df_ds.index.name or "index"
    
                # Build 3 stacked subplots sharing X
                fig = make_subplots(
                    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                    subplot_titles=[y_col, der1_label, der2_label]
                )
    
                # Subplot 1: original series
                fig.add_trace(
                    go.Scatter(
                        x=x_vals, y=plot_df_ds[y_col],
                        mode="lines+markers" if show_markers else "lines",
                        name=y_col, line=dict(width=line_width),
                        hovertemplate="%{x}<br>%{y}<extra>"+y_col+"</extra>",
                    ),
                    row=1, col=1
                )
    
                # Subplot 2: first derivative (NaN at first point; aligned at current t)
                fig.add_trace(
                    go.Scatter(
                        x=x_vals, y=plot_df_ds[der1_label],
                        mode="lines+markers" if show_markers else "lines",
                        name=der1_label, line=dict(width=line_width),
                        hovertemplate="%{x}<br>%{y}<extra>"+der1_label+"</extra>",
                    ),
                    row=2, col=1
                )
    
                # Subplot 3: second derivative (NaN at first two points; aligned at current t)
                fig.add_trace(
                    go.Scatter(
                        x=x_vals, y=plot_df_ds[der2_label],
                        mode="lines+markers" if show_markers else "lines",
                        name=der2_label, line=dict(width=line_width),
                        hovertemplate="%{x}<br>%{y}<extra>"+der2_label+"</extra>",
                    ),
                    row=3, col=1
                )
    
                # Layout & axes
                fig.update_layout(
                    hovermode="x unified" if unified_hover else "closest",
                    margin=dict(l=40, r=20, t=60, b=40),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                    showlegend=True,
                )
                fig.update_xaxes(
                    title=x_label if isinstance(work.index, pd.DatetimeIndex) else x_name,
                    showspikes=show_spikes, spikemode="across", spikesnap="cursor",
                    rangeslider=dict(visible=range_slider),
                    row=3, col=1
                )
                for r in [1, 2, 3]:
                    fig.update_yaxes(showline=True, zeroline=False, row=r, col=1)
    
                st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    
                # Download data exactly as plotted
                with st.expander("Download (derivatives data)"):
                    out = plot_df_ds.reset_index().rename(columns={"index": x_name})
                    st.download_button(
                        "Download CSV",
                        data=out.to_csv(index=False).encode("utf-8"),
                        file_name=f"derivatives_{y_col}.csv",
                        mime="text/csv",
                        help="Contains the series and its first/second derivatives (after local downsampling)."
                    )
    

    
    add_buy_me_coffee("https://paypal.me/jgoncalocouto/1")
