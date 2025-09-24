# matloader_v73.py
# Reusable loader for MATLAB -v7.3 (HDF5) .mat files with "#refs#" layout.
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import h5py
import os

# --- optional debug log ---
DEBUG_EVENTS: List[Dict[str, Any]] = []

def _dbg(event: str, **info):
    try:
        payload = {"event": event}
        for k, v in info.items():
            if isinstance(v, np.ndarray):
                payload[k] = {"dtype": str(v.dtype), "shape": list(v.shape)}
            else:
                payload[k] = v
        DEBUG_EVENTS.append(payload)
    except Exception:
        pass

def get_debug_log(clear: bool = False) -> List[Dict[str, Any]]:
    global DEBUG_EVENTS
    log = list(DEBUG_EVENTS)
    if clear:
        DEBUG_EVENTS = []
    return log

# --- small helpers ---
def _decode_char_dataset(ds: h5py.Dataset) -> str:
    data = np.array(ds[()])
    return "".join(chr(int(c)) for c in data.flatten(order="F") if int(c) != 0)

def _coerce_column_vector(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim == 2 and 1 in a.shape:
        return a.reshape(-1, order="F")
    return a.reshape(-1)

def _matlab_datenum_to_datetime(arr: np.ndarray) -> pd.DatetimeIndex:
    arr = np.asarray(arr, dtype=float)
    return pd.to_datetime(arr - 719529, unit="D", origin="1970-01-01")

def _index_from_time_vector(a: np.ndarray) -> pd.Index:
    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.datetime64): return pd.DatetimeIndex(a)
    if np.issubdtype(a.dtype, np.timedelta64): return pd.TimedeltaIndex(a)
    vals = a.astype(np.float64).reshape(-1)
    if vals.size == 0: return pd.Index(vals, name="t")
    vmax = float(np.nanmax(vals))
    def _try(unit):
        try: return pd.DatetimeIndex(pd.to_datetime(vals, unit=unit, origin="unix"))
        except Exception: return None
    if 1e9 <= vmax < 1e10:   # seconds
        idx = _try("s");  return idx if idx is not None else pd.Index(vals, name="t")
    if 1e12 <= vmax < 1e13:  # milliseconds
        idx = _try("ms"); return idx if idx is not None else pd.Index(vals, name="t")
    if 1e15 <= vmax < 1e16:  # microseconds
        idx = _try("us"); return idx if idx is not None else pd.Index(vals, name="t")
    if 1e18 <= vmax < 1e19:  # nanoseconds
        idx = _try("ns"); return idx if idx is not None else pd.Index(vals, name="t")
    if 5e4 <= vmax < 1e6:    # MATLAB datenum
        try: return _matlab_datenum_to_datetime(vals)
        except Exception: pass
    return pd.Index(vals, name="t")

# --- robust v7.3 (HDF5) detection ---
_HDF5_SIG = b"\x89HDF\r\n\x1a\n"

def _is_hdf5_mat(path: str) -> bool:
    """
    MATLAB -v7.3 MAT-files start with a textual header ("MATLAB 7.3 MAT-file ...")
    and the HDF5 signature typically appears at offset 512. Accept either:
      - signature at 0 (pure HDF5)
      - signature at 512 (MATLAB 7.3)
    As a last check, try opening with h5py.
    """
    try:
        with open(path, "rb") as f:
            head0 = f.read(8)
            if head0 == _HDF5_SIG:
                return True
            f.seek(512)
            head512 = f.read(8)
            if head512 == _HDF5_SIG:
                return True
    except Exception:
        pass
    # fallback probe: can h5py open it?
    try:
        with h5py.File(path, "r"):
            return True
    except Exception:
        return False

# --- v7.3 patterns ---
def _collect_refs_float_arrays(path: str, group_path: str = "#refs#") -> Dict[str, np.ndarray]:
    buf: Dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        if group_path not in f: return {}
        g = f[group_path]
        for name, obj in g.items():
            if hasattr(obj, "dtype") and hasattr(obj, "shape") and len(obj.shape) in (1,2) and obj.dtype.kind == "f":
                buf[str(name)] = np.array(obj[()]).reshape(-1)
    if not buf: return {}
    lengths = pd.Series({k: len(v) for k, v in buf.items()})
    mode_len = int(lengths.mode().iloc[0])
    aligned = {k: v for k, v in buf.items() if len(v) == mode_len}
    _dbg("refs:aligned_sets", count=len(aligned), length=mode_len, names=sorted(aligned.keys())[:12])
    return aligned

def _guess_time_unit(arr: np.ndarray) -> Tuple[str, Optional[pd.DatetimeIndex], bool]:
    a = np.asarray(arr, dtype=float)
    if a.size == 0 or np.any(~np.isfinite(a)): return ("unknown", None, False)
    vmax = float(np.nanmax(a)); mono = bool(np.all(np.diff(a) >= -1e-12))
    if 5e4 <= vmax < 1e6:
        try: return ("matlab_datenum_days", pd.to_datetime(a - 719529, unit="D", origin="1970-01-01"), mono)
        except Exception: pass
    if 1e9 <= vmax < 1e10:
        try: return ("unix_seconds", pd.to_datetime(a, unit="s", origin="unix"), mono)
        except Exception: pass
    if 1e12 <= vmax < 1e13:
        try: return ("unix_milliseconds", pd.to_datetime(a, unit="ms", origin="unix"), mono)
        except Exception: pass
    if 1e15 <= vmax < 1e16:
        try: return ("unix_microseconds", pd.to_datetime(a, unit="us", origin="unix"), mono)
        except Exception: pass
    if 1e18 <= vmax < 1e19:
        try: return ("unix_nanoseconds", pd.to_datetime(a, unit="ns", origin="unix"), mono)
        except Exception: pass
    return ("unknown", None, mono)

def _choose_time_vector(cands: Dict[str, np.ndarray]) -> Tuple[str, pd.Index, str]:
    rows = []
    for name, arr in sorted(cands.items()):
        unit, dtidx, mono = _guess_time_unit(arr)
        rows.append({"dataset": name, "unit": unit, "monotonic": mono, "has_dt": dtidx is not None})
    df = pd.DataFrame(rows)
    order = ["matlab_datenum_days", "unix_seconds", "unix_milliseconds", "unix_microseconds", "unix_nanoseconds"]
    c1 = df[(df["has_dt"]) & (df["monotonic"])]
    if not c1.empty:
        c1 = c1.assign(rank=c1["unit"].apply(lambda u: order.index(u) if u in order else 99))
        best_name = c1.sort_values(["rank","dataset"]).iloc[0]["dataset"]
    else:
        best_name = df.iloc[0]["dataset"]
    arr = cands[best_name]
    unit, dtidx, _ = _guess_time_unit(arr)
    idx = pd.DatetimeIndex(dtidx) if dtidx is not None else _index_from_time_vector(arr)
    _dbg("refs:time_choice", chosen_dataset=best_name, unit=unit)
    return best_name, idx, unit

def _build_df_from_refs_d(path: str, group_path: str = "#refs#") -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
    with h5py.File(path, "r") as f:
        if group_path not in f or "d" not in f[group_path]: return None
        d = f[group_path]["d"]
        if ("varNames" not in d and "VariableNames" not in d) or ("data" not in d and "Data" not in d):
            return None
        vnames_ds = d["varNames"] if "varNames" in d else d["VariableNames"]
        data_ds   = d["data"]     if "data"     in d else d["Data"]

        # names
        varnames: List[str] = []
        for ref in np.asarray(vnames_ds[()]).flatten():
            if isinstance(ref, h5py.Reference) and ref: varnames.append(_decode_char_dataset(f[ref]))
            else: varnames.append(None)

        # arrays
        targets: List[np.ndarray] = []
        for ref in np.asarray(data_ds[()]).flatten():
            if isinstance(ref, h5py.Reference) and ref: targets.append(np.array(f[ref][()]).reshape(-1))
            else: targets.append(None)

        # time selection
        aligned = _collect_refs_float_arrays(path, group_path)
        if not aligned: return None
        time_key, idx, unit = _choose_time_vector(aligned)

        # assemble DF
        N = len(idx)
        cols = {nm: arr for nm, arr in zip(varnames, targets) if (nm is not None and arr is not None and len(arr) == N)}
        df = pd.DataFrame(cols, index=idx)

        # index naming
        try:
            label = _decode_char_dataset(f[group_path]["T"]).strip() if "T" in f[group_path] else "Time"
        except Exception:
            label = "Time"
        if isinstance(df.index, (pd.DatetimeIndex, pd.TimedeltaIndex)): df.index.name = label

        meta = {"time_dataset": time_key, "time_unit": unit, "n_cols": len(df.columns), "index_name": label}
        _dbg("refs_d:built", **meta)
        return df, meta

def _build_df_from_refs_block(path: str, group_path: str = "#refs#") -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
    aligned = _collect_refs_float_arrays(path, group_path)
    if not aligned: return None
    time_key, idx, unit = _choose_time_vector(aligned)
    N = len(idx)
    cols = {k: v for k, v in aligned.items() if k != time_key and len(v) == N}
    df = pd.DataFrame(cols, index=idx)
    if isinstance(df.index, (pd.DatetimeIndex, pd.TimedeltaIndex)): df.index.name = "Time"
    meta = {"time_dataset": time_key, "time_unit": unit, "n_cols": len(df.columns), "index_name": df.index.name}
    _dbg("refs_block:built", **meta)
    return df, meta

def _h5_try_read_timetable_like_group(h5: h5py.File, grp: h5py.Group) -> Optional[Tuple[List[str], List[np.ndarray], Optional[np.ndarray]]]:
    name_fields = ["VariableNames", "varNames", "VarNames", "vars", "Names"]
    data_fields = ["Data", "data", "Variables", "variables", "_data"]
    time_fields = ["rowTimes", "RowTimes", "Time", "times"]

    # names
    names: Optional[List[str]] = None
    for nf in name_fields:
        if nf in grp:
            node = grp[nf]
            if isinstance(node, h5py.Dataset):
                refs = node[()]
                out = []
                for ref in np.asarray(refs).flatten():
                    if isinstance(ref, h5py.Reference) and ref:
                        data = np.array(h5[ref][()])
                        out.append("".join(chr(int(c)) for c in data.flatten(order="F") if int(c) != 0))
                if out: names = out
            if names: break

    # data columns
    cols: List[np.ndarray] = []
    found = False
    for df in data_fields:
        if df in grp:
            found = True
            node = grp[df]
            if isinstance(node, h5py.Dataset) and node.dtype.kind == "O":
                for ref in np.asarray(node[()]).flatten():
                    if isinstance(ref, h5py.Reference) and ref:
                        cols.append(_coerce_column_vector(np.array(h5[ref][()])))
            elif isinstance(node, h5py.Group):
                for _, ds in node.items():
                    if isinstance(ds, h5py.Dataset):
                        cols.append(_coerce_column_vector(np.array(ds[()])))
            elif isinstance(node, h5py.Dataset):
                arr = np.array(node[()])
                if arr.ndim == 2:
                    for i in range(arr.shape[1]): cols.append(_coerce_column_vector(arr[:, i]))
                else:
                    cols.append(_coerce_column_vector(arr))
            break
    if not cols: return None if not found else None

    # time vector
    tvec = None
    for tf in time_fields:
        if tf in grp:
            try:
                tvec = _coerce_column_vector(np.array(grp[tf][()]))
            except Exception:
                pass
            break

    return names or [], cols, tvec

def _build_df_from_objects_anywhere(path: str) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
    best = None
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            nonlocal best
            if not isinstance(obj, h5py.Group): return
            parsed = _h5_try_read_timetable_like_group(f, obj)
            if not parsed: return
            names, arrays, tvec = parsed
            if arrays:
                rows = len(arrays[0]); cols = len(arrays)
                if best is None or (rows, cols) > (best[0], best[1]):
                    best = (rows, cols, name, names, arrays, tvec)
        f.visititems(visit)

    if best is None: return None

    rows, cols, h5path, names, arrays, tvec = best
    idx = _index_from_time_vector(tvec) if tvec is not None else pd.RangeIndex(start=0, stop=rows, name="t")
    data = {}
    if names and len(names) == len(arrays):
        for nm, arr in zip(names, arrays):
            data[str(nm) if nm is not None else "Var"] = _coerce_column_vector(arr)
    else:
        for i, arr in enumerate(arrays, 1):
            data[f"Var{i}"] = _coerce_column_vector(arr)
    df = pd.DataFrame(data, index=idx)
    if isinstance(df.index, (pd.DatetimeIndex, pd.TimedeltaIndex)): df.index.name = "Time"
    meta = {"source": "objects_scan", "h5path": h5path, "rows": rows, "cols": len(df.columns)}
    _dbg("objects_scan:built", **meta)
    return df, meta

# --- public API ---
def read_mat_v73(path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load a MATLAB -v7.3 .mat file and return (DataFrame, meta).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if not _is_hdf5_mat(path):
        raise ValueError("Only MATLAB -v7.3 (HDF5) .mat files are supported by this loader.")

    # Preferred: #refs#/d mapping
    res = _build_df_from_refs_d(path, "#refs#")
    if res is not None:
        df, meta = res; return df, {"kind": "refs_d", **meta}
    # Fallback: aligned datasets under #refs#
    res = _build_df_from_refs_block(path, "#refs#")
    if res is not None:
        df, meta = res; return df, {"kind": "refs_block", **meta}
    # Last: scan objects anywhere
    res = _build_df_from_objects_anywhere(path)
    if res is not None:
        df, meta = res; return df, {"kind": "objects_scan", **meta}
    raise ValueError("Could not discover a timetable/table in this -v7.3 file under #refs#.")
