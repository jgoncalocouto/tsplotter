"""Grouping controls and derived columns shared by the scatter views."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
import streamlit as st

from grouping import NumericBins, bin_numeric_groups


@dataclass(frozen=True)
class GroupSettings:
    source: str
    encoding: str
    mode: str = "Individual values"
    bins: int = 5
    width: float | None = None


@dataclass
class PreparedGroup:
    column: str
    settings: GroupSettings
    categories: list[str]
    binning: NumericBins | None = None


def render_group_controls(frame: pd.DataFrame, column: str, *, encoding: str,
                          key_prefix: str, discrete: bool = True) -> GroupSettings | None:
    """Keep binning settings beside the selected numeric grouping column."""
    source_key = f"{key_prefix}_bin_source"
    if st.session_state.get(source_key) != column:
        for suffix in ("mode", "count", "width"):
            st.session_state.pop(f"{key_prefix}_bin_{suffix}", None)
        st.session_state[source_key] = column
    if column == "(none)":
        return None
    settings = GroupSettings(source=column, encoding=encoding)
    series = frame[column]
    count = series.nunique(dropna=True)
    if (not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_complex_dtype(series)):
        st.caption(f"{count:,} categories · grouped by individual values")
        return settings

    mode = st.selectbox(
        f"{encoding.title()} grouping", ["Individual values", "Number of bins", "Bin width"],
        index=1 if discrete and count > 50 else 0, key=f"{key_prefix}_bin_mode",
        help="Keep the original values, split the numeric range into equal-width bins, or choose an interval width.",
    )
    if mode == "Individual values":
        st.caption(f"{count:,} unique values · choose bins to combine nearby values")
        return settings
    if count > 50:
        st.caption(f"{count:,} unique values · intervals keep the groups readable. Individual values remains available.")

    if mode == "Number of bins":
        bins = st.number_input(
            f"{encoding.title()} bin count", min_value=1, max_value=100, value=5, step=1,
            key=f"{key_prefix}_bin_count", help="Equal-width intervals spanning the selected data range.",
        )
        return GroupSettings(column, encoding, mode, int(bins))

    values = series.to_numpy(dtype=float, na_value=np.nan)
    values = values[np.isfinite(values)]
    with np.errstate(over="ignore", invalid="ignore"):
        span = float(values.max() - values.min()) if len(values) else 0.0
    default_width = span / 5 if math.isfinite(span) and span > 0 else 1.0
    if default_width <= 0:
        default_width = 1.0
    width = st.number_input(
        f"{encoding.title()} bin width", min_value=0.0, value=default_width, format="%.8g",
        key=f"{key_prefix}_bin_width",
        help="Must be greater than zero. Intervals start at the smallest value in the selected data; at most 100 bins.",
    )
    return GroupSettings(column, encoding, mode, width=float(width))


def complete_scatter_data(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Drop incomplete/nonfinite selected rows without changing the source table."""
    result = frame[list(dict.fromkeys(columns))].copy()
    numeric = result.select_dtypes(include=[np.number]).columns
    result[numeric] = result[numeric].replace([np.inf, -np.inf], np.nan)
    return result.dropna()


def prepare_group(frame: pd.DataFrame, settings: GroupSettings | None, *,
                  discrete: bool = True) -> PreparedGroup | None:
    """Append a separate grouping column, preserving raw axes and source values."""
    if settings is None:
        return None
    series = frame[settings.source]
    if settings.mode == "Individual values":
        if not discrete:
            return PreparedGroup(settings.source, settings, [])
        values = series.astype(str)
        categories = list(pd.unique(values))
        binning = None
        suffix = "group"
    else:
        binning = bin_numeric_groups(series, mode=settings.mode, bins=settings.bins, width=settings.width)
        values = binning.groups
        categories = list(values.cat.remove_unused_categories().cat.categories)
        suffix = "bin"
    base = f"{settings.source} [{settings.encoding} {suffix}]"
    column = base
    number = 2
    while column in frame.columns:
        column = f"{base} {number}"
        number += 1
    frame[column] = values
    return PreparedGroup(column, settings, categories, binning)


def group_categories(frame: pd.DataFrame, group: PreparedGroup | None) -> list[str]:
    """Order occupied legend items by the full dataset's bin/category order."""
    if group is None:
        return ["All"]
    present = set(frame[group.column].dropna())
    return [category for category in group.categories if category in present]


def chart_export(frame: pd.DataFrame, groups: list[PreparedGroup | None]) -> pd.DataFrame:
    """Retain raw columns plus bin labels; omit raw-value grouping copies."""
    copies = [group.column for group in groups if group is not None
              and group.binning is None and group.column != group.settings.source]
    return frame.drop(columns=copies)


def render_bin_details(groups: list[PreparedGroup | None]) -> None:
    """Explain bin membership using counts before chart downsampling."""
    for group in groups:
        if group is None or group.binning is None:
            continue
        result = group.binning
        st.markdown(f"**{group.settings.encoding.title()} bins · {group.settings.source}**")
        if result.effective_width == 0:
            st.caption("One group: all selected values are equal · counts before downsampling")
        else:
            st.caption(f"{len(result.labels)} intervals · width {result.effective_width:.8g} · counts before downsampling")
        st.dataframe(pd.DataFrame({"Interval": result.labels, "Rows": result.counts.to_numpy()}),
                     hide_index=True, use_container_width=True)
    if any(group is not None and group.binning is not None for group in groups):
        st.caption("Intervals include their lower edge. Only the last interval also includes its upper edge. Empty bins are omitted from the chart legend.")
