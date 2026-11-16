"""Numeric interval groups shared by the scatter views."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, localcontext
import math
from numbers import Integral, Real

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NumericBins:
    groups: pd.Series
    edges: np.ndarray
    labels: list[str]
    counts: pd.Series
    effective_width: float


def _edge_labels(edges: np.ndarray) -> list[str]:
    # A displayed boundary must round-trip to the actual edge: merely distinct
    # rounded labels can contradict membership for values near that boundary.
    for precision in (12, 15, 17):
        labels = [format(float(edge), f".{precision}g") for edge in edges]
        if (len(set(labels)) == len(labels)
                and all(float(label) == edge for label, edge in zip(labels, edges))):
            # repr is sometimes shorter than .17g while preserving the same
            # exact float (for example 0.3333333333333333).
            return [min((label, repr(float(edge))), key=len) for label, edge in zip(labels, edges)]
    raise ValueError("These intervals are too narrow to label accurately. Use fewer bins or a larger bin width.")


def _result(series: pd.Series, values: np.ndarray, finite: np.ndarray,
            edges: np.ndarray, labels: list[str], width: float) -> NumericBins:
    codes = np.full(len(series), -1, dtype=np.int64)
    # The last boundary is inclusive; internal boundaries start a new bin.
    assigned = np.searchsorted(edges, values[finite], side="right") - 1
    codes[finite] = np.minimum(assigned, len(labels) - 1)
    groups = pd.Series(pd.Categorical.from_codes(codes, categories=labels, ordered=True),
                       index=series.index, name=series.name)
    counts = pd.Series(np.bincount(codes[codes >= 0], minlength=len(labels)),
                       index=pd.Index(labels, name="bin"), name="count")
    return NumericBins(groups, edges, labels, counts, width)


def bin_numeric_groups(
    series: pd.Series, *, mode: str, bins: int = 5, width: float | None = None,
    max_bins: int = 100,
) -> NumericBins:
    """Group finite numeric values into ordered, equal-width intervals.

    Bins start at the finite minimum. They include their lower boundary and
    exclude their upper boundary, except that the last bin includes its upper
    boundary. Width mode retains the requested width, including in the last
    interval, which can extend beyond the observed maximum. Missing and infinite
    values are unassigned. Constant values form one exact-value group.

    Returned counts include empty intervals. Edges and groups are computed from
    the complete supplied series, so callers can sample rows afterwards without
    changing the meaning or ordering of the groups.
    """
    if (not pd.api.types.is_numeric_dtype(series.dtype)
            or pd.api.types.is_bool_dtype(series.dtype)
            or pd.api.types.is_complex_dtype(series.dtype)):
        raise ValueError("Binning requires a numeric column.")
    if isinstance(max_bins, bool) or not isinstance(max_bins, Integral) or max_bins < 1:
        raise ValueError("The bin limit must be a positive whole number.")
    if mode not in {"Number of bins", "Bin width"}:
        raise ValueError("Choose Number of bins or Bin width.")
    if mode == "Number of bins":
        if isinstance(bins, bool) or not isinstance(bins, Integral) or not 1 <= bins <= max_bins:
            raise ValueError(f"Choose a whole number of bins between 1 and {max_bins}.")
    elif (isinstance(width, bool) or not isinstance(width, Real)
          or not math.isfinite(width) or width <= 0):
        raise ValueError("Bin width must be a finite number greater than zero.")

    if pd.api.types.is_integer_dtype(series.dtype):
        # Float conversion must not silently merge distinct large integers.
        large = series.dropna()
        large = large[(large >= 2**53) | (large <= -(2**53))]
        if any(int(value) != int(float(value)) for value in large.unique()):
            raise ValueError("These integer values are too large to bin accurately. Scale the grouping column first.")
    values = series.to_numpy(dtype=np.float64, na_value=np.nan)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("This column has no finite numeric values to bin.")
    low, high = float(values[finite].min()), float(values[finite].max())
    if low == high:
        # repr keeps the shortest value that round-trips without exposing
        # artifacts such as 1.1000000000000001 for a constant 1.1 signal.
        label = repr(low)
        if label.endswith(".0"):
            label = label[:-2]
        if low == 0:
            label = "0"
        return _result(series, values, finite, np.array([low, high]), [label], 0.0)
    span = high - low
    if not math.isfinite(span):
        raise ValueError("This numeric range is too large to bin accurately. Scale the grouping column first.")

    decimal_low, decimal_high = Decimal(str(low)), Decimal(str(high))
    decimal_inputs = [decimal_low, decimal_high]
    if mode == "Bin width":
        decimal_inputs.append(Decimal(str(float(width))))
    # Decimal inputs describe the shortest round-trip values the user sees.
    # Build edges in that domain before the one float conversion, so repeated
    # steps of 0.1 produce an edge of 0.3 rather than 0.30000000000000004.
    # Include the exponent spread to preserve narrow values at large offsets
    # and very small endpoints in otherwise broad ranges.
    precision = max(50, max(value.adjusted() for value in decimal_inputs)
                    - min(value.as_tuple().exponent for value in decimal_inputs) + 30)
    with localcontext() as context:
        context.prec = precision
        decimal_span = decimal_high - decimal_low
        if mode == "Number of bins":
            count = int(bins)
            decimal_width = decimal_span / count
        else:
            decimal_width = decimal_inputs[-1]
            count_decimal = (decimal_span / decimal_width).to_integral_value(rounding=ROUND_CEILING)
            if count_decimal > max_bins:
                raise ValueError(f"This width would create more than {max_bins} bins. Use a larger bin width.")
            count = max(1, int(count_decimal))
        effective_width = float(decimal_width)
        edges = np.array([float(decimal_low + i * decimal_width) for i in range(count + 1)])
        if mode == "Number of bins":
            # Repeating decimal widths may round at the context precision;
            # the requested range still ends at the exact observed maximum.
            edges[-1] = high

    if (not np.isfinite(edges).all() or effective_width <= 0
            or not np.all(edges[1:] > edges[:-1])):
        raise ValueError("These intervals are too narrow or too large to represent accurately. Use fewer bins, a larger width, or scale the grouping column.")
    boundaries = _edge_labels(edges)
    labels = [f"[{boundaries[i]}, {boundaries[i + 1]}{']' if i == len(edges) - 2 else ')'}"
              for i in range(len(edges) - 1)]
    return _result(series, values, finite, edges, labels, effective_width)
