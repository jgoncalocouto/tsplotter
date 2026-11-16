import numpy as np
import pandas as pd
import pytest

from grouping import bin_numeric_groups


def test_count_mode_uses_equal_width_bins_with_clear_boundary_membership():
    values = pd.Series([0., 1., 2., 3., 4.], index=list("abcde"), name="signal")
    result = bin_numeric_groups(values, mode="Number of bins", bins=2)
    assert result.edges.tolist() == [0, 2, 4]
    assert result.labels == ["[0, 2)", "[2, 4]"]
    assert result.groups.tolist() == ["[0, 2)", "[0, 2)", "[2, 4]", "[2, 4]", "[2, 4]"]
    assert result.groups.index.equals(values.index)
    assert result.groups.name == "signal"
    assert result.groups.cat.ordered
    assert result.counts.tolist() == [2, 3]
    assert result.effective_width == 2
    pd.testing.assert_series_equal(values, pd.Series([0., 1., 2., 3., 4.], index=list("abcde"), name="signal"))


def test_width_mode_keeps_requested_width_and_covers_final_partial_interval():
    result = bin_numeric_groups(pd.Series([-3., -1., 1., 2.]), mode="Bin width", width=2)
    assert result.edges.tolist() == [-3, -1, 1, 3]
    assert result.groups.tolist() == ["[-3, -1)", "[-1, 1)", "[1, 3]", "[1, 3]"]
    assert result.effective_width == 2


def test_width_mode_exact_maximum_is_in_last_bin_without_empty_extra_bin():
    result = bin_numeric_groups(pd.Series([.1, .2, .3, .4]), mode="Bin width", width=.1)
    assert len(result.labels) == 3
    assert result.groups.notna().all()
    assert result.groups.iloc[-1] == result.labels[-1]
    assert result.edges[0] == .1
    assert result.edges[-1] >= .4


@pytest.mark.parametrize("mode,kwargs", [
    ("Bin width", {"width": .1}),
    ("Number of bins", {"bins": 3}),
])
def test_decimal_internal_boundaries_start_the_bin_shown_by_the_label(mode, kwargs):
    result = bin_numeric_groups(pd.Series([.1, .2, .3, .4]), mode=mode, **kwargs)
    assert result.edges.tolist() == [.1, .2, .3, .4]
    assert result.labels == ["[0.1, 0.2)", "[0.2, 0.3)", "[0.3, 0.4]"]
    assert result.groups.tolist() == ["[0.1, 0.2)", "[0.2, 0.3)", "[0.3, 0.4]", "[0.3, 0.4]"]


def test_repeating_decimal_edges_have_round_trip_labels_and_consistent_membership():
    boundary = 1 / 3
    result = bin_numeric_groups(pd.Series([0., boundary, 1.]), mode="Number of bins", bins=3)
    assert result.groups.iloc[1] == result.labels[1]
    for i, label in enumerate(result.labels):
        low, high = label[1:-1].split(", ")
        assert float(low) == result.edges[i]
        assert float(high) == result.edges[i + 1]


def test_empty_intervals_are_retained_in_order_with_zero_counts():
    result = bin_numeric_groups(pd.Series([0, 10, 0, 10]), mode="Number of bins", bins=5)
    assert len(result.labels) == 5
    assert result.counts.tolist() == [2, 0, 0, 0, 2]
    assert list(result.groups.cat.categories) == result.labels
    assert list(result.groups.cat.remove_unused_categories().cat.categories) == [result.labels[0], result.labels[-1]]


@pytest.mark.parametrize("dtype", ["Float64", "Int64"])
def test_nullable_numeric_values_keep_row_identity_and_missing_values(dtype):
    series = pd.Series([0, None, 2, 4], dtype=dtype, index=[7, 7, 9, 11])
    result = bin_numeric_groups(series, mode="Number of bins", bins=2)
    assert result.groups.index.tolist() == [7, 7, 9, 11]
    assert pd.isna(result.groups.iloc[1])
    assert result.counts.sum() == 3


def test_nonfinite_values_are_unassigned_and_not_used_for_edges():
    result = bin_numeric_groups(pd.Series([np.nan, -np.inf, 1., 3., np.inf]), mode="Number of bins", bins=2)
    assert result.edges.tolist() == [1, 2, 3]
    assert result.groups.isna().tolist() == [True, True, False, False, True]
    assert result.counts.sum() == 2


@pytest.mark.parametrize("mode,kwargs", [("Number of bins", {"bins": 5}), ("Bin width", {"width": 2.})])
def test_constant_numeric_values_form_one_exact_value_group(mode, kwargs):
    result = bin_numeric_groups(pd.Series([5., np.nan, 5.]), mode=mode, **kwargs)
    assert result.labels == ["5"]
    assert result.counts.tolist() == [2]
    assert result.groups.iloc[0] == "5"
    assert result.effective_width == 0


def test_constant_decimal_has_a_readable_round_trip_label():
    result = bin_numeric_groups(pd.Series([1.1, 1.1]), mode="Number of bins")
    assert result.labels == ["1.1"]
    assert float(result.labels[0]) == 1.1


@pytest.mark.parametrize("width", [0., -1., np.nan, np.inf, -np.inf, None, True])
def test_width_must_be_finite_and_positive(width):
    with pytest.raises(ValueError, match="width"):
        bin_numeric_groups(pd.Series([0., 1.]), mode="Bin width", width=width)


@pytest.mark.parametrize("count", [0, -1, 101, 2.5, True])
def test_bin_count_must_be_a_whole_number_within_the_limit(count):
    with pytest.raises(ValueError, match="whole number"):
        bin_numeric_groups(pd.Series([0., 1.]), mode="Number of bins", bins=count)


def test_tiny_width_is_rejected_before_allocating_bins():
    with pytest.raises(ValueError, match="larger"):
        bin_numeric_groups(pd.Series([0., 1.]), mode="Bin width", width=1e-300)


@pytest.mark.parametrize("series", [pd.Series([np.nan, np.inf, -np.inf]), pd.Series([], dtype=float)])
def test_no_finite_values_has_a_clear_error(series):
    with pytest.raises(ValueError, match="no finite"):
        bin_numeric_groups(series, mode="Number of bins")


@pytest.mark.parametrize("series", [pd.Series(["1", "2"]), pd.Series([True, False]), pd.Series([1+2j, 3+4j])])
def test_only_real_numeric_columns_can_be_binned(series):
    with pytest.raises(ValueError, match="numeric column"):
        bin_numeric_groups(series, mode="Number of bins")


def test_large_offsets_get_distinct_interval_labels_in_numeric_order():
    result = bin_numeric_groups(pd.Series([1e12, 1e12 + 1]), mode="Number of bins", bins=4)
    assert len(result.labels) == 4
    assert len(set(result.labels)) == 4
    assert result.counts.tolist() == [1, 0, 0, 1]
    assert result.labels[0] == "[1000000000000, 1000000000000.25)"


def test_reported_width_uses_the_same_decimal_range_as_count_mode_edges():
    result = bin_numeric_groups(pd.Series([1e12 + .1, 1e12 + .2]), mode="Number of bins", bins=2)
    assert result.effective_width == .05
    assert result.edges.tolist() == [1e12 + .1, 1e12 + .15, 1e12 + .2]


@pytest.mark.parametrize("values,kwargs", [
    ([1., np.nextafter(1., 2.)], {"mode": "Number of bins", "bins": 5}),
    ([-1e308, 1e308], {"mode": "Number of bins", "bins": 2}),
    ([2**53, 2**53 + 1], {"mode": "Number of bins", "bins": 2}),
    ([1e308, np.nextafter(1e308, np.inf)], {"mode": "Bin width", "width": 1e308}),
])
def test_unrepresentable_intervals_fail_explicitly(values, kwargs):
    with pytest.raises(ValueError, match="accurately|represent"):
        bin_numeric_groups(pd.Series(values), **kwargs)
