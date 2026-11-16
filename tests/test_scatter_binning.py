"""Check numeric scatter grouping through the actual UI, figure, and CSV export."""

import io
import json
from pathlib import Path

import pandas as pd
import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
MAIN = Path(__file__).resolve().parents[1] / "main.py"
SMALL_DATA = (
    "pd.DataFrame({'timestamp': pd.date_range('2025-01-01', periods=10, freq='h'), "
    "'value': [9., 0., 8., 1., 7., 2., 6., 3., 5., 4.], "
    "'response': [19., 10., 18., 11., 17., 12., 16., 13., 15., 14.], "
    "'category': ['A', 'B'] * 5})"
)


def _app(chart="Grouped scatter", frame=SMALL_DATA):
    source = (
        "import streamlit as st\n"
        "import pandas as pd\n"
        "from pathlib import Path\n"
        "from unittest.mock import patch\n"
        "from workspace_ui import render_chart as original_render\n"
        "original_download = st.download_button\n"
        "def capture_download(*args, **kwargs):\n"
        "    if kwargs.get('key') in {'sc_download', 'sc2_download'}:\n"
        "        st.session_state['captured_scatter_csv'] = kwargs['data']\n"
        "    return original_download(*args, **kwargs)\n"
        "def capture_chart(fig, *args, **kwargs):\n"
        "    original_render(fig, *args, **kwargs)\n"
        "    st.session_state['captured_scatter_figure'] = fig.to_json()\n"
        f"frame = {frame}\n"
        "with patch('workspace_ui.example_dataset', return_value=frame), "
        "patch('workspace_ui.render_chart', capture_chart), "
        "patch.object(st, 'download_button', capture_download):\n"
        f"    exec(compile(Path({str(MAIN)!r}).read_text(encoding='utf-8'), {str(MAIN)!r}, 'exec'))\n"
    )
    app = AppTest.from_string(source)
    app.session_state["data_source"] = "Example dataset"
    app.session_state["plot_type"] = chart
    app.run(timeout=30)
    assert not app.exception
    return app


def _export(app):
    return pd.read_csv(io.BytesIO(app.session_state["captured_scatter_csv"]))


def _bin_column(frame, encoding):
    matches = [column for column in frame.columns if f"[{encoding} bin]" in column]
    assert len(matches) == 1
    return matches[0]


def _set_binning(app, prefix, mode, setting):
    app.selectbox(key=f"{prefix}_bin_mode").select(mode).run(timeout=30)
    suffix = "count" if mode == "Number of bins" else "width"
    app.number_input(key=f"{prefix}_bin_{suffix}").set_value(setting).run(timeout=30)
    assert not app.exception


def _assert_numeric_axes(app):
    chart = json.loads(app.session_state["captured_scatter_figure"])
    points = [trace for trace in chart["data"] if any(value is not None for value in trace.get("x", []))]
    assert points
    for trace in points:
        assert all(isinstance(value, (float, int)) for value in trace["x"])
        assert all(isinstance(value, (float, int)) for value in trace["y"])
    return chart


@pytest.mark.parametrize("encoding,group_key", [("color", "sc2_g1"), ("marker", "sc2_g2")])
@pytest.mark.parametrize("mode,setting,expected_count", [
    ("Number of bins", 3, 3), ("Bin width", 2.0, 5),
])
def test_each_grouped_encoding_can_bin_numeric_values_independently(encoding, group_key, mode, setting, expected_count):
    app = _app()
    app.selectbox(key=group_key).select("value").run(timeout=30)
    _set_binning(app, f"sc2_{encoding}", mode, setting)
    exported = _export(app)
    column = _bin_column(exported, encoding)
    assert exported[column].nunique() == expected_count
    assert exported["value"].tolist() == [9., 0., 8., 1., 7., 2., 6., 3., 5., 4.]
    assert exported["response"].tolist() == [19., 10., 18., 11., 17., 12., 16., 13., 15., 14.]
    assert exported[column].notna().all()
    _assert_numeric_axes(app)


def test_same_numeric_column_can_use_different_color_and_marker_bins():
    app = _app()
    app.selectbox(key="sc2_g1").select("value")
    app.selectbox(key="sc2_g2").select("value").run(timeout=30)
    _set_binning(app, "sc2_color", "Number of bins", 2)
    _set_binning(app, "sc2_marker", "Bin width", 2.0)
    exported = _export(app)
    color_column = _bin_column(exported, "color")
    marker_column = _bin_column(exported, "marker")
    assert color_column != marker_column
    assert exported[color_column].nunique() == 2
    assert exported[marker_column].nunique() == 5
    assert pd.api.types.is_numeric_dtype(exported["value"])
    chart = _assert_numeric_axes(app)
    # The original row order starts with the maximum; the legend still reads low to high.
    expected_colors = exported.groupby(color_column)["value"].min().sort_values().index.tolist()
    expected_markers = exported.groupby(marker_column)["value"].min().sort_values().index.tolist()
    color_legend = [trace["name"] for trace in chart["data"]
                    if trace.get("showlegend") and str(trace.get("legendgroup", "")).startswith("color_")]
    marker_legend = [trace["name"] for trace in chart["data"]
                     if trace.get("showlegend") and str(trace.get("legendgroup", "")).startswith("marker_")]
    assert color_legend == expected_colors
    assert marker_legend == expected_markers


@pytest.mark.parametrize("chart", ["Scatter", "Grouped scatter"])
def test_individual_numeric_groups_preserve_raw_axes_and_export_values(chart):
    app = _app(chart)
    if chart == "Scatter":
        app.selectbox(key="sc_group").select("value").run(timeout=30)
        assert app.selectbox(key="sc_color_bin_mode").value == "Individual values"
    else:
        app.selectbox(key="sc2_g1").select("value")
        app.selectbox(key="sc2_g2").select("value").run(timeout=30)
        assert app.selectbox(key="sc2_color_bin_mode").value == "Individual values"
        assert app.selectbox(key="sc2_marker_bin_mode").value == "Individual values"
    assert not app.exception
    exported = _export(app)
    assert pd.api.types.is_numeric_dtype(exported["value"])
    assert exported["value"].tolist() == [9., 0., 8., 1., 7., 2., 6., 3., 5., 4.]
    _assert_numeric_axes(app)


@pytest.mark.parametrize("mode,setting,expected_count", [
    ("Number of bins", 3, 3), ("Bin width", 2.0, 5),
])
def test_ordinary_scatter_can_switch_numeric_color_to_discrete_bins(mode, setting, expected_count):
    app = _app("Scatter")
    app.selectbox(key="sc_group").select("value").run(timeout=30)
    _set_binning(app, "sc_color", mode, setting)
    exported = _export(app)
    column = _bin_column(exported, "color")
    assert exported[column].nunique() == expected_count
    assert exported["value"].tolist() == [9., 0., 8., 1., 7., 2., 6., 3., 5., 4.]
    chart = _assert_numeric_axes(app)
    expected_order = exported.groupby(column)["value"].min().sort_values().index.tolist()
    assert [trace["name"] for trace in chart["data"] if trace.get("showlegend")] == expected_order


@pytest.mark.parametrize("chart", ["Scatter", "Grouped scatter"])
def test_categorical_groups_do_not_offer_numeric_binning(chart):
    app = _app(chart)
    if chart == "Scatter":
        app.selectbox(key="sc_group").select("category").run(timeout=30)
    else:
        app.selectbox(key="sc2_g1").select("category")
        app.selectbox(key="sc2_g2").select("category").run(timeout=30)
    assert not app.exception
    assert not any(str(widget.key).endswith("_bin_mode") for widget in app.selectbox)
    assert "category" in _export(app).columns


def test_group_bin_settings_persist_when_switching_chart_types():
    app = _app()
    app.selectbox(key="sc2_g1").select("value")
    app.selectbox(key="sc2_g2").select("response").run(timeout=30)
    _set_binning(app, "sc2_color", "Number of bins", 3)
    _set_binning(app, "sc2_marker", "Bin width", 2.0)
    app.radio(key="plot_type").set_value("Trends").run(timeout=30)
    app.radio(key="plot_type").set_value("Grouped scatter").run(timeout=30)
    assert not app.exception
    assert app.selectbox(key="sc2_g1").value == "value"
    assert app.selectbox(key="sc2_g2").value == "response"
    assert app.selectbox(key="sc2_color_bin_mode").value == "Number of bins"
    assert app.number_input(key="sc2_color_bin_count").value == 3
    assert app.selectbox(key="sc2_marker_bin_mode").value == "Bin width"
    assert app.number_input(key="sc2_marker_bin_width").value == 2.0


@pytest.mark.parametrize("invalid_width", [0.0, 1e-20])
def test_invalid_or_excessive_width_has_a_clear_recoverable_error(invalid_width):
    app = _app()
    app.selectbox(key="sc2_g1").select("value").run(timeout=30)
    _set_binning(app, "sc2_color", "Bin width", invalid_width)
    assert not app.exception
    assert app.error
    assert not app.get("plotly_chart")
    app.number_input(key="sc2_color_bin_width").set_value(2.0).run(timeout=30)
    assert not app.exception
    assert not app.error
    assert len(app.get("plotly_chart")) == 1
    assert _export(app)[_bin_column(_export(app), "color")].nunique() == 5


def test_derived_bin_column_does_not_overwrite_an_existing_axis_column():
    data = SMALL_DATA + ".assign(**{'value [color bin]': range(10)})"
    app = _app(frame=data)
    app.selectbox(key="sc2_y").select("value [color bin]")
    app.selectbox(key="sc2_g1").select("value").run(timeout=30)
    _set_binning(app, "sc2_color", "Number of bins", 3)
    exported = _export(app)
    assert exported["value [color bin]"].tolist() == list(range(10))
    derived = [column for column in exported.columns
               if column.startswith("value [color bin]") and column != "value [color bin]"]
    assert len(derived) == 1
    assert exported[derived[0]].nunique() == 3
    _assert_numeric_axes(app)


def test_constant_numeric_groups_have_one_complete_bin_for_each_encoding():
    app = _app(frame=SMALL_DATA + ".assign(value=5.0)")
    app.selectbox(key="sc2_g1").select("value")
    app.selectbox(key="sc2_g2").select("value").run(timeout=30)
    _set_binning(app, "sc2_color", "Number of bins", 3)
    _set_binning(app, "sc2_marker", "Bin width", 2.0)
    exported = _export(app)
    for encoding in ("color", "marker"):
        column = _bin_column(exported, encoding)
        assert exported[column].nunique() == 1
        assert exported[column].notna().all()
    assert not app.error
    assert len(exported) == 10
    _assert_numeric_axes(app)


@pytest.mark.parametrize("chart,group_key,prefix,slider_key", [
    ("Scatter", "sc_group", "sc_color", "sc_maxpts"),
    ("Grouped scatter", "sc2_g1", "sc2_color", "sc2_maxpts"),
])
def test_filtering_and_display_downsampling_keep_bin_assignments_consistent(chart, group_key, prefix, slider_key):
    data = (
        "pd.DataFrame({'timestamp': pd.date_range('2025-01-01', periods=2001, freq='min'), "
        "'value': range(2001), 'response': range(10, 2011)})"
    )
    app = _app(chart, data)
    app.selectbox(key=group_key).select("value").run(timeout=30)
    _set_binning(app, prefix, "Number of bins", 3)
    app.button(key="flt2_add").click().run(timeout=30)
    app.selectbox(key="flt2_var_0").select("value").run(timeout=30)
    app.text_input(key="flt2_low_num_0").input("123").run(timeout=30)
    complete = _export(app)
    label_column = _bin_column(complete, "color")
    assert len(complete) == 1878
    assert complete["value"].min() == 123
    app.slider(key=slider_key).set_value(1000).run(timeout=30)
    assert not app.exception
    sampled = _export(app)
    assert 0 < len(sampled) <= 1000
    expected = complete.set_index("value")[label_column]
    assert sampled[label_column].tolist() == expected.loc[sampled["value"]].tolist()


@pytest.mark.parametrize("chart", ["Scatter", "Grouped scatter"])
def test_sampling_does_not_reassign_colors_or_symbols_when_a_bin_disappears(chart):
    # Sampling skips row 1, which is the only observation in the lower bin.
    data = (
        "pd.DataFrame({'timestamp': pd.date_range('2025-01-01', periods=2001, freq='min'), "
        "'value': range(2001), 'response': range(10, 2011), "
        "'numeric_group': [100., 0.] + [100.] * 1999})"
    )
    app = _app(chart, data)
    if chart == "Scatter":
        app.selectbox(key="sc_group").select("numeric_group").run(timeout=30)
        _set_binning(app, "sc_color", "Number of bins", 2)
        slider_key = "sc_maxpts"
    else:
        app.selectbox(key="sc2_g1").select("numeric_group")
        app.selectbox(key="sc2_g2").select("numeric_group").run(timeout=30)
        _set_binning(app, "sc2_color", "Number of bins", 2)
        _set_binning(app, "sc2_marker", "Number of bins", 2)
        slider_key = "sc2_maxpts"

    def legend_styles():
        figure = json.loads(app.session_state["captured_scatter_figure"])
        return {
            (trace.get("legendgroup", ""), trace["name"]):
                (trace["marker"].get("color"), trace["marker"].get("symbol"))
            for trace in figure["data"] if trace.get("showlegend")
        }

    full_styles = legend_styles()
    app.slider(key=slider_key).set_value(1000).run(timeout=30)
    assert not app.exception
    sampled_styles = legend_styles()
    assert 0 < len(sampled_styles) < len(full_styles)
    for category, style in sampled_styles.items():
        assert full_styles[category] == style


@pytest.mark.parametrize("chart", ["Scatter", "Grouped scatter"])
def test_high_cardinality_defaults_are_readable_and_preserve_explicit_raw_choices(chart):
    data = (
        "pd.DataFrame({'timestamp': pd.date_range('2025-01-01', periods=101, freq='min'), "
        "'value': range(101), 'response': range(100, 201)})"
    )
    app = _app(chart, data)
    if chart == "Scatter":
        app.selectbox(key="sc_group").select("value").run(timeout=30)
        assert app.selectbox(key="sc_color_bin_mode").value == "Individual values"
        assert not any("[color bin]" in column for column in _export(app).columns)
    else:
        app.selectbox(key="sc2_g1").select("value")
        app.selectbox(key="sc2_g2").select("response").run(timeout=30)
        for encoding in ("color", "marker"):
            assert app.selectbox(key=f"sc2_{encoding}_bin_mode").value == "Number of bins"
            assert app.number_input(key=f"sc2_{encoding}_bin_count").value == 5
            assert _export(app)[_bin_column(_export(app), encoding)].nunique() == 5
            app.selectbox(key=f"sc2_{encoding}_bin_mode").select("Individual values").run(timeout=30)
        app.slider(key="sc2_msize").set_value(8).run(timeout=30)
        assert not app.exception
        assert app.selectbox(key="sc2_color_bin_mode").value == "Individual values"
        assert app.selectbox(key="sc2_marker_bin_mode").value == "Individual values"
        assert not any(" bin]" in column for column in _export(app).columns)
    _assert_numeric_axes(app)


def test_many_color_marker_combinations_use_one_point_trace_matching_exported_bins():
    data = (
        "pd.DataFrame({'timestamp': pd.date_range('2025-01-01', periods=10000, freq='min'), "
        "'value': range(10000), 'response': [i % 100 for i in range(10000)], "
        "'color_group': [i // 100 for i in range(10000)], "
        "'marker_group': [i % 100 for i in range(10000)]})"
    )
    app = _app(frame=data)
    app.selectbox(key="sc2_g1").select("color_group")
    app.selectbox(key="sc2_g2").select("marker_group").run(timeout=30)
    _set_binning(app, "sc2_color", "Number of bins", 100)
    _set_binning(app, "sc2_marker", "Number of bins", 100)
    exported = _export(app)
    color_column = _bin_column(exported, "color")
    marker_column = _bin_column(exported, "marker")
    assert len(exported[[color_column, marker_column]].drop_duplicates()) == 10000
    figure = _assert_numeric_axes(app)
    points = [trace for trace in figure["data"] if any(value is not None for value in trace.get("x", []))]
    assert len(points) == 1
    points = points[0]
    assert points["type"] == "scattergl"
    assert points["x"] == exported["value"].tolist()
    assert points["y"] == exported["response"].tolist()
    colors = {trace["name"]: trace["marker"]["color"] for trace in figure["data"]
              if trace.get("showlegend") and trace.get("legendgroup", "").startswith("color_")}
    symbols = {trace["name"]: trace["marker"]["symbol"] for trace in figure["data"]
               if trace.get("showlegend") and trace.get("legendgroup", "").startswith("marker_")}
    assert points["marker"]["color"] == [colors[label] for label in exported[color_column]]
    assert points["marker"]["symbol"] == [symbols[label] for label in exported[marker_column]]
    assert len(figure["data"]) == 1 + len(colors) + len(symbols)


def test_sampled_palette_collisions_are_reported_even_with_only_two_visible_groups():
    data = (
        "pd.DataFrame({'timestamp': pd.date_range('2025-01-01', periods=2001, freq='min'), "
        "'value': range(2001), 'response': range(10, 2011), "
        "'numeric_group': [0, 1, 2, 0, 3, 4, 0, 5, 6, 0, 7, 8, 0, 9, 10] "
        "+ [0 if i % 2 == 0 else 10 for i in range(15, 2001)]})"
    )
    app = _app(frame=data)
    app.selectbox(key="sc2_g1").select("numeric_group").run(timeout=30)
    app.slider(key="sc2_maxpts").set_value(1000).run(timeout=30)
    assert not app.exception
    assert _export(app)["numeric_group"].nunique() == 2
    assert any("color" in warning.value.lower() and "share" in warning.value.lower()
               for warning in app.warning)
    figure = json.loads(app.session_state["captured_scatter_figure"])
    colors = [trace["marker"]["color"] for trace in figure["data"]
              if trace.get("showlegend") and trace.get("legendgroup", "").startswith("color_")]
    assert len(colors) == 2
    assert len(set(colors)) == 1
