"""Exercise the actual analysis workflow and its import/plotting boundaries."""

import ast
import io
from pathlib import Path

import pandas as pd
import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main.py"


@pytest.fixture(scope="module")
def helpers():
    """Load pure helpers without starting the app or relying on UI session state."""
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    names = {"_read_table_from_bytes", "_maybe_downsample_for_plot", "_guess_datetime_col"}
    body = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            body.append(node)
    namespace = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(MAIN), "exec"), namespace)
    return namespace


def _example_app():
    app = AppTest.from_file(str(MAIN)).run(timeout=30)
    app.button(key="empty_example").click().run(timeout=30)
    assert not app.exception
    return app


def test_empty_state_example_and_return_to_upload():
    app = AppTest.from_file(str(MAIN)).run(timeout=30)
    assert not app.exception
    assert not app.get("plotly_chart")
    assert app.button(key="empty_example")
    app.button(key="empty_example").click().run(timeout=30)
    assert not app.exception
    assert app.metric[0].value == "720"
    assert app.radio(key="setup_time_mode").value == "Absolute time"
    assert app.selectbox(key="setup_out_tz").value == "(keep)"
    app.radio(key="data_source").set_value("Upload a file").run(timeout=30)
    assert not app.exception
    assert app.session_state["raw_df"] is None
    assert not app.get("plotly_chart")
    assert app.button(key="empty_example")


@pytest.mark.parametrize("chart", [
    "Trends", "Distribution", "Scatter", "Grouped scatter", "Rolling window", "Derivatives",
])
def test_each_chart_type_renders_one_figure(chart):
    app = _example_app()
    app.radio(key="plot_type").set_value(chart).run(timeout=30)
    assert not app.exception
    assert len(app.get("plotly_chart")) == 1
    assert any(element.key == "dataset_download"
               for element in app.get("download_button"))


def test_chart_selections_persist_when_switching_views():
    app = _example_app()
    app.multiselect(key="mlc_cols_0").set_value(["humidity"]).run(timeout=30)
    app.radio(key="plot_type").set_value("Scatter").run(timeout=30)
    app.radio(key="plot_type").set_value("Trends").run(timeout=30)
    assert not app.exception
    assert app.multiselect(key="mlc_cols_0").value == ["humidity"]


@pytest.mark.parametrize("chart,prefix", [("Scatter", "sc"), ("Grouped scatter", "sc2")])
def test_scatter_allows_the_same_column_as_an_axis_and_group(chart, prefix):
    app = _example_app()
    app.radio(key="plot_type").set_value(chart).run(timeout=30)
    app.selectbox(key=f"{prefix}_x").select("temperature")
    app.selectbox(key=f"{prefix}_y").select("temperature")
    group_key = "sc_group" if prefix == "sc" else "sc2_g1"
    app.selectbox(key=group_key).select("temperature").run(timeout=30)
    assert not app.exception
    assert len(app.get("plotly_chart")) == 1


def test_filtered_preview_and_dataset_export_contain_the_same_rows():
    # Capture the actual bytes handed to Streamlit, before its media endpoint.
    source = (
        "import streamlit as st\n"
        "from pathlib import Path\n"
        "from unittest.mock import patch\n"
        "original_download = st.download_button\n"
        "def capture_download(*args, **kwargs):\n"
        "    if kwargs.get('key') == 'dataset_download':\n"
        "        st.session_state['exported_csv'] = kwargs['data']\n"
        "    return original_download(*args, **kwargs)\n"
        "with patch.object(st, 'download_button', capture_download):\n"
        f"    exec(compile(Path({str(MAIN)!r}).read_text(encoding='utf-8'), {str(MAIN)!r}, 'exec'))\n"
    )
    app = AppTest.from_string(source)
    app.session_state["data_source"] = "Example dataset"
    app.run(timeout=30)
    app.button(key="flt2_add").click().run(timeout=30)
    app.selectbox(key="flt2_var_0").select("temperature").run(timeout=30)
    app.text_input(key="flt2_low_num_0").input("23").run(timeout=30)
    assert not app.exception
    preview = app.dataframe[0].value
    exported = pd.read_csv(io.BytesIO(app.session_state["exported_csv"]))
    assert 0 < len(preview) < 720
    assert preview["temperature"].min() >= 23
    assert exported["temperature"].min() >= 23
    assert len(exported) == len(preview) == int(app.metric[0].value.replace(",", ""))
    assert int(app.metric[3].value.replace(",", "")) == 720 - len(preview)
    app.button(key="flt2_clear").click().run(timeout=30)
    assert not app.exception
    assert len(app.dataframe[0].value) == 720


def test_numeric_only_data_defaults_to_relative_time():
    source = (
        "import pandas as pd\n"
        "from pathlib import Path\n"
        "from unittest.mock import patch\n"
        "with patch('workspace_ui.example_dataset', return_value=pd.DataFrame({'signal': [1., 2., 3.]})):\n"
        f"    exec(compile(Path({str(MAIN)!r}).read_text(encoding='utf-8'), {str(MAIN)!r}, 'exec'))\n"
    )
    app = AppTest.from_string(source)
    app.session_state["data_source"] = "Example dataset"
    app.run(timeout=30)
    assert not app.exception
    assert app.radio(key="setup_time_mode").value == "Relative time"
    assert len(app.get("plotly_chart")) == 1
    assert app.metric[0].value == "3"


@pytest.mark.parametrize("name,delimiter,encoding", [
    ("table.csv", ",", "utf-8-sig"),
    ("table.csv", ";", "cp1252"),
    ("table.tsv", "\t", "utf-16"),
    ("table.csv", "|", "utf-8"),
])
def test_import_detects_delimiters_and_windows_encodings(helpers, name, delimiter, encoding):
    text = delimiter.join(["signal", "label"]) + "\n" + delimiter.join(["1.5", "café"]) + "\n"
    frame, metadata = helpers["_read_table_from_bytes"](name, text.encode(encoding), None)
    assert frame.columns.tolist() == ["signal", "label"]
    assert frame.loc[0, "signal"] == 1.5
    assert frame.loc[0, "label"] == "café"
    assert metadata is None


@pytest.mark.parametrize("payload", [
    '[{"signal": 1.5, "label": "alpha"}, {"signal": 2.5, "label": "beta"}]',
    '{"signal": 1.5, "label": "alpha"}\n{"signal": 2.5, "label": "beta"}\n',
    '{"signal": [1.5, 2.5], "label": ["alpha", "beta"]}',
])
def test_json_records_jsonl_and_column_oriented_data_produce_the_same_table(helpers, payload):
    frame, metadata = helpers["_read_table_from_bytes"]("table.json", payload.encode("utf-8"), None)
    expected = pd.DataFrame({"signal": [1.5, 2.5], "label": ["alpha", "beta"]})
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), expected)
    assert metadata is None


@pytest.mark.parametrize("row_count", [30_000, 30_001, 59_999, 120_005])
def test_downsampling_honors_point_limit_and_preserves_row_identity(helpers, row_count):
    frame = pd.DataFrame({"signal": range(row_count)}, index=pd.RangeIndex(row_count, name="sample"))
    sampled = helpers["_maybe_downsample_for_plot"](frame, max_points=30_000)
    assert len(sampled) <= 30_000
    assert sampled.index.name == "sample"
    assert sampled["signal"].tolist() == sampled.index.tolist()
    assert len(frame) == row_count
