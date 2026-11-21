import io
import pandas as pd
import pytest
from hosting_safety import read_parquet_checked, preview_frame, export_allowed


def test_parquet_preserves_values_and_datetime_index():
    frame = pd.DataFrame({'signal': [1., 2., 3.]}, index=pd.date_range('2026-01-01', periods=3, name='time'))
    data = io.BytesIO()
    frame.to_parquet(data)
    data.seek(0)
    pd.testing.assert_frame_equal(read_parquet_checked(data), frame, check_freq=False)


def test_preview_is_bounded_without_changing_full_data():
    frame = pd.DataFrame(1., index=range(2000), columns=range(60))
    preview = preview_frame(frame)
    assert preview.shape == (1000, 50)
    assert frame.shape == (2000, 60)


def test_large_exports_are_available_on_request():
    assert export_allowed(pd.DataFrame(1., index=range(20000), columns=range(60)))


def test_large_dataset_does_not_eagerly_build_csv_or_send_full_preview():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    main = Path(__file__).resolve().parents[1] / 'main.py'
    source = (
        "import pandas as pd\nfrom pathlib import Path\nfrom unittest.mock import patch\n"
        "frame = pd.DataFrame({'signal': range(20000), 'other': range(20000)})\n"
        "original_csv = pd.DataFrame.to_csv\n"
        "def checked_csv(self, *args, **kwargs):\n"
        "    assert len(self) <= 1000, 'full dataset serialized without request'\n"
        "    return original_csv(self, *args, **kwargs)\n"
        "with patch('workspace_ui.example_dataset', return_value=frame), patch.object(pd.DataFrame, 'to_csv', checked_csv):\n"
        f"    exec(compile(Path({str(main)!r}).read_text(), {str(main)!r}, 'exec'))\n"
    )
    app = AppTest.from_string(source)
    app.session_state['data_source'] = 'Example dataset'
    # Distribution avoids an unrelated chart-specific CSV export for this test.
    app.session_state['plot_type'] = 'Distribution'
    app.run(timeout=30)
    assert not app.exception
    assert len(app.dataframe[0].value) == 1000
    assert app.button(key='dataset_prepare_download')
    assert app.metric[0].value == '20,000'
