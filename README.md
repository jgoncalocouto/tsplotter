# TSPlotter

A Streamlit workspace for exploring time series with Plotly. Import a dataset, choose a time axis, filter rows, and explore one chart at a time.

Start with **Explore example data** to try 720 sensor readings without uploading a file. Supported imports: CSV, TSV, Excel, Parquet, JSON, and MATLAB v7.3 (`.mat`).

## Run locally

Requires Python 3.10 or newer. In PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m streamlit run main.py
```

Open the local URL printed by Streamlit. To use a different port, add `--server.port 8080`.

## Explore a dataset

1. Choose **Upload a file** or **Example dataset** in the sidebar.
2. Use **Absolute time** for timestamps, or **Relative time** for row numbers or another axis. Timestamp layouts, epoch units, and timezone conversion are available under **Time parsing & timezone**. The existing timezone is preserved by default.
3. Open **Filter data** to add numeric, datetime, or categorical conditions. Invalid conditions show an error before charts are calculated; **Clear filters** restores all prepared rows.
4. Choose **Trends**, **Distribution**, **Scatter**, **Grouped scatter**, **Rolling window**, or **Derivatives**. Primary signal controls stay visible; display settings are collapsed.
5. Open the chart's download panel for its plotted data, or **Data preview & export** for the full filtered dataset. The chart toolbar also exports a PNG.

The row metrics and preview reflect the active filters. Invalid timestamps are excluded with a warning. Chart-specific missing-value removal and sampling can reduce plotted rows; chart CSV exports match those plotted rows. Histogram exports contain bin counts. Chart selections survive switching views and reset when the dataset changes.

### Save and load plotting configurations

Open **Plot configuration** in the sidebar:

1. Set up your charts, then click **Save configuration** to download `tsplotter-config.json`.
2. In another session, choose the dataset, upload the saved JSON file under **Configuration file**, and click **Load configuration**.

The file remembers the selected chart, columns, panels, styling, palettes, marker shapes, numeric bins, performance limits, time parsing/timezones, filters, and summary selections. Settings from previously used chart views are included too. Plot zoom and pan are temporary interactions; use the chart controls to define the saved setup.

The data file is loaded separately. You can load a configuration first and choose the dataset afterwards. Saved columns are matched by name; missing columns or incompatible filters are skipped with notices, and an unavailable active time column falls back to row numbers. Bin intervals are recalculated from the current filtered data using the saved bin count or width.

Loading replaces the previous setup, including resetting options absent from the saved file to their defaults. An invalid or unsupported file leaves the current setup intact. Save is unavailable until a dataset is selected and any pending time setup is usable.

### Bin scatter groups

In **Grouped scatter**, select a numeric **Color by** or **Marker by** column, then choose its grouping mode:

- **Individual values** keeps each distinct value as a group.
- **Number of bins** splits the numeric range into the requested number of equal-width intervals.
- **Bin width** creates intervals of the specified width, starting at the smallest selected value. The last interval can extend beyond the largest value.

Color and marker settings are independent, including when both use the same column. Numeric color groups in the ordinary **Scatter** view also support binning; individual numeric values there retain the continuous color scale.

New numeric selections in **Grouped scatter** start with five bins when there are more than 50 distinct values. You can switch back to **Individual values** or adjust the bins. Points use one shared drawing trace so combinations of color and marker bins do not create thousands of traces.

Bins use complete, finite rows from the filtered dataset before chart downsampling. Intervals include the lower boundary and exclude the upper boundary, except the last interval includes both. Constant columns produce one group. **Grouping details** (or **Bin details** in Scatter) shows interval membership counts, including empty bins; chart legends show only bins with plotted points, ordered from low to high. There is a limit of 100 bins per grouping variable.

The chart CSV preserves original values and adds separate color/marker bin-label columns. Binning never changes the X or Y values.

## Tests

```powershell
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python -m pytest -q
```

Tests cover the filter engine and Streamlit workflows, including chart navigation, input validation, dataset resets, and consistency between filtered previews and CSV exports.

`main.py` coordinates loading and analysis, `workspace_ui.py` contains shared presentation helpers and the example dataset, `grouping.py` computes numeric intervals, `scatter_grouping.py` handles scatter grouping controls, `plot_config.py` validates portable configuration files, `config_ui.py` applies them to the dataset, and `filters/` separates filter controls from filtering logic.

## Docker

```sh
docker build -t tsplotter .
docker run --rm -p 8501:8501 tsplotter
```
