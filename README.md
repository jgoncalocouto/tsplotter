# TSPlotter

TSPlotter runs file reading, filtering and calculations on the user's computer with DuckDB-Wasm. Plotly supplies the familiar chart tools: zoom, pan, hover, legend selection and PNG export. The website server serves application assets; datasets are not uploaded to it.

This JavaScript application is the only maintained application in the repository. The website needs static file hosting; there is no separate analysis server or alternate application.

## Run locally

Use Node.js 20.19+ or 22.12+ and run from the repository root:

```sh
npm ci
npm run prepare:assets
npm run dev
```

Open the URL printed by Vite, usually `http://127.0.0.1:5173/`. On Windows PowerShell, use `npm.cmd` if execution policy blocks `npm.ps1`.

## Build and deploy

```sh
npm run build
```

Serve the generated `dist/` directory with a static web server. The build prepares the pinned local WebAssembly assets automatically and supports deployment under a subdirectory. To inspect the production build locally, run `npm run preview`; this preview server is for local verification.

The Docker image builds with Node and serves the finished files with Nginx:

```sh
docker build -t tsplotter .
docker run --rm -p 8080:8080 tsplotter
```

Open `http://localhost:8080/`. The container supports a `PORT` environment variable (default `8080`) and a `/healthz` endpoint. Only static hosting is required at runtime; file parsing and analysis remain on the user's computer. If an existing deployment used the previous `browser/` project directory, update its build root to the repository root and its publish directory to `dist/`.

## Open and explore a dataset

1. Open a local file, or choose **Explore example data**.
2. Select an absolute time column and its parsing options, or use a relative axis with row numbers or another source column.
3. Choose a view using the chart tabs: Trends, Distribution, Scatter, Rolling window or Derivatives. Search for signals in the selection controls; selected signals appear as removable chips.
4. Add filters if needed, then select **Update chart**.
5. Use Plotly's chart toolbar to zoom, pan or export an image. Download chart data or the full filtered dataset with the separate CSV actions.

Dataset, time-axis and saved-settings controls are in the sidebar; on a phone, open **Dataset & settings** to access the drawer. Filters and chart appearance expand below the signal controls. Changes to selections, filters and appearance display **Unapplied changes** until **Update chart** runs. Chart/image exports are disabled while those changes are pending; preview, statistics and full-dataset export apply the current settings first.

**Scatter** combines plain and grouped plots. Choose X and Y, then optionally assign color and marker columns. Numeric groups can use a bin count or interval width; unbinned numeric color can use a continuous scale alongside grouped markers. **Add scatter** creates another independent panel, and **Duplicate** copies the selected panel's settings. Keep up to six panels with their own axes, bins and appearance. Select a panel name or **Edit settings** to edit it, then **Update charts** to apply changes. Dataset filters apply to every panel. Each card provides its own CSV and PNG exports; cards stack on narrow screens.

**Distribution** also supports up to six independent panels. Use **Add distribution** or **Duplicate** to compare different signals or bin settings side by side. Each panel can overlay multiple signals using shared edges within that panel, with automatic bins, a fixed bin width or a target bin count. Select a panel to edit its title, signals and bins. Shared filters apply to every distribution; each card exports its own bin edges and counts as CSV or its chart as PNG. A panel without finite values shows an empty state while the others remain usable.

Every selected row is available to calculations and charts. There is no downsampling or automatic point limit. A histogram counts all observations in its bins. Scatter plots exclude rows with incomplete or nonfinite values in the selected axes/grouping columns; invalid timestamps cannot be plotted on an absolute time axis. Those exclusions are distinct from sampling.

Large datasets use the user's RAM, CPU and graphics resources. Browsers still have memory and rendering limits; retaining all points does not guarantee that an arbitrarily large chart can be rendered. Parquet permits local column/range reads and usually avoids the largest intermediate file conversions. Excel and MATLAB currently need to decode their selected data into browser memory before passing it to the analysis worker.

## Supported input formats

- **Parquet:** local Parquet files; only fields required by a query need to be scanned.
- **CSV / TSV:** delimited text with automatic detection; TSV uses tabs. UTF-8 text and UTF-16 files with a byte-order mark are supported. Convert other text encodings to UTF-8 if necessary. Parse failures are reported rather than silently skipping malformed rows.
- **JSON / JSONL:** row-oriented JSON records and newline-delimited records. Nested fields can be retained in the source/export, but the plotting controls principally target scalar numeric, time and categorical fields.
- **Excel `.xlsx` / `.xls`:** the first worksheet is selected by default. Expand **Import options** and enter another worksheet's exact name in **Excel worksheet** before opening the file, or choose **Apply worksheet** to reread the current workbook. The first row supplies column names. Duplicate and blank headings receive distinct names; internal blank rows are retained. Spreadsheet formulas are not executed; existing cached values are used.
- **MATLAB v7.3 `.mat`:** plain real numeric vectors and 2-D matrices stored in HDF5. MATLAB matrix dimensions are restored to their MATLAB row/column orientation. Only datasets with the same number of rows can form one table; a mismatch raises an error instead of discarding rows. Non-numeric/internal objects are excluded with a notice. Integers larger than JavaScript's exact integer range are rejected instead of rounded silently.

The MATLAB importer does **not** support MATLAB v5/v7 files, complex arrays, arbitrary cells/struct objects or timetable/table layouts encoded through object references such as `#refs#`. Export the desired MATLAB table to Parquet or CSV before opening it. A `.mat` extension alone does not establish a supported layout.

Opening a local file grants the page access to that selected file for the current session. Reloading the page clears the in-memory workspace; reopen the file to continue. There is no persistent dataset upload or server-side dataset history.

## Filters and time parsing

Filter bounds are inclusive. Conditions are combined in their displayed order, so `A OR B AND C` means `(A OR B) AND C`. Empty numeric bounds and empty categorical conditions are no-ops.

Numeric filters accept finite decimal/scientific notation. Datetime filters use their selected lower/upper local date and time. A **Time index** condition follows the configured absolute time column and its parsing settings. Categorical filters support exact membership, literal contains/starts/ends comparisons and regular expressions. Exact membership remains case-sensitive; the case option controls the string/pattern operations. Regular expressions use DuckDB’s RE2 syntax; lookaround and pattern backreferences are not supported. **Include missing** also matches missing values. Preview, summary and full export use the same configured filters and time preparation as the chart workspace.

Absolute time setup includes ISO/automatic parsing, explicit DMY/MDY/YMD layouts, Unix seconds/milliseconds, custom datetime formats and source/display timezone options. Check the selected profile when a text timestamp could be interpreted in more than one way. The app does not invent timestamps for invalid rows. Check your parsing settings and source data when rows cannot be used on an absolute time axis.

## Portable plot settings

**Save settings** downloads a JSON document using the TSPlotter version 2 configuration format. It includes every scatter and distribution panel, chart selections and appearance, numeric grouping bins, time parsing/timezones and filter clauses. It contains no dataset, local path, file handle, credentials or uploaded file bytes.

TSPlotter version 1 and 2 configurations can be loaded before or after opening their dataset, including version 1 files created before the JavaScript migration. Older Scatter and Grouped scatter settings become panels in the unified Scatter view; distinct saved plots are preserved as separate panels. Older histogram selections and bin settings become one Distribution panel. Column names must match the new dataset. Missing columns or incompatible saved filters are skipped with notices; a missing saved absolute time column falls back to relative row numbers. Invalid files are rejected before settings are applied. Unknown settings are ignored with notices, and unsupported versions or duplicate JSON fields are rejected.

Older `*_maxpts` fields may be retained for configuration compatibility but have no effect: the application always retains every selected point. Configurations can be reused across datasets, but results depend on the data and compatible column types. Save settings while the desired dataset/view is active; saving the configuration does not save that dataset.

## Develop and verify

```sh
npm ci
npm test
npm run build
npm run dev
```

Run these commands from the repository root.

`npm run build` prepares local DuckDB assets before creating `dist/`. For browser workflow checks, install Chromium once with `npx playwright install chromium`, keep the development server running at `http://127.0.0.1:5173/`, and run `npm run test:browser` in another terminal. Run `node tests/browser-scatter-panels.mjs` and `node tests/browser-distribution-panels.mjs` for independent panels, migration, per-panel exports and mobile layout. Set `BASE_URL` if using a different server URL; set `CHROMIUM_PATH` to use an existing Chromium executable. The configuration/file-format tests cover version 1 migration, version 2 panels, unsafe or ambiguous JSON, filter adapters, real Excel workbooks and numeric HDF5 reads.

Dependencies are pinned in `package-lock.json`. SheetJS is obtained from its official distribution; browser file reading does not call SheetJS or another hosted parsing service.

### Reproducible DuckDB assets

Use the Node version required by Vite (Node 20.19+ or 22.12+) and run `npm ci` before building. `npm run prepare:assets` copies the pinned DuckDB-Wasm runtime from `node_modules` and downloads its six official Parquet/JSON/ICU extensions for the MVP and EH targets. `npm run build` invokes this preparation automatically. Before a development-only run, prepare assets once and then use `npm run dev`.

[`scripts/duckdb-extensions.json`](scripts/duckdb-extensions.json) records the exact runtime/package versions, official HTTPS extension URLs, decoded file sizes and SHA256 hashes matching the deployed application. Node's fetch decodes HTTP content encoding (the official extension responses use Brotli); the recorded hashes describe the resulting WebAssembly files. Existing files are reused only after their size and hash match. Missing or modified extensions are downloaded to memory, verified, and installed with an atomic rename. A changed/failed response leaves existing files intact and fails the build.

The first preparation needs access to `extensions.duckdb.org`. Subsequent builds can work offline with `node_modules` and all verified generated assets already available. No extension is fetched by a user's browser from the official site: the built deployment serves these files itself. Changing a dependency version requires deliberately verifying/updating the asset manifest as well as the package lock.

`public/duckdb/`, `node_modules/`, `dist/`, generated Parquet fixtures and browser-test reports are ignored by Git. Commit the source, manifest and lockfile; do not commit these generated asset copies. `public/icon.svg` remains a source asset. The packaging tests cover offline cache reuse, missing assets, cache tampering, rejection of unexpected downloads and preservation of existing files on failure.
