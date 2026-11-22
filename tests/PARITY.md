# Browser workspace acceptance checks

These checks define the JavaScript application's behavior. Browser charts must
retain every usable point. Older sampling fields imported from configuration
files must be ignored with a notice; there must be no row-stride, point cap, or hidden sampling.
Preview pagination is presentation only and must not affect analysis or exports.

## Data and preparation

- CSV/TSV delimiter detection includes comma, semicolon, tab and pipe. The
  importer handles UTF-8 and UTF-16 with a byte-order mark. Convert other text
  encodings to UTF-8 before importing. Quoted delimiters/newlines are data, not
  row breaks.
- Parquet must preserve numeric values, nulls, timestamps and row counts. A
  pandas datetime index can exist in Parquet metadata rather than an ordinary
  visible column; make this usable as the time axis.
- JSON supports records, JSONL and column-oriented objects. Excel uses the first
  sheet by default and allows selecting another. MATLAB v7.3 supports the plain
  numeric vector/matrix layouts documented in the README; unsupported layouts
  must produce an error explaining how to export a compatible table.
- Auto-detection must not treat ordinary numeric measurements as nanosecond
  timestamps. Numeric epochs require explicit seconds/milliseconds selection.
- Absolute-time preparation excludes invalid timestamps with a count, sorts by
  timestamp, and removes the time source from Y signals. Relative row numbers
  start at zero. An explicit relative axis is preserved and sorted.
- Preserve timezone handling, DMY/MDY/YMD/custom layouts and epoch units; do not
  silently reinterpret ambiguous dates according to the browser locale.

## Filters

- Bounds are inclusive. Empty conditions are no-ops. Conditions combine in
  visible left-to-right order; the first condition's AND/OR choice is ignored.
- Numeric invalid/reversed bounds and invalid regular expressions produce
  recoverable errors before showing results. Zero matches is a valid empty view.
- Categorical operations: membership, inverse membership, contains, inverse
  contains, starts with, ends with, regex; case sensitivity and null matching.
- Preview, summary metrics, all chart preparations and full-data export use the
  same filtered row set. Changing dataset resets stale column/view state.

## Five chart views

Every view uses the same Add/Duplicate/Remove panel lifecycle and selected editor.
Panels keep independent settings and exports; filters remain global by design.

1. **Trends:** 1–6 independent panels with signal lists; markers,
   six line shapes, line width, log/linear scale, range slider, unified hover and
   spikes. Retain gaps. SVG line simplification must be disabled when used.
2. **Distribution:** 1–6 independent titled panels with add/duplicate/remove and
   one selected settings editor. Each has its own signals, bin settings and
   CSV/PNG exports; overlays use shared edges within that panel. Automatic
   Freedman–Diaconis width with Scott/Sturges fallbacks, fixed width and target
   count. Final edge is inclusive. CSV exports edges, centers, counts and bin
   metadata rather than raw samples. Shared filters affect every panel; empty
   or all-nonfinite panels clear their output without blocking valid siblings.
3. **Scatter:** one view for plain and grouped plots, with 1–6 independent
   panels and one selected settings editor. Add, duplicate and remove panels;
   keep each panel's title, X/Y, color, markers, bins and appearance independent.
   Continuous numeric color works alongside category markers. Color and marker
   may use the same numeric source with different bins. Binning preserves raw
   axes and exports raw values plus distinct derived labels. Legends follow
   category/bin order and toggle the corresponding point groups. Exclude only
   rows missing/nonfinite in selected axes/group variables. Shared filters apply
   to every panel; each card exports its own data/image and stacks on mobile.
4. **Rolling window:** 1–6 independent analyses of trailing min/mean/max measured in rows, current-row
   inclusion flag, full-window flag (`min_periods=window` versus 1), min/max band,
   raw markers and hover/range options. Excluding current shifts the input by
   exactly one row before the rolling calculation. Nulls do not become zeros.
5. **Derivatives:** 1–6 analyses, each containing three aligned plots for raw/first/second backward differences.
   First derivative begins with null; second begins with two nulls. Respect
   seconds/minutes/hours/days/sample spacing or explicit numeric X. A zero delta
   produces null. Optional trailing rolling-mean smoothing changes derivatives
   only, preserving leading nulls. Keep raw signal unchanged.

## Numeric groups

- Fit edges on complete finite selected rows after filters by default, not on a
  preview. Full-dataset fitting keeps boundaries fixed across filters without
  changing which observations are plotted. Distribution can fit shared edges
  over all panels' selected columns using the first panel's bin settings.
- Intervals begin at finite minimum and are lower inclusive / upper exclusive,
  except final upper edge inclusive. Width mode retains requested interval width
  so the final edge can exceed the maximum. Constant values form one group.
- Counts include empty bins; chart legends only include occupied bins. Maximum
  100 bins per encoding. Narrow/unrepresentable intervals fail explicitly.
- Decimal boundaries such as 0.3 must not drift into the preceding interval.
- Keep original values at full precision. Integers beyond JavaScript's exact
  53-bit range must not silently collapse into the same value.

## Comparisons, configuration and export

- Category colors/symbols and source-series colors remain stable across filters
  and panel ordering. Optional axis linking shares ranges/zoom/reset only for
  compatible variables, scales and derivative units; it performs no data queries.
- Save/load is versioned and data-free. Version 3 preserves every panel
  and inactive chart as well as the current view. Accept v1/v2 settings,
  migrating distinct Scatter and Grouped scatter plots into independent panels
  and old histogram/analysis settings into their respective panels;
  ignore all legacy `*_maxpts` without sampling.
- Missing columns fall back or skip with notices. Loading defaults clears prior
  overrides. Invalid config leaves previous valid state untouched.
- Configuration can be loaded before the dataset. Data is always selected
  separately. Saving must not include source file bytes, private content, caches,
  user paths or credentials.
- Valid inactive bin preferences survive saving; invalid unused numeric fields
  may reset. Local recovery stores only portable settings and the saved-file
  baseline, never dataset bytes or source file metadata. Invalid drafts retain
  the last valid recovery. Clearing recovery persists until a later edit.
- Unsaved settings and unapplied chart changes are distinct states. Preview,
  summary and full export use current global filters/time preparation without
  rendering, validating or applying unfinished chart settings.
- Full export covers the full filtered data. Chart export matches exactly what
  is represented; bins are exported as separate columns, never overwritten axes.
- Native Plotly zoom, pan, hover, legend toggling and PNG export remain available.

## Row accounting

- Matched rows counts global filters before time preparation. Filtered out
  counts only filter exclusions; invalid time is separate.
- Every chart provides source/matched/filteredOut/invalidTime/usable/excludedValues.
  Source equals filteredOut + invalidTime + excludedValues + usable.
- Multi-signal charts use the union of usable rows and report finite value
  counts per signal. Scatter requires all selected axes/group values. Derivative
  leading nulls are expected outputs, not excluded source values. Histogram bin
  counts and preview limits must never be presented as source-row counts.

## Privacy, hosting and runtime

- Selecting and processing a file sends no POST/PUT/PATCH, file content, rows or
  application telemetry over HTTP/WebSockets. Serve dependencies locally.
- Repeated view switches reuse/purge the Plotly graph so WebGL contexts do not
  accumulate. A failed import leaves a usable application, not a blank screen.
- Test >35,000 and >200,000 rows, beyond former limits; inspect Plotly trace
  arrays, row metrics and exports for exact point counts and endpoints.
- Test at least one real compressed Parquet file, not only CSV, and inspect all
  network requests while reading/filtering/plotting it.

Primary Plotly references: [line.simplify](https://plotly.com/javascript/reference/scatter/#scatter-line-simplify),
[WebGL context limits](https://plotly.com/javascript/webgl-vs-svg/),
[Plotly.react](https://plotly.com/javascript/plotlyjs-function-reference/#plotlyreact).
