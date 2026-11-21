# Browser migration acceptance checks

The original Streamlit application remains the behavioral reference except that
browser charts must retain every usable point. Legacy sampling settings must be
ignored with a notice; there must be no row-stride, point cap, or hidden sampling.
Preview pagination is presentation only and must not affect analysis or exports.

## Data and preparation

- CSV/TSV delimiter detection includes comma, semicolon, tab and pipe. Original
  readers handle UTF-8 BOM, UTF-16, CP1252 and Latin-1. Quoted delimiters/newlines
  are data, not row breaks.
- Parquet must preserve numeric values, nulls, timestamps and row counts. A
  pandas datetime index can exist in Parquet metadata rather than an ordinary
  visible column; make this usable as the time axis.
- JSON supports records, JSONL and column-oriented objects. Excel uses the first
  sheet by default and allows selecting another. MATLAB v7.3 is part of the
  original import contract; unsupported import types must say so clearly.
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

## Six chart views

1. **Trends:** 1–6 synchronized-X panels with independent signal lists; markers,
   six line shapes, line width, log/linear scale, range slider, unified hover and
   spikes. Retain gaps. SVG line simplification must be disabled when used.
2. **Distribution:** multi-signal overlaid counts using shared edges; automatic
   Freedman–Diaconis width with Scott/Sturges fallbacks, fixed width and target
   count. Final edge is inclusive. CSV exports edges, centers, counts and bin
   metadata rather than raw samples.
3. **Scatter:** arbitrary X/Y including same source; optional continuous numeric
   or categorical color, numeric color bins, palettes, marker size and opacity.
   Exclude only rows missing/nonfinite in the selected axes/group variables.
4. **Grouped scatter:** independent color and marker columns/modes, palettes and
   custom marker sequences. Color and marker may use the same numeric source
   with different bins. Binning preserves original axes and exports raw values
   plus distinct derived labels. Legend order follows category/bin order.
5. **Rolling window:** trailing min/mean/max measured in rows, current-row
   inclusion flag, full-window flag (`min_periods=window` versus 1), min/max band,
   raw markers and hover/range options. Excluding current shifts the input by
   exactly one row before the rolling calculation. Nulls do not become zeros.
6. **Derivatives:** three aligned panels for raw/first/second backward differences.
   First derivative begins with null; second begins with two nulls. Respect
   seconds/minutes/hours/days/sample spacing or explicit numeric X. A zero delta
   produces null. Optional trailing rolling-mean smoothing changes derivatives
   only, preserving leading nulls. Keep raw signal unchanged.

## Numeric groups

- Fit edges on complete finite selected rows after filters, not on a preview.
- Intervals begin at finite minimum and are lower inclusive / upper exclusive,
  except final upper edge inclusive. Width mode retains requested interval width
  so the final edge can exceed the maximum. Constant values form one group.
- Counts include empty bins; chart legends only include occupied bins. Maximum
  100 bins per encoding. Narrow/unrepresentable intervals fail explicitly.
- Decimal boundaries such as 0.3 must not drift into the preceding interval.
- Keep original values at full precision. Integers beyond JavaScript's exact
  53-bit range must not silently collapse into the same value.

## Configuration and export

- Save/load is versioned and data-free. Restore inactive charts as well as the
  current view. Accept original v1 configuration keys or clearly communicate
  unsupported settings; ignore all legacy `*_maxpts` without sampling.
- Missing columns fall back or skip with notices. Loading defaults clears prior
  overrides. Invalid config leaves previous valid state untouched.
- Configuration can be loaded before the dataset. Data is always selected
  separately. Saving must not include source file bytes, private content, caches,
  user paths or credentials.
- Full export covers the full filtered data. Chart export matches exactly what
  is represented; bins are exported as separate columns, never overwritten axes.
- Native Plotly zoom, pan, hover, legend toggling and PNG export remain available.

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
