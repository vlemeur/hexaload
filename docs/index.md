# hexaload

National French electricity load, forecast **day-ahead (D-1)** from RTE's open
data (eco2mix national, hourly, UTC) — and benchmarked against **RTE's own D-1
forecast**, walk-forward only: 36 folds over 2023→2026, sliding 365-day train
window, no random split, anti-leakage by construction.

<div class="stat-cards">
  <div class="stat"><div class="stat-num">−54%</div><div class="stat-label">MASE vs RTE's D-1 over 36 folds (Timestral 0.237 vs RTE 0.516)</div></div>
  <div class="stat"><div class="stat-num">0.156</div><div class="stat-label">MASE on the 2026 hold-out — 11 folds, scored once, tuned params</div></div>
  <div class="stat"><div class="stat-num">289 MW</div><div class="stat-label">CRPS on the 2026 hold-out (Timestral), 90% intervals included</div></div>
</div>

## Results on 2026 (hold-out)

<div id="holdout-table">Loading…</div>

Full 36-fold tables (tuned and constructor defaults), tuning studies, and the
metric formulas live on the [Methodology](method.md) page.

## How it works

<div class="card-grid">
  <div class="card"><h3>1 · Collect</h3><p>RTE eco2mix consumption (hourly, UTC) + Open-Meteo weather forecasts, cached on disk. Only the daily job goes to the network.</p></div>
  <div class="card"><h3>2 · Freeze</h3><p>Each fold uses only what was known at its origin: target-reading features are masked delay-aware; weather enters only as archived forecasts, never observations.</p></div>
  <div class="card"><h3>3 · Forecast</h3><p>Baselines → ridge / HGB / XGB → AutoETS / Theta → deep from scratch (MLP, RNN, TCN, N-HiTS, PatchTST) → zero-shot foundation models (Timestral, Chronos-Bolt), all with 19-quantile intervals.</p></div>
  <div class="card"><h3>4 · Judge</h3><p>Walk-forward is the only judge, RTE prevision_j1 the absolute reference, the 2026 hold-out scored once.</p></div>
</div>

## Forecast — tomorrow (D-1)

Published daily by the `forecast` GitHub Action (~11:50 Paris). This section
reads a JSON file client-side, so a daily push updates it without a rebuild.

<div id="forecast-charts">Loading…</div>

## Accuracy — the last 7 days, each forecast made at D-1

For each of the last 7 days: the same D-1 forecast (h=24) the models would
have published the day before at 12:00 Paris, against actuals and RTE's D-1
forecast. Feature models (ridge/HGB/XGB) are predicted hour by hour over the
12 h gap before the target day, feeding their own predictions back as lags —
the same recursive scheme the daily job uses. RTE's D-1 forecast is published
only later on J-1, so where it is not yet available at the freeze, the models
use the 24 h load lag as its proxy.

<div id="accuracy-chart">Loading…</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script src="js/hexaload.js"></script>
<script>
  renderHoldout(document.getElementById("holdout-table"));
  renderForecast(document.getElementById("forecast-charts"));
  renderAccuracy(document.getElementById("accuracy-chart"));
</script>
