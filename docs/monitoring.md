# Monitoring

Daily scoring of the published D-1 forecasts against actuals, and drift
detection (two-sided CUSUM on the model-vs-RTE daily MAE gap, k=0.5, h=5,
standardized). Generated daily by the `forecast` GitHub Action
(`scripts/monitor.py`) into `docs/data/monitoring/*.json`, fetched here
client-side — a JSON push updates this page without a rebuild.

<div id="monitoring-status">Loading…</div>

## Drift — CUSUM on the model-vs-RTE gap

<div id="monitoring-cusum">Loading…</div>

## Daily MAE (MW)

<div id="monitoring-mae">Loading…</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script src="js/hexaload.js"></script>
<script>
  renderMonitoring(
    document.getElementById("monitoring-status"),
    document.getElementById("monitoring-cusum"),
    document.getElementById("monitoring-mae")
  );
</script>
