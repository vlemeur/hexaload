/* Shared client-side data loaders/renderers for hexaload.
 * All JSON is fetched from the repo (raw.githubusercontent) so that a daily JSON push
 * updates the site WITHOUT a rebuild. */
const HEXA_RAW = "https://raw.githubusercontent.com/vlemeur/hexaload/main/docs/data";

/* Local preview (zensical serve): the built site ships docs/data/ relatively,
 * so try the relative path first. In production, try the raw repo first so a
 * JSON push updates the page without a rebuild, then fall back to the copy
 * baked into the built site. */
function hexaCandidates(path) {
  const local = ["localhost", "127.0.0.1"].includes(location.hostname);
  const rel = `data/${path}`;
  const raw = `${HEXA_RAW}/${path}`;
  return local ? [rel, raw] : [raw, rel];
}

async function hexaFetch(path) {
  let lastErr;
  for (const url of hexaCandidates(path)) {
    try {
      const r = await fetch(url, { cache: "no-store" });
      if (!r.ok) throw new Error(String(r.status));
      return await r.json();
    } catch (e) { lastErr = e; }
  }
  throw lastErr;
}

/* What each model is allowed to see (anti-leakage contract in method.md). */
function hexaInputs(name) {
  if (name.endsWith("_norte"))
    return "exogenous + load lags — NO RTE forecast as input";
  if (name === "rte_prevision_j1") return "RTE's own D-1 forecast (the reference)";
  if (name === "timestral_209m" || name === "chronos_bolt")
    return "exogenous only: calendar + archived weather forecasts — never RTE's forecast";
  if (name === "hgb_features" || name === "xgb_features" || name === "ridge_features"
      || name.startsWith("hgb_features_") || name === "ens_features")
    return "exogenous + load lags + RTE's D-1 forecast (published input)";
  if (name.startsWith("seasonal_naive") || name === "naive") return "load history only";
  if (name.startsWith("sktime_")) return "load history only";
  return "exogenous + load lags";
}

function hexaTable(rows, cols) {
  // cols: [key, header, decimals|null]
  const head = "<tr>" + cols.map(c => `<th>${c[1]}</th>`).join("") + "</tr>";
  const body = rows.map(r =>
    "<tr>" + cols.map(([k, , dec]) => {
      const v = r[k];
      const txt = v == null ? "—" : (dec == null ? String(v) : Number(v).toFixed(dec));
      return `<td>${txt}</td>`;
    }).join("") + "</tr>"
  ).join("");
  return `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

/* ---- Results on 2026 (hold-out table) ---- */
async function renderHoldout(el) {
  try {
    const d = await hexaFetch("results.json");
    if (!d.holdout) throw new Error("no holdout");
    const rows = d.holdout
      .slice()
      .sort((a, b) => a.mase_mean - b.mase_mean);
    el.innerHTML = '<div class="holdout-cards">' + rows.map((r, i) =>
      `<div class="stat holdout-card${i === 0 ? " best" : ""}">
        <div class="holdout-model">${r.model}</div>
        <div class="stat-num">${Number(r.mase_mean).toFixed(3)}</div>
        <div class="stat-label">MASE · CRPS ${Number(r.crps_mean).toFixed(0)} MW · ${r.mase_count} folds<br><span class="holdout-inputs">${hexaInputs(r.model)}</span></div>
      </div>`).join("") + "</div>";
  } catch (e) {
    el.innerHTML = "<p><em>Results not published yet.</em></p>";
  }
}

/* ---- Full tables on the Methodology page ---- */
async function renderFullTables(el) {
  try {
    const d = await hexaFetch("results.json");
    let html = "";
    if (d.backtest_tuned) {
      const t = d.backtest_tuned.slice().sort((a, b) => a.mase_mean - b.mase_mean);
      html += "<h3>D-1 backtest, 36 folds (2023→2026), tuned — RTE prevision_j1 = reference</h3>"
        + hexaTable(t.map(r => ({ ...r, inputs: hexaInputs(r.model) })),
          [["model", "model", null], ["inputs", "inputs", null], ["mae_mean", "MAE", 0], ["rmse_mean", "RMSE", 0], ["mase_mean", "MASE", 3], ["crps_mean", "CRPS", 0]]);
    }
    if (d.backtest_default) {
      const t = d.backtest_default.slice().sort((a, b) => a.mase_mean - b.mase_mean);
      html += "<h3>D-1 backtest, constructor defaults</h3>"
        + hexaTable(t.map(r => ({ ...r, inputs: hexaInputs(r.model) })),
          [["model", "model", null], ["inputs", "inputs", null], ["mase_mean", "MASE", 3], ["crps_mean", "CRPS", 0]]);
    }
    if (d.tuning && Object.keys(d.tuning).length) {
      const rows = Object.entries(d.tuning).map(([model, s]) => ({ model, ...s }))
        .sort((a, b) => (a.best_mase ?? 9) - (b.best_mase ?? 9));
      html += "<h3>Tuning studies (Optuna NSGA-II, MASE + CRPS)</h3>"
        + hexaTable(rows, [["model", "model", null], ["trials", "trials", 0], ["complete", "complete", 0], ["pareto", "Pareto front", 0], ["best_mase", "best MASE", 3], ["best_crps", "best CRPS", 0]]);
    }
    el.innerHTML = html || "<p><em>Results not published yet.</em></p>";
  } catch (e) {
    el.innerHTML = "<p><em>Results not published yet.</em></p>";
  }
}

/* ---- Forecast charts (latest published day) ---- */
async function latestForecast() {
  // index.json lists published target dates (file is named by TARGET day J,
  // not by publication day). Take the latest; if its rte_d1 is still null
  // (RTE publishes J's forecast later on J-1), fall back to the most recent
  // date that HAS RTE, so the comparison curve is shown whenever available.
  try {
    const idx = await hexaFetch("forecasts/index.json");
    if (idx.dates && idx.dates.length) {
      const dates = idx.dates.slice().reverse();
      let fallback = null;
      for (const d of dates) {
        try {
          const j = await hexaFetch(`forecasts/${d}.json`);
          const hasRte = (j.rte_d1 || []).some(v => v != null);
          if (hasRte) return [d, j];
          if (!fallback) fallback = [d, j];
        } catch (e) { /* next */ }
      }
      if (fallback) return fallback;
    }
  } catch (e) { /* fall back to probing */ }
  const today = new Date().toISOString().slice(0, 10);
  const probe = [1, 0, -1].map(k => new Date(Date.now() + k * 864e5).toISOString().slice(0, 10));
  for (const d of probe) {
    try { return [d, await hexaFetch(`forecasts/${d}.json`)]; } catch (e) { /* next */ }
  }
  return null;
}

function hexaChart(canvas, cfg) {
  return new Chart(canvas.getContext("2d"), cfg);
}

async function renderForecast(el) {
  const found = await latestForecast();
  if (!found) {
    el.innerHTML = "<p><em>No forecast published yet — the daily job (forecast.yml, ~11:50 Paris time) will feed this section automatically.</em></p>";
    return;
  }
  const [date, j] = found;
  const first = j.models ? Object.values(j.models)[0] : null;
  const labels = first ? first.median.map((_, i) => `+${i + 1}h`) : [];
  const hasRte = j.rte_d1 && j.rte_d1.some(v => v != null);
  const prev = new Date(j.date + "T00:00:00Z");
  prev.setUTCDate(prev.getUTCDate() - 1);
  el.innerHTML = `<p>Forecast for <b>${j.date}</b> — fit on data available at ${j.as_of} UTC, horizon 24 h.${hasRte ? "" : " (latest published day; RTE's curve appears once published)"}</p>
    ${hasRte ? "" : `<p class="hexa-note">RTE's D-1 forecast for ${j.date} is published later on ${prev.toISOString().slice(0, 10)} — the red curve appears here after the evening backfill. Recent RTE comparisons: see the accuracy section below.</p>`}
    <canvas id="chart-forecast"></canvas>`;
  const palette = { timestral_209m: "#b39ddb", chronos_bolt: "#ce93d8", hgb_features: "#80cbc4", hgb_features_ewma_mondrian: "#80cbc4", xgb_features: "#4db6ac", ridge_features: "#f48fb1" };
  const pretty = { timestral_209m: "Timestral-209M", chronos_bolt: "Chronos-Bolt", hgb_features: "HGB", hgb_features_ewma_mondrian: "HGB + EWMA + Mondrian", xgb_features: "XGBoost", ridge_features: "Ridge" };
  const ds = [];
  for (const [name, m] of Object.entries(j.models || {})) {
    const c = palette[name] || "#90caf9";
    if (m.q05 && m.q95) {
      ds.push({ label: "_" + name + "_lo", data: m.q05, borderWidth: 0, pointRadius: 0, showLine: false, fill: false });
      ds.push({ label: (pretty[name] || name) + " 90%", data: m.q95, borderWidth: 0, pointRadius: 0, showLine: false, backgroundColor: c + "2e", fill: "-1" });
    }
    ds.push({
      label: pretty[name] || name,
      data: m.median,
      borderColor: c,
      borderDash: name === "ridge_features" ? [2, 3] : [],
      fill: false, tension: 0.25, pointRadius: 0, borderWidth: 2,
    });
  }
  if (hasRte) {
    ds.push({ label: "RTE prevision_j1 (D-1)", data: j.rte_d1, borderColor: "#ef5350", borderDash: [6, 4], fill: false, tension: 0.2, pointRadius: 0, borderWidth: 2 });
  }
  hexaChart(document.getElementById("chart-forecast"), {
    type: "line",
    data: { labels, datasets: ds },
    options: { responsive: true,
      plugins: { legend: { labels: { boxWidth: 12, filter: (it) => !String(it.text).startsWith("_") } } },
      scales: { x: { title: { display: true, text: "hours ahead (UTC)" } }, y: { title: { display: true, text: "MW" } } } },
  });
}

/* ---- Accuracy: replay of the last 7 days — each day forecast at D-1 (24 h)
 * the day before. ONE chart over the whole week: actual vs Timestral vs GBM
 * vs RTE. Generated by scripts/site_curves.py -> docs/data/accuracy.json ---- */
async function renderAccuracy(el) {
  let j;
  try { j = await hexaFetch("accuracy.json"); } catch (e) {
    el.innerHTML = "<p><em>Accuracy replay not generated yet — run <code>just site-curves</code>.</em></p>";
    return;
  }
  const days = j.days || [];
  if (!days.length) { el.innerHTML = "<p><em>No replay days yet.</em></p>"; return; }
  const labels = [];
  const actual = [];
  const series = {};  // name -> {median, q05, q95}
  const short = { timestral_209m: "timestral", hgb_features: "hgb", hgb_features_ewma_mondrian: "hgb+ewma+mondrian", ridge_features: "ridge", rte_prevision_j1: "RTE D-1" };
  for (const d of days) {
    for (let h = 0; h < 24; h++) labels.push(`${d.date.slice(5)} ${String(h).padStart(2, "0")}h`);
    actual.push(...d.actual);
    for (const [name, m] of Object.entries(d.models || {})) {
      series[name] = series[name] || { median: [], q05: [], q95: [] };
      const med = Array.isArray(m) ? m : m.median;
      series[name].median.push(...med);
      if (!Array.isArray(m) && m.q05) { series[name].q05.push(...m.q05); series[name].q95.push(...m.q95); }
    }
    if (d.rte_d1) (series.rte_prevision_j1 = series.rte_prevision_j1 || { median: [] }).median.push(...d.rte_d1);
  }
  // score tables: models as rows, days as columns, best cell per day highlighted
  const modelNames = [];
  for (const d of days) for (const k of Object.keys(d.mase || {})) if (!modelNames.includes(k)) modelNames.push(k);
  function scoreTable(metric, dec) {
    const vals = modelNames.map(n => days.map(d => (d[metric] || {})[n] ?? null));
    const means = vals.map(v => { const p = v.filter(x => x != null); return p.length ? p.reduce((a, b) => a + b, 0) / p.length : null; });
    const bestMean = Math.min(...means.filter(v => v != null));
    const bestDay = days.map((_, i) => Math.min(...vals.map(v => v[i]).filter(x => x != null)));
    const head = "<tr><th>model</th>" + days.map(d => `<th>${d.date.slice(5)}</th>`).join("") + "<th>7-day mean</th></tr>";
    const body = modelNames.map((n, r) => {
      const cells = vals[r].map((v, i) => v == null ? "<td>—</td>"
        : `<td${v === bestDay[i] ? ' class="hexa-best"' : ""}>${v.toFixed(dec)}</td>`).join("");
      const mc = means[r];
      return `<tr><td>${short[n] || n}</td>${cells}<td${mc === bestMean ? ' class="hexa-best"' : ""}><b>${mc == null ? "—" : mc.toFixed(dec)}</b></td></tr>`;
    }).join("");
    return `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
  }
  el.innerHTML = `<canvas id="chart-accuracy"></canvas>
    <h4>MASE per day (D-1) — lower is better</h4>${scoreTable("mase", 3)}
    <h4>CRPS per day, MW (D-1) — lower is better</h4>${scoreTable("crps", 0)}`;
  const colors = { timestral_209m: "#b39ddb", hgb_features_ewma_mondrian: "#80cbc4", hgb_features: "#80cbc4", ridge_features: "#f48fb1", rte_prevision_j1: "#ef5350" };
  const dashes = { timestral_209m: [6, 4], hgb_features_ewma_mondrian: [6, 4], hgb_features: [6, 4], ridge_features: [2, 3], rte_prevision_j1: [] };
  const ds = [{ label: "actual", data: actual, borderColor: "#e6e6e6", borderWidth: 2.5, pointRadius: 0, tension: 0.15 }];
  for (const [name, s] of Object.entries(series)) {
    if (name === "rte_prevision_j1") continue;
    const c = colors[name];
    if (s.q05.length) {
      ds.push({ label: "_" + name + "_lo", data: s.q05, borderWidth: 0, pointRadius: 0, showLine: false, fill: false });
      ds.push({ label: name.replace(/_.*/, "") + " 90%", data: s.q95, borderWidth: 0, pointRadius: 0, showLine: false, backgroundColor: c + "1f", fill: "-1" });
    }
    ds.push({ label: name.replace("_209m", "").replace("_features", "") + " (D-1)", data: s.median, borderColor: c, borderDash: dashes[name], borderWidth: 1.8, pointRadius: 0, tension: 0.15 });
  }
  if (series.rte_prevision_j1 && series.rte_prevision_j1.median.some(v => v != null)) ds.push({ label: "RTE D-1", data: series.rte_prevision_j1.median, borderColor: "#ef5350", borderWidth: 1.8, pointRadius: 0, tension: 0.15 });
  hexaChart(document.getElementById("chart-accuracy"), {
    type: "line",
    data: { labels, datasets: ds },
    options: { responsive: true, aspectRatio: 2.6, spanGaps: true,
      plugins: { legend: { labels: { boxWidth: 12, filter: (it) => !String(it.text).startsWith("_") } } },
      scales: { x: { ticks: { autoSkip: true, maxTicksLimit: 14, font: { size: 9 } } }, y: { title: { display: true, text: "MW" } } } },
  });
}

/* ---- Monitoring page: status badge + CUSUM chart + daily MAE bars ---- */
async function renderMonitoring(statusEl, cusumEl, maeEl) {
  let daily, status;
  try { daily = await hexaFetch("monitoring/daily.json"); } catch (e) {
    statusEl.innerHTML = "<p><em>Monitoring not generated yet — the daily job (forecast.yml) will feed this page.</em></p>";
    cusumEl.innerHTML = ""; maeEl.innerHTML = ""; return;
  }
  try { status = await hexaFetch("monitoring/status.json"); } catch (e) { status = null; }
  if (status) {
    const cls = status.status === "served" ? "hexa-badge-ok" : (status.status === "fallback" ? "hexa-badge-warn" : "hexa-badge-bad");
    statusEl.innerHTML = `<p>Status: <span class="hexa-badge ${cls}">${status.status}</span>
      ${status.date ? `· latest forecast: <b>${status.date}</b>` : ""}
      ${status.degraded_inputs && status.degraded_inputs.length ? "<br><span class='hexa-note'>" + status.degraded_inputs.join("; ") + "</span>" : ""}</p>`;
  }
  const days = daily.days || [];
  if (!days.length) { cusumEl.innerHTML = "<p><em>No scored days yet.</em></p>"; maeEl.innerHTML = ""; return; }
  const labels = days.map(d => d.date.slice(5));
  const models = [...new Set(days.flatMap(d => Object.keys(d).filter(k => k.endsWith("__mae")).map(k => k.slice(0, -5))))];
  const palette = { timestral_209m: "#b39ddb", hgb_features: "#80cbc4", hgb_features_ewma_mondrian: "#80cbc4", ridge_features: "#f48fb1", rte_prevision_j1: "#ef5350" };
  // CUSUM chart
  const cusumDs = models.filter(m => days.some(d => d[m + "__cusum"] != null)).map(m => ({
    label: m, data: days.map(d => d[m + "__cusum"] ?? null),
    borderColor: palette[m] || "#90caf9", borderWidth: 1.8, pointRadius: 0, tension: 0.2,
  }));
  cusumDs.push({ label: "+h=5", data: labels.map(() => 5), borderColor: "#ef5350", borderDash: [4, 4], borderWidth: 1, pointRadius: 0 });
  cusumDs.push({ label: "−h=5", data: labels.map(() => -5), borderColor: "#ef5350", borderDash: [4, 4], borderWidth: 1, pointRadius: 0 });
  cusumEl.innerHTML = "<canvas id='chart-cusum'></canvas>";
  hexaChart(document.getElementById("chart-cusum"), {
    type: "line", data: { labels, datasets: cusumDs },
    options: { responsive: true, spanGaps: true, plugins: { legend: { labels: { boxWidth: 10, font: { size: 9 } } } },
      scales: { x: { ticks: { font: { size: 9 } } }, y: { title: { display: true, text: "CUSUM (σ units)" } } } },
  });
  // MAE bars
  const maeDs = models.map(m => ({
    label: m, data: days.map(d => d[m + "__mae"] ?? null),
    backgroundColor: (palette[m] || "#90caf9") + "aa", borderWidth: 0,
  }));
  if (days.some(d => d.rte_mae != null)) maeDs.push({ label: "RTE D-1", data: days.map(d => d.rte_mae ?? null), backgroundColor: "#ef5350aa", borderWidth: 0 });
  maeEl.innerHTML = "<canvas id='chart-mae'></canvas>";
  hexaChart(document.getElementById("chart-mae"), {
    type: "bar", data: { labels, datasets: maeDs },
    options: { responsive: true, plugins: { legend: { labels: { boxWidth: 10, font: { size: 9 } } } },
      scales: { x: { ticks: { font: { size: 9 } } }, y: { title: { display: true, text: "MAE (MW)" } } } },
  });
}
