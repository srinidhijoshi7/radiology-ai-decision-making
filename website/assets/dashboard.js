/* =========================================================================
   dashboard.js — Plotly charts for dashboard.html
   ======================================================================= */

const PALETTE = {
  primary:   "#1f4e79",
  accent:    "#f5a623",
  danger:    "#c8102e",
  primaryLt: "#3a6da3",
  muted:     "#9aa0a6",
};

const PLOTLY_BASE = {
  font:        { family: "Inter, system-ui, sans-serif", size: 12, color: "#1a1d21" },
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor:  "rgba(0,0,0,0)",
  margin:      { t: 30, r: 20, b: 60, l: 60 },
  hoverlabel:  { bgcolor: "#fff", font: { size: 12 } },
};

document.addEventListener("DOMContentLoaded", async () => {
  const r = await fetch("assets/results.json").then(r => r.json());

  // ---------- KPI tiles ----------
  document.getElementById("kpi-n").textContent = r.n_studies;
  document.getElementById("kpi-auc").textContent =
    mean(Object.values(r.auc)).toFixed(3);
  document.getElementById("kpi-brier").textContent =
    mean(Object.values(r.brier)).toFixed(3);
  document.getElementById("kpi-kappa").textContent =
    mean(Object.values(r.cohen_kappa)).toFixed(3);

  // ---------- AUC bar ----------
  const pathologies = Object.keys(r.auc);
  const aucs = pathologies.map(p => r.auc[p]);
  Plotly.newPlot("auc-chart", [{
    type: "bar", orientation: "h",
    x: aucs, y: pathologies,
    marker: { color: aucs.map(a => colorScale(a, 0.75, 0.95)) },
    text: aucs.map(a => a.toFixed(3)),
    textposition: "outside",
    hovertemplate: "%{y}<br>AUC = %{x:.3f}<extra></extra>",
  }], {
    ...PLOTLY_BASE,
    title: { text: "ROC-AUC per pathology", font: { size: 15, weight: 700 } },
    xaxis: { range: [0.7, 1.0], title: "AUC", gridcolor: "#eee" },
    yaxis: { automargin: true },
    margin: { ...PLOTLY_BASE.margin, l: 130 },
  }, { responsive: true, displaylogo: false });

  // ---------- Brier bar ----------
  const briers = pathologies.map(p => r.brier[p]);
  Plotly.newPlot("brier-chart", [{
    type: "bar", orientation: "h",
    x: briers, y: pathologies,
    marker: { color: briers.map(b => colorScale(1 - b * 4, 0.4, 0.95)) },
    text: briers.map(b => b.toFixed(3)),
    textposition: "outside",
    hovertemplate: "%{y}<br>Brier = %{x:.3f}<extra></extra>",
  }], {
    ...PLOTLY_BASE,
    title: { text: "Brier score (calibration loss)", font: { size: 15, weight: 700 } },
    xaxis: { title: "Brier score (lower = better)", gridcolor: "#eee" },
    yaxis: { automargin: true },
    margin: { ...PLOTLY_BASE.margin, l: 130 },
  }, { responsive: true, displaylogo: false });

  // ---------- Subgroup: sex ----------
  drawSubgroup("sex-chart", r.by_sex, "Performance by sex");
  drawSubgroup("age-chart", r.by_age, "Performance by age band");

  // ---------- Summary table ----------
  const tbody = document.querySelector("#summary-table tbody");
  pathologies.forEach(p => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${p}</strong></td>
      <td>${r.auc[p].toFixed(3)}</td>
      <td>${r.brier[p].toFixed(3)}</td>
      <td>${r.cohen_kappa[p].toFixed(3)}</td>`;
    tbody.appendChild(tr);
  });
});

function drawSubgroup(id, rows, title) {
  rows = [...rows].sort((a, b) => String(a.group).localeCompare(String(b.group)));
  const groups = rows.map(r => r.group);
  Plotly.newPlot(id, [
    { type: "bar", name: "AUC",  x: groups, y: rows.map(r => r.auc),
      marker: { color: PALETTE.primary }, text: rows.map(r => r.auc.toFixed(2)),
      textposition: "outside" },
    { type: "bar", name: "TPR",  x: groups, y: rows.map(r => r.tpr),
      marker: { color: PALETTE.accent }, text: rows.map(r => r.tpr.toFixed(2)),
      textposition: "outside" },
    { type: "bar", name: "FPR",  x: groups, y: rows.map(r => r.fpr),
      marker: { color: PALETTE.danger }, text: rows.map(r => r.fpr.toFixed(2)),
      textposition: "outside" },
  ], {
    ...PLOTLY_BASE,
    title: { text: title, font: { size: 15, weight: 700 } },
    barmode: "group",
    yaxis: { range: [0, 1.05], gridcolor: "#eee" },
    legend: { orientation: "h", y: -0.2 },
  }, { responsive: true, displaylogo: false });
}

function colorScale(v, lo, hi) {
  // lo→muted blue, hi→accent orange (continuous)
  const t = Math.min(1, Math.max(0, (v - lo) / (hi - lo)));
  return `rgb(${Math.round(31 + (245 - 31) * t)}, ${Math.round(78 + (166 - 78) * t)}, ${Math.round(121 + (35 - 121) * t)})`;
}
function mean(a) { return a.reduce((s, x) => s + x, 0) / a.length; }
