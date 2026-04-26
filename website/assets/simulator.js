/* =========================================================================
   simulator.js — interactive anchoring simulator
   ======================================================================= */

const PALETTE = {
  primary: "#1f4e79",
  accent:  "#f5a623",
  danger:  "#c8102e",
  muted:   "#9aa0a6",
};

let GRID = null;

document.addEventListener("DOMContentLoaded", async () => {
  GRID = await fetch("assets/simulator_grid.json").then(r => r.json());

  drawSensitivityChart();
  populateProtocolTable();

  const slider  = document.getElementById("w-slider");
  const onChange = () => updateOutcomes(parseFloat(slider.value));
  slider.addEventListener("input", onChange);
  onChange();
});

function updateOutcomes(w) {
  document.getElementById("w-readout").textContent = w.toFixed(2);

  const i  = nearestIdx(GRID.weights, w);
  const o  = GRID.overall[i];
  const ar = GRID.ai_right[i];
  const aw = GRID.ai_wrong[i];

  setVal("out-overall", o);
  setVal("out-airight", ar);
  setVal("out-aiwrong", aw);

  // Move the vertical marker on the chart
  Plotly.relayout("sensitivity-chart", {
    shapes: [{
      type: "line",
      x0: w, x1: w, y0: 0, y1: 1.0, yref: "paper",
      line: { color: PALETTE.muted, width: 2, dash: "dash" },
    }],
    annotations: [{
      x: w, y: 1.02, yref: "paper",
      text: `w<sub>AI</sub> = ${w.toFixed(2)}`,
      showarrow: false, font: { color: PALETTE.muted, size: 11 },
    }],
  });
}

function drawSensitivityChart() {
  const traces = [
    { x: GRID.weights, y: GRID.overall, name: "Overall accuracy",
      mode: "lines+markers", line: { color: PALETTE.primary, width: 3 },
      marker: { size: 7 } },
    { x: GRID.weights, y: GRID.ai_right, name: "When AI is correct",
      mode: "lines+markers", line: { color: PALETTE.accent,  width: 3 },
      marker: { size: 7 } },
    { x: GRID.weights, y: GRID.ai_wrong, name: "When AI is wrong",
      mode: "lines+markers", line: { color: PALETTE.danger,  width: 3 },
      marker: { size: 7 } },
  ];

  Plotly.newPlot("sensitivity-chart", traces, {
    title: { text: "Clinician accuracy vs anchoring weight",
             font: { size: 16, weight: 700 } },
    font:  { family: "Inter, system-ui, sans-serif", color: "#1a1d21" },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor:  "rgba(0,0,0,0)",
    margin: { t: 40, r: 20, b: 60, l: 60 },
    xaxis: { title: "Anchoring weight on AI (w<sub>AI</sub>)",
             range: [0, 1], gridcolor: "#eee" },
    yaxis: { title: "Clinician accuracy", range: [0, 1.05], gridcolor: "#eee" },
    legend: { orientation: "h", y: -0.18 },
    hovermode: "x unified",
  }, { responsive: true, displaylogo: false });
}

function populateProtocolTable() {
  const tbody = document.getElementById("protocol-table");
  const weights = {
    "Independent":     [1.00, 0.00],
    "AI shown first":  [0.55, 0.45],
    "AI shown after":  [0.85, 0.15],
  };

  // GRID.protocols rows look like:
  // { pathology, Independent, "AI shown first", "AI shown after" }
  // Average each protocol column across pathologies.
  const overallAvg = {};
  ["Independent", "AI shown first", "AI shown after"].forEach(proto => {
    const vals = GRID.protocols.map(row => row[proto]).filter(v => v != null);
    overallAvg[proto] = mean(vals);
  });

  const aiWrong = {};
  GRID.ai_wrong_by_protocol.forEach(r => { aiWrong[r.protocol] = r["AI wrong"]; });

  ["Independent", "AI shown first", "AI shown after"].forEach(proto => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${proto}</strong></td>
      <td>${weights[proto][0].toFixed(2)}</td>
      <td>${weights[proto][1].toFixed(2)}</td>
      <td>${(overallAvg[proto] ?? 0).toFixed(3)}</td>
      <td>${(aiWrong[proto] ?? 0).toFixed(3)}</td>`;
    tbody.appendChild(tr);
  });
}
function mean(a) { return a.reduce((s, x) => s + x, 0) / a.length; }
function nearestIdx(arr, x) {
  let best = 0, bestD = Infinity;
  for (let i = 0; i < arr.length; i++) {
    const d = Math.abs(arr[i] - x);
    if (d < bestD) { best = i; bestD = d; }
  }
  return best;
}
function setVal(id, p) {
  const el = document.getElementById(id);
  if (el) el.textContent = (p * 100).toFixed(1) + "%";
}
