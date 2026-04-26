/* =========================================================================
   site-wide JS:
   - Sets the GitHub repo link from window.REPO_URL (overridable per page).
   - Populates KPI tiles on the landing page from results.json + simulator_grid.
   ======================================================================= */

window.REPO_URL = window.REPO_URL ||
  "https://github.com/your-username/radiology-ai-decision-making";

document.addEventListener("DOMContentLoaded", () => {
  // 1. Patch any links pointing to the repo placeholder
  document.querySelectorAll("#repo-link, #repo-footer").forEach(a => {
    a.href = window.REPO_URL;
  });

  // 2. Landing-page KPIs (only run if the tiles exist)
  if (document.getElementById("kpi-auc") && !document.getElementById("kpi-n")) {
    populateLandingKpis();
  }
});

async function populateLandingKpis() {
  try {
    const [results, sim] = await Promise.all([
      fetch("assets/results.json").then(r => r.json()),
      fetch("assets/simulator_grid.json").then(r => r.json()),
    ]);

    const auc = mean(Object.values(results.auc));
    setText("kpi-auc", auc.toFixed(3));

    // TPR gap: female vs male
    const sexRows = results.by_sex;
    const m = sexRows.find(r => r.group === "M")?.tpr ?? 0;
    const f = sexRows.find(r => r.group === "F")?.tpr ?? 0;
    const sexGap = m - f;
    setText("kpi-sex", `${(sexGap * 100).toFixed(1)} pts`);

    const ageRows = results.by_age;
    const tprs = ageRows.map(r => r.tpr);
    const ageGap = Math.max(...tprs) - Math.min(...tprs);
    setText("kpi-age", `${(ageGap * 100).toFixed(1)} pts`);

    // Anchoring penalty: drop in "AI shown first" vs "Independent" when AI is wrong.
    const protoMap = sim.ai_wrong_by_protocol.reduce((acc, r) => {
      acc[r.protocol] = r["AI wrong"];
      return acc;
    }, {});
    const drop = (protoMap["Independent"] ?? 0) - (protoMap["AI shown first"] ?? 0);
    setText("kpi-anchor", `−${(drop * 100).toFixed(1)} pts`);
  } catch (err) {
    console.error("Failed to populate landing KPIs", err);
  }
}

function mean(arr) {
  return arr.reduce((s, x) => s + x, 0) / arr.length;
}
function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}
