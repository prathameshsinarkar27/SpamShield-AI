/* ══════════════════════════════════════════════════════════════════════════
   SpamShield AI  —  dashboard.js
   Visualization dashboard: stats cards, charts, confusion matrices,
   GridSearchCV results, DNN training history. Used by dashboard.html.
   ══════════════════════════════════════════════════════════════════════════ */

"use strict";

let charts = {};

window.addEventListener("DOMContentLoaded", () => {
  loadDashboard();
});

// ── DASHBOARD ─────────────────────────────────────────────────────────────
async function loadDashboard() {
  showId("dashboard-loading");
  hideId("dashboard-content");

  try {
    const [statsData, metricsData] = await Promise.all([
      apiFetch(API.stats),
      apiFetch(API.metrics),
    ]);

    // Dashboard stat row
    $("stat-total").textContent = statsData.total?.toLocaleString() || "—";
    $("stat-ham").textContent   = statsData.ham?.toLocaleString()   || "—";
    $("stat-spam").textContent  = statsData.spam?.toLocaleString()  || "—";
    $("stat-ratio").textContent = statsData.class_ratio != null ? statsData.class_ratio + "%" : "—";

    // Charts
    buildCharts(statsData, metricsData);

    // Top spam words (learned, not hardcoded)
    buildTopWords(metricsData.top_spam_words || []);

    // GridSearch results
    buildGridSearchInfo(metricsData);

    // DNN history — fetched independently of model-load status (Phase 2 fix:
    // the chart only needs dnn_history.json, not the loaded Keras model, so
    // it shouldn't be hidden just because dnn_available is false). The fetch
    // itself self-gates: loadDnnHistory() no-ops if no history data exists.
    loadDnnHistory();

    hideId("dashboard-loading");
    showId("dashboard-content");
  } catch(e) {
    $("dashboard-loading").innerHTML =
      `<div class="loading-row" style="color:var(--accent)">⚠ Dashboard error: ${escHtml(e.message)}</div>`;
  }
}

// ── CHARTS ────────────────────────────────────────────────────────────────
function buildCharts(stats, metrics) {
  Chart.defaults.color       = "#9aa0b4";
  Chart.defaults.borderColor = "rgba(42,47,69,0.9)";

  const cats     = stats.categories || {};
  const spamOnly = Object.fromEntries(
    Object.entries(cats).filter(([k]) => k !== "normal")
  );
  const modelsData = metrics.models || {};

  // ── C1: Class distribution donut ──────────────────────────────────────
  if (charts.c1) charts.c1.destroy();
  charts.c1 = new Chart($("c1"), {
    type: "doughnut",
    data: {
      labels: [`Ham (${stats.ham?.toLocaleString()})`, `Spam (${stats.spam?.toLocaleString()})`],
      datasets: [{ data: [stats.ham, stats.spam], backgroundColor: ["#00d4aa","#ff4d6d"], borderWidth:0, hoverOffset:5 }],
    },
    options: {
      responsive:true, maintainAspectRatio:false, cutout:"65%",
      plugins:{ legend:{display:false}, tooltip:{callbacks:{label:c=>`${c.label}: ${c.parsed?.toLocaleString()}`}} },
    },
  });
  $("legend-c1").innerHTML = `
    <span><span class="legend-dot" style="background:#00d4aa"></span>Ham — ${stats.ham?.toLocaleString()}</span>
    <span><span class="legend-dot" style="background:#ff4d6d"></span>Spam — ${stats.spam?.toLocaleString()}</span>`;

  // ── C2: Spam categories bar ────────────────────────────────────────────
  const catLabels = Object.keys(spamOnly);
  const catVals   = Object.values(spamOnly);
  const CAT_COLORS = { financial:"#ffa500", promotional:"#4db8ff", scam:"#ff784d", adult:"#c84dff", phishing:"#ff4d6d" };
  if (charts.c2) charts.c2.destroy();
  charts.c2 = new Chart($("c2"), {
    type:"bar",
    data:{
      labels: catLabels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
      datasets:[{ data:catVals, backgroundColor:catLabels.map(l=>CAT_COLORS[l]||"#7b61ff"), borderWidth:0, borderRadius:4 }],
    },
    options:{
      responsive:true, maintainAspectRatio:false,
      plugins:{legend:{display:false}},
      scales:{x:{ticks:{font:{size:10}}}, y:{ticks:{font:{size:10}}}},
    },
  });

  // ── C3: Model performance grouped bar ─────────────────────────────────
  const modelKeys    = Object.keys(modelsData);
  const modelNames   = modelKeys.map(k => modelsData[k].name || k);
  const precData     = modelKeys.map(k => modelsData[k].precision  ?? 0);
  const recData      = modelKeys.map(k => modelsData[k].recall     ?? 0);
  const f1Data       = modelKeys.map(k => modelsData[k].f1         ?? 0);
  if (charts.c3) charts.c3.destroy();
  charts.c3 = new Chart($("c3"), {
    type:"bar",
    data:{
      labels: modelNames,
      datasets:[
        { label:"Precision", data:precData, backgroundColor:"#7b61ff", borderRadius:4 },
        { label:"Recall",    data:recData,  backgroundColor:"#ff4d6d", borderRadius:4 },
        { label:"F1-Score",  data:f1Data,   backgroundColor:"#00d4aa", borderRadius:4 },
      ],
    },
    options:{
      responsive:true, maintainAspectRatio:false,
      plugins:{legend:{display:false}},
      scales:{
        // min:0 (not a hardcoded 90) — a model scoring below 90 on any
        // metric (e.g. Naive Bayes recall) would otherwise have its bar
        // clipped to invisible by Chart.js.
        y:{ min:0, max:101, ticks:{font:{size:10}} },
        x:{ ticks:{font:{size:11}, autoSkip:false, maxRotation:0} },
      },
    },
  });

  // ── C5: ROC curves ────────────────────────────────────────────────────
  // Real FPR/TPR points come from metrics.models[key].roc, computed in
  // train.py via sklearn.metrics.roc_curve and saved to results.json.
  // (Phase 2 fix — previously these were hardcoded placeholder coordinates.)
  const ROC_COLORS = { naive_bayes:"#7b61ff", svm:"#ff4d6d", dnn:"#00d4aa" };
  const modelsWithRoc = modelKeys.filter(k => Array.isArray(modelsData[k].roc) && modelsData[k].roc.length);

  if (modelsWithRoc.length) {
    const rocDS = modelsWithRoc.map(k => ({
      label: `${modelsData[k].name || k} (AUC=${modelsData[k].auc ?? "—"})`,
      data:  modelsData[k].roc,
      borderColor: ROC_COLORS[k] || "#7b61ff",
      borderWidth: 2.5, pointRadius: 0, fill: false, tension: 0.2,
    }));
    rocDS.push({ label:"Random", data:[{x:0,y:0},{x:1,y:1}], borderColor:"rgba(90,96,128,.4)", borderWidth:1, pointRadius:0, fill:false, borderDash:[5,4] });

    $("roc-legend").innerHTML = modelsWithRoc
      .map(k => `<span><span class="legend-dot" style="background:${ROC_COLORS[k]||"#7b61ff"}"></span>${modelsData[k].name||k} (AUC ${modelsData[k].auc||"—"})</span>`)
      .join("");

    if (charts.c5) charts.c5.destroy();
    charts.c5 = new Chart($("c5"), {
      type:"line", data:{datasets:rocDS},
      options:{
        responsive:true, maintainAspectRatio:false,
        plugins:{legend:{display:false}},
        scales:{
          x:{type:"linear",min:0,max:1,title:{display:true,text:"False Positive Rate",font:{size:11}},ticks:{font:{size:10}}},
          y:{min:0,max:1,   title:{display:true,text:"True Positive Rate",   font:{size:11}},ticks:{font:{size:10}}},
        },
      },
    });
  } else {
    // No trained results.json with roc data yet (e.g. fallback metrics, or
    // an older results.json from before this fix) — show a clear message
    // instead of plotting fake curves.
    if (charts.c5) { charts.c5.destroy(); charts.c5 = null; }
    $("roc-legend").innerHTML = "";
    $("c5").parentElement.innerHTML =
      '<div class="loading-row">ROC data not available — run <code>python train.py</code> to generate it.</div>';
  }

  // Confusion matrices from real results
  buildConfusionMatrices(modelsData);
}

function buildConfusionMatrices(modelsData) {
  const container = $("cm-grid");
  const entries   = Object.entries(modelsData).filter(([, m]) => m.cm);
  if (!entries.length) { container.innerHTML = '<div class="loading-row">Confusion matrices not available — run train.py</div>'; return; }
  container.innerHTML = entries.map(([key, m]) => {
    const [[tn, fp], [fn, tp]] = m.cm || [[0,0],[0,0]];
    const isDnn = key === "dnn";
    return `<div class="cm-block">
      <div class="cm-block-title">${escHtml(m.name || key)}${isDnn?" 🧠":""}</div>
      <div class="cm-wrap">
        <div class="cm-y-label">Predicted</div>
        <div class="cm-inner">
          <div class="cm-header"><span>Ham</span><span>Spam</span></div>
          <div class="cm-row">
            <span class="cm-row-label">Ham</span>
            <div class="cm-cell cm-tn">${Number(tn).toLocaleString()}</div>
            <div class="cm-cell cm-fp">${fp}</div>
          </div>
          <div class="cm-row">
            <span class="cm-row-label">Spam</span>
            <div class="cm-cell cm-fn">${fn}</div>
            <div class="cm-cell cm-tp">${tp}</div>
          </div>
        </div>
      </div>
    </div>`;
  }).join("");
}

function buildTopWords(words) {
  const el = $("top-words");
  if (!words.length) { el.innerHTML = '<div class="loading-row">Run train.py to generate learned vocabulary.</div>'; return; }
  const maxScore = Math.max(...words.map(w => Math.abs(w.score || 0)));
  el.innerHTML = words.map(w => {
    const pct = maxScore > 0 ? Math.round((Math.abs(w.score) / maxScore) * 100) : 50;
    return `<div class="word-row">
      <span class="word-label" title="${escHtml(w.word)}">${escHtml(w.word)}</span>
      <div class="word-bar-bg"><div class="word-bar-fill" style="width:${pct}%"></div></div>
      <span class="word-score">${(w.score ?? 0).toFixed(2)}</span>
    </div>`;
  }).join("");
}

function buildGridSearchInfo(metricsData) {
  const svm = metricsData.models?.svm;
  if (!svm?.best_params) return;
  const card = $("gridsearch-card");
  card.style.display = "block";
  const chips = Object.entries(svm.best_params)
    .map(([k, v]) => `<div class="gs-param">${escHtml(k.split("__").pop())}: <span>${escHtml(String(v))}</span></div>`)
    .join("");
  $("gridsearch-content").innerHTML =
    chips + (svm.best_cv_f1 != null
      ? `<div class="gs-f1">Best CV F1: <strong>${svm.best_cv_f1}%</strong></div>` : "");
}

async function loadDnnHistory() {
  try {
    const data = await apiFetch(API.dnnHistory);
    if (!data?.loss) return;
    showId("dnn-history-card");
    const epochs = data.loss.map((_, i) => i + 1);
    if (charts.c6) charts.c6.destroy();
    charts.c6 = new Chart($("c6"), {
      type:"line",
      data:{
        labels: epochs,
        datasets:[
          { label:"Train Loss", data:data.loss,         borderColor:"#7b61ff", borderWidth:2, pointRadius:0, fill:false, tension:0.3, yAxisID:"y"  },
          { label:"Val Loss",   data:data.val_loss,     borderColor:"#ff4d6d", borderWidth:2, pointRadius:0, fill:false, tension:0.3, yAxisID:"y", borderDash:[4,3] },
          { label:"Train Acc",  data:data.accuracy,     borderColor:"#00d4aa", borderWidth:2, pointRadius:0, fill:false, tension:0.3, yAxisID:"y2" },
          { label:"Val Acc",    data:data.val_accuracy, borderColor:"#ffa500", borderWidth:2, pointRadius:0, fill:false, tension:0.3, yAxisID:"y2", borderDash:[4,3] },
        ],
      },
      options:{
        responsive:true, maintainAspectRatio:false,
        plugins:{legend:{display:false}},
        scales:{
          x:  { title:{display:true,text:"Epoch",font:{size:11}}, ticks:{font:{size:10}} },
          y:  { title:{display:true,text:"Loss",   font:{size:11}}, ticks:{font:{size:10}}, position:"left" },
          y2: { title:{display:true,text:"Accuracy",font:{size:11}}, ticks:{font:{size:10}}, position:"right", grid:{drawOnChartArea:false} },
        },
      },
    });
  } catch(e) {
    console.warn("DNN history not available:", e.message);
  }
}
