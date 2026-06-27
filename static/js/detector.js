/* ══════════════════════════════════════════════════════════════════════════
   SpamShield Pro  —  detector.js
   Dataset browsing, custom text input, prediction, and LIME explanation.
   Used by detector.html. Functions are exposed on window so gmail.js
   ══════════════════════════════════════════════════════════════════════════ */

"use strict";

const EXAMPLES = {
  financial: "Congratulations! You have been selected to receive a £5,000 cash prize. To claim your reward call 09071512433. Offer expires tonight. Reply STOP to unsubscribe.",
  phishing:  "Your PayPal account has been suspended due to suspicious login attempts. Verify your identity now: www.paypal-secure-login.xyz/verify",
  ham:       "Hey! Are you coming to the study group tonight? We're meeting at the library at 7pm. Let me know if you need the address.",
};

// ── State ────────────────────────────────────────────────────────────────
let allMessages   = [];
let selectedModel = "svm";
let activeFilter  = "all";
let currentText   = "";

// ── INIT ──────────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", async () => {
  setupTabs();
  setupFilters();
  setupExamples();
  $("btn-analyze")?.addEventListener("click", predictCustom);
  $("btn-explain")?.addEventListener("click", runExplain);

  // detector.js is also loaded on gmail.html (for shared result/XAI
  // rendering), which has no dataset/custom panel — only run dataset
  // loading on pages that actually have it (i.e. has #msg-list).
  if ($("msg-list")) {
    await Promise.all([loadDataset(), loadModelGrid()]);
  } else if ($("models-grid")) {
    await loadModelGrid();
  }
});

// ── TABS (Dataset / Custom) ─────────────────────────────────────────────
function setupTabs() {
  qsa(".tab").forEach(tab => {
    tab.addEventListener("click", () => {
      qsa(".tab").forEach(t => t.classList.remove("active"));
      qsa(".panel").forEach(p => p.classList.remove("active"));
      tab.classList.add("active");
      $("panel-" + tab.dataset.panel).classList.add("active");
    });
  });
}

// ── FILTERS ───────────────────────────────────────────────────────────────
function setupFilters() {
  qsa(".filter-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      qsa(".filter-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      activeFilter = btn.dataset.filter;
      renderMsgList();
    });
  });
}

// ── EXAMPLES ──────────────────────────────────────────────────────────────
function setupExamples() {
  qsa("[data-example]").forEach(btn => {
    btn.addEventListener("click", () => {
      $("custom-input").value = EXAMPLES[btn.dataset.example] || "";
    });
  });
}

// ── DATASET ───────────────────────────────────────────────────────────────
async function loadDataset() {
  try {
    const data = await apiFetch(API.dataset);
    allMessages = data.messages || [];
    renderMsgList();
    const firstSpam = allMessages.find(m => m.label === "spam");
    if (firstSpam) selectMsg(firstSpam);
  } catch(e) {
    $("msg-list").innerHTML = `<div class="loading-row" style="color:var(--accent)">⚠ ${escHtml(e.message)}</div>`;
  }
}

function renderMsgList() {
  const list = $("msg-list");
  const filtered = allMessages.filter(m => {
    if (activeFilter === "all")  return true;
    if (activeFilter === "ham" || activeFilter === "spam") return m.label === activeFilter;
    return m.category === activeFilter;
  });
  if (!filtered.length) {
    list.innerHTML = `<div class="loading-row">No messages match this filter.</div>`;
    return;
  }
  list.innerHTML = filtered.map(m => {
    const idx = allMessages.indexOf(m);
    return `<div class="msg-item" data-idx="${idx}" onclick="selectMsgByIdx(${idx})">
      <div class="msg-item-header">
        <span class="msg-label ${m.label}">${m.label.toUpperCase()}</span>
        <span class="msg-cat">${escHtml(m.category || "")}</span>
      </div>
      <div class="msg-preview">${escHtml(m.text || "")}</div>
    </div>`;
  }).join("");
}

window.selectMsgByIdx = idx => selectMsg(allMessages[idx]);

function selectMsg(msg) {
  qsa(".msg-item").forEach(el =>
    el.classList.toggle("selected", parseInt(el.dataset.idx) === allMessages.indexOf(msg))
  );
  currentText = msg.text || "";
  $("msg-display").textContent = msg.text || "";
  resetXai();
  hideId("result-area");
  hideId("token-section");
  runPredict(msg.text || "");
}

// ── MODEL GRID ────────────────────────────────────────────────────────────
// Fetches its own metrics (lighter than the full dashboard load — this page
// only needs the model cards, not the charts).
async function loadModelGrid() {
  try {
    const metricsData = await apiFetch(API.metrics);
    buildModelGrid(metricsData);
  } catch(e) {
    $("models-grid").innerHTML = `<div class="loading-row" style="color:var(--accent)">⚠ ${escHtml(e.message)}</div>`;
  }
}

function buildModelGrid(metricsData) {
  const grid = $("models-grid");
  const models = metricsData.models || {};

  const ICONS  = { naive_bayes: "📊", svm: "⚡", dnn: "🧠" };
  const LABELS = { naive_bayes: "Naive Bayes", svm: "SVM", dnn: "DNN" };
  const TYPES  = {
    naive_bayes: "Pipeline · TF-IDF + MNB",
    svm:         "GridSearchCV · LinearSVC",
    dnn:         "Deep Neural Net · TF-IDF",
  };

  grid.innerHTML = Object.entries(models).map(([key, m]) => {
    const isDnn    = key === "dnn";
    const isSel    = key === selectedModel;
    const bestPrms = m.best_params ? Object.entries(m.best_params)
      .map(([k, v]) => `<span class="model-badge">${k.split("__").pop()}=${v}</span>`).join("") : "";
    return `<div class="model-card${isDnn ? " dnn-card" : ""}${isSel ? " selected" : ""}" onclick="chooseModel('${key}')">
      <div class="model-card-top">
        <span class="model-icon">${ICONS[key] || "🤖"}</span>
        ${isDnn ? '<span class="dnn-label">DNN + XAI</span>' : ""}
      </div>
      <div class="model-name">${m.name || LABELS[key]}</div>
      <div class="model-type">${m.type || TYPES[key]}</div>
      <div class="model-acc">${m.accuracy != null ? m.accuracy + "%" : "—"}</div>
      <div class="model-tags">
        <span class="model-tag">P:${m.precision ?? "—"}%</span>
        <span class="model-tag">R:${m.recall ?? "—"}%</span>
        <span class="model-tag">F1:${m.f1 ?? "—"}%</span>
      </div>
      ${bestPrms}
    </div>`;
  }).join("");
}

window.chooseModel = key => {
  selectedModel = key;
  qsa(".model-card").forEach(c => c.classList.remove("selected"));
  qsa(".model-card").forEach((c, i) => {
    const k = Object.keys({ naive_bayes:1, svm:1, dnn:1 })[i];
    if (k === key) c.classList.add("selected");
  });
  if (currentText) runPredict(currentText);
};

// ── PREDICT ───────────────────────────────────────────────────────────────
async function runPredict(text) {
  if (!text.trim()) return;
  currentText = text;
  const btn = $("btn-analyze");
  if (btn) btn.disabled = true;

  try {
    const data = await apiFetch(API.predict, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text, model: selectedModel }),
    });
    renderResult(data);
  } catch(e) {
    $("msg-display").textContent = `Prediction error: ${e.message}`;
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function predictCustom() {
  const text = $("custom-input").value.trim();
  if (!text) return;
  currentText = text;
  $("msg-display").textContent = text;
  hideId("token-section");
  hideId("result-area");
  resetXai();

  const btn     = $("btn-analyze");
  const spinner = $("analyze-spinner");
  btn.disabled = true;
  show(spinner);
  try {
    await runPredict(text);
  } finally {
    btn.disabled = false;
    hide(spinner);
  }
}

function renderResult(data) {
  const isSpam = data.label === "spam";

  // ── Tokens ──────────────────────────────────────────────────────────
  if (data.tokens?.length) {
    showId("token-section");
    $("token-display").innerHTML = data.tokens.map(t =>
      `<span class="token ${t.is_spam ? "spam-w" : "normal"}">${escHtml(t.word)}</span>`
    ).join("");
  }

  // ── Banner ──────────────────────────────────────────────────────────
  const banner = $("result-banner");
  banner.className = `result-banner ${isSpam ? "spam" : "ham"}`;
  $("result-label").textContent = isSpam ? "SPAM DETECTED" : "HAM — NOT SPAM";

  const catEl = $("result-cat");
  catEl.className = `result-category cat-${data.category}`;
  catEl.textContent = `Category: ${(data.category || "").toUpperCase()}`;

  $("result-conf").textContent = `${data.confidence}%`;
  $("result-conf").style.color = isSpam ? "var(--accent)" : "var(--accent3)";

  const fill = $("conf-fill");
  fill.style.width = `${data.confidence}%`;
  fill.className   = `conf-fill ${isSpam ? "fill-spam" : "fill-ham"}`;

  // ── Spam Risk Score ──────────────────────────────────────────────────
  renderRiskScore(data.risk_score);
  
  // ── Processing time ─────────────────────────────────────────────────
  if (data.processing_time_ms != null) {
    const t = $("predict-timing");
    t.textContent = `⚡ ${data.processing_time_ms}ms`;
    show(t);
  }

  // ── Text meta chips ─────────────────────────────────────────────────
  const tf = data.text_features || {};
  $("text-meta").innerHTML = [
    tf.char_count  != null ? `<span class="meta-chip">📏 ${tf.char_count} chars</span>` : "",
    tf.word_count  != null ? `<span class="meta-chip">📝 ${tf.word_count} words</span>` : "",
    tf.has_url     ? `<span class="meta-chip" style="color:var(--warn)">🔗 URL detected</span>` : "",
    tf.has_currency ? `<span class="meta-chip" style="color:var(--warn)">💰 Currency symbol</span>` : "",
    tf.has_phone   ? `<span class="meta-chip" style="color:var(--warn)">📞 Phone number</span>` : "",
    tf.exclamation_ct > 1 ? `<span class="meta-chip" style="color:var(--accent)">! × ${tf.exclamation_ct}</span>` : "",
  ].join("");

  // ── Ensemble consensus ─────────────────────────────────────────────
  renderEnsemble(data.ensemble);


  // ── All models ──────────────────────────────────────────────────────
  $("all-models-grid").innerHTML = Object.entries(data.all_models || {}).map(([k, m]) => {
    const isS  = m.label === "spam";
    const col  = isS ? "var(--accent)" : "var(--accent3)";
    const isDnn = k === "dnn";
    return `<div class="mini-model${isDnn ? " mini-dnn" : ""}">
      <div class="m-name">${escHtml(m.name)}${isDnn ? " 🧠" : ""}</div>
      <div class="m-pred" style="color:${col}">${isS ? "SPAM" : "HAM"}</div>
      <div class="m-conf">${m.confidence}%</div>
    </div>`;
  }).join("");

  showId("result-area");
}

function renderEnsemble(ensemble) {
  const card = $("ensemble-card");
  if (!ensemble) {
    // Fewer than 2 models loaded — nothing meaningful to vote on.
    hide(card);
    return;
  }
  const isSpam = ensemble.label === "spam";
  card.className = `ensemble-card ${isSpam ? "spam" : "ham"}`;
  show(card);

  $("ensemble-agreement").textContent = `${ensemble.agreement}% agreement`;
  $("ensemble-label").textContent = isSpam ? "SPAM" : "HAM";
  $("ensemble-label").style.color = isSpam ? "var(--accent)" : "var(--accent3)";
  $("ensemble-conf").textContent = `${ensemble.confidence}%`;

  const votes = ensemble.votes || {};
  $("ensemble-votes").innerHTML = `
    <span class="vote-chip vote-spam">${votes.spam ?? 0} spam</span>
    <span class="vote-chip vote-ham">${votes.ham ?? 0} ham</span>
  `;
}

const RISK_TIER_COLORS = {
  critical: "#ff4d6d", high: "#ff784d", medium: "#ffa500", low: "#00d4aa",
};
const RISK_LABELS = { url: "URL", keyword: "Keyword", urgency: "Urgency", probability: "Probability" };

function renderRiskScore(risk) {
  const card = $("risk-card");
  if (!risk) { hide(card); return; }
  show(card);

  const tierColor = RISK_TIER_COLORS[risk.tier] || "#9aa0b4";
  $("risk-tier").textContent = risk.tier.toUpperCase();
  $("risk-tier").style.color = tierColor;
  $("risk-tier").style.background = tierColor + "22";

  $("risk-composite").textContent = risk.composite;
  $("risk-composite").style.color = tierColor;
  $("risk-composite-fill").style.width = `${risk.composite}%`;
  $("risk-composite-fill").style.background = tierColor;

  const breakdown = risk.breakdown || {};
  $("risk-breakdown").innerHTML = Object.entries(breakdown).map(([key, val]) => `
    <div class="risk-sub-row">
      <span class="risk-sub-label">${RISK_LABELS[key] || key}</span>
      <div class="risk-sub-bar"><div class="risk-sub-fill" style="width:${val}%"></div></div>
      <span class="risk-sub-val">${val}</span>
    </div>
  `).join("");
}


function resetXai() {
  hideId("xai-result");
  hideId("xai-loading");
  showId("xai-placeholder");
  $("xai-placeholder").innerHTML = "Click <strong>Generate Explanation</strong> to see which words drive this prediction.";
  $("xai-bars").innerHTML = "";
  $("explain-btn-text").textContent = "Generate Explanation";
  const b = $("btn-explain");
  b.disabled = false;
  hide($("explain-spinner"));
  xaiCache = {};
}

async function runExplain() {
  const text = currentText || $("msg-display").textContent;
  if (!text || text.startsWith("Select a message") || text.startsWith("Fetch your Gmail")) return;

  const btn     = $("btn-explain");
  const spinner = $("explain-spinner");
  btn.disabled  = true;
  show(spinner);
  $("explain-btn-text").textContent = `Running ${cfg.label}...`;
  $("xai-loading-text").textContent = cfg.loadingText;
  hideId("xai-result");
  showId("xai-loading");
  hideId("xai-placeholder");

  try {
    const data = await apiFetch(API[cfg.endpoint], {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text, model: selectedModel, top_n: 12 }),
    });
    const exps = data.explanation || [];
    if (!exps.length) {
      $("xai-placeholder").innerHTML = `<span style="color:var(--accent)">⚠ No explanation available. Install: <code>${cfg.notInstalledHint}</code></span>`;
      showId("xai-placeholder");
    } else {
      xaiCache[activeXaiMethod] = exps;
      renderXaiBars(exps);
      showId("xai-result");
    }
  } catch(e) {
    $("xai-placeholder").innerHTML = `<span style="color:var(--accent)">⚠ ${cfg.label} error: ${escHtml(e.message)}</span>`;
    showId("xai-placeholder");
  } finally {
    btn.disabled = false;
    hide(spinner);
    hideId("xai-loading");
    $("explain-btn-text").textContent = "Re-run Explanation";
  }
}

function renderXaiBars(exps) {
  const maxAbs = Math.max(...exps.map(e => Math.abs(e.weight)));
  $("xai-bars").innerHTML = exps.map(e => {
    const pct    = Math.round((Math.abs(e.weight) / maxAbs) * 100);
    const isSpam = e.direction === "spam";
    const col    = isSpam ? "#ff4d6d" : "#00d4aa";
    const bg     = isSpam ? "rgba(255,77,109,.08)" : "rgba(0,212,170,.08)";
    const valStr = (e.weight > 0 ? "+" : "") + e.weight.toFixed(4);
    return `<div class="xai-bar-row" style="background:${bg}">
      <div class="xai-bar-label" title="${escHtml(e.word)}">${escHtml(e.word)}</div>
      <div class="xai-bar-track"><div class="xai-bar-fill" style="width:${pct}%;background:${col}"></div></div>
      <div class="xai-bar-val" style="color:${col}">${valStr}</div>
      <div class="xai-dir-badge ${isSpam ? "xai-spam" : "xai-ham"}">${isSpam ? "→SPAM" : "→HAM"}</div>
    </div>`;
  }).join("");
}

// ── Expose for gmail.js reuse on the Gmail page ────────────────────────────
window.runPredict  = runPredict;
window.renderResult = renderResult;
window.resetXai     = resetXai;
window.runExplain   = runExplain;
window.getSelectedModel = () => selectedModel;
window.setCurrentText   = t => { currentText = t; };
