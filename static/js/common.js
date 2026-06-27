/* ══════════════════════════════════════════════════════════════════════════
   SpamShield AI  —  common.js
   Shared across every page: API endpoints, DOM helpers, header status check.
   Loaded by layout.html before any page-specific script.
   ══════════════════════════════════════════════════════════════════════════ */

"use strict";

// ── API endpoints (match Flask blueprints) ────────────────────────────────
const API = {
  dataset:    "/api/dataset",
  stats:      "/api/stats",
  predict:    "/api/predict",
  explain:    "/api/explain",
  explainShap: "/api/explain-shap",
  metrics:    "/api/metrics",
  dnnHistory: "/api/dnn-history",
  gmailFetch: "/api/gmail/fetch",
  gmailClassify: "/api/gmail/classify",
};

// ── DOM helpers ───────────────────────────────────────────────────────────
const $  = id => document.getElementById(id);
const qsa = s => document.querySelectorAll(s);

function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }
function showId(id) { const el = $(id); if (el) show(el); }
function hideId(id) { const el = $(id); if (el) hide(el); }

function escHtml(str) {
  return String(str ?? "")
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

async function apiFetch(url, opts = {}) {
  try {
    const r = await fetch(url, opts);
    if (!r.ok) {
      const err = await r.json().catch(() => ({ error: r.statusText }));
      throw new Error(err.error || r.statusText);
    }
    return await r.json();
  } catch(e) {
    console.error(`API error [${url}]:`, e.message);
    throw e;
  }
}

// ── Header: server status + message-count pills (present on every page) ───
async function checkServerStatus() {
  const dot  = $("status-dot");
  const text = $("status-text");
  try {
    const statsData = await apiFetch(API.stats);
    dot.className = "status-dot ok";
    text.textContent = "Models Ready";

    // Header stat pills live in layout.html, so update them here too.
    $("hs-total").querySelector(".hstat-val").textContent = statsData.total?.toLocaleString() || "—";
    $("hs-spam").querySelector(".hstat-val").textContent  = statsData.spam?.toLocaleString()  || "—";
    $("hs-ham").querySelector(".hstat-val").textContent   = statsData.ham?.toLocaleString()   || "—";
  } catch {
    dot.className = "status-dot err";
    text.textContent = "Server Error";
  }
}

window.addEventListener("DOMContentLoaded", () => {
  checkServerStatus();
});
