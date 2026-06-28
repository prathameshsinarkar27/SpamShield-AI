/* ══════════════════════════════════════════════════════════════════════════
   SpamShield AI  —  analytics.js
   ══════════════════════════════════════════════════════════════════════════ */

"use strict";

window.addEventListener("DOMContentLoaded", () => {
  loadNlpIntelligence();
});

async function loadNlpIntelligence() {
  try {
    const data = await apiFetch("/api/nlp-intelligence");

    $("nlp-spam-chars").textContent = data.spam_length?.avg_chars ?? "—";
    $("nlp-spam-words").textContent = data.spam_length?.avg_words ?? "—";
    $("nlp-ham-chars").textContent  = data.ham_length?.avg_chars  ?? "—";
    $("nlp-ham-words").textContent  = data.ham_length?.avg_words  ?? "—";

    renderTermList("nlp-spam-unigrams", data.spam_unigrams, "spam");
    renderTermList("nlp-spam-bigrams",  data.spam_bigrams,  "spam");
    renderTermList("nlp-ham-unigrams",  data.ham_unigrams,  "ham");

    hideId("nlp-loading");
    showId("nlp-content");
  } catch(e) {
    $("nlp-loading").innerHTML =
      `<span style="color:var(--accent)">⚠ NLP intelligence error: ${escHtml(e.message)}</span>`;
  }
}

function renderTermList(elId, terms, tone) {
  const el = $(elId);
  if (!terms || !terms.length) {
    el.innerHTML = '<div class="loading-row" style="padding:.5rem 0">Not enough data — run with a larger dataset.</div>';
    return;
  }
  const maxCount = Math.max(...terms.map(t => t.count));
  el.innerHTML = terms.map(t => {
    const pct = Math.round((t.count / maxCount) * 100);
    return `<div class="nlp-term-row">
      <span class="nlp-term-label" title="${escHtml(t.term)}">${escHtml(t.term)}</span>
      <div class="nlp-term-bar-bg"><div class="nlp-term-bar-fill nlp-fill-${tone}" style="width:${pct}%"></div></div>
      <span class="nlp-term-count">${t.count}</span>
    </div>`;
  }).join("");
}
