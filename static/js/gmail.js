/* ══════════════════════════════════════════════════════════════════════════
   SpamShield AI  —  gmail.js
   Live Gmail inbox fetch, preview, and analyze. Used by gmail.html.
   Calls into detector.js's runPredict / resetXai (loaded first on this page)
   to render results in this page's own Prediction Result + XAI card.
   ══════════════════════════════════════════════════════════════════════════ */

"use strict";

let gmailEmails = {};

window.addEventListener("DOMContentLoaded", () => {
  setupGmailPanel();
});

function setupGmailPanel() {
  const btnFetch   = $("btn-gmail-fetch");
  const btnAnalyze = $("btn-gmail-analyze");
  const selectEl   = $("gmail-email-select");
  const statusEl   = $("gmail-status");

  // Fetch emails
  btnFetch.addEventListener("click", async () => {
    const count   = $("gmail-count").value;
    const filter  = $("gmail-filter").value;
    const spinner = $("gmail-fetch-spinner");

    btnFetch.disabled = true;
    show(spinner);
    $("gmail-btn-text").textContent = "Fetching...";
    showId("gmail-status");
    statusEl.className = "gmail-status";
    statusEl.textContent = "Connecting to Gmail...";
    hide($("gmail-preview-card"));

    try {
      const data = await apiFetch(`${API.gmailFetch}?count=${count}&q=${encodeURIComponent(filter)}`);
      const emails = data.emails || [];

      if (!emails.length) {
        statusEl.textContent = "📭 No emails found for this filter.";
        hideId("gmail-dropdown-wrap");
        return;
      }

      gmailEmails = {};
      emails.forEach(e => { gmailEmails[e.id] = e; });

      selectEl.innerHTML = '<option value="">-- Choose an email to analyze --</option>';
      emails.forEach(e => {
        const opt = document.createElement("option");
        opt.value = e.id;
        const unread  = e.is_unread ? "🔵 " : "";
        const subject = (e.subject || "(No Subject)").substring(0, 45);
        const sender  = (e.sender_short || "").substring(0, 22);
        opt.textContent = `${unread}${subject}  ·  ${sender}`;
        opt.title = `From: ${e.sender}\nDate: ${e.date}\n\n${e.snippet}`;
        selectEl.appendChild(opt);
      });

      showId("gmail-dropdown-wrap");
      hideId("gmail-placeholder");
      btnAnalyze.disabled = true;
      statusEl.textContent = `✅ Fetched ${emails.length} emails — select one below`;
    } catch(e) {
      statusEl.className = "gmail-status error";
      statusEl.textContent = `❌ ${e.message}`;
    } finally {
      btnFetch.disabled = false;
      hide(spinner);
      $("gmail-btn-text").textContent = "Fetch Emails";
    }
  });

  // Select email → preview
  selectEl.addEventListener("change", () => {
    const id    = selectEl.value;
    const prev  = $("gmail-preview-card");
    if (!id) { hide(prev); btnAnalyze.disabled = true; return; }
    const email = gmailEmails[id];
    if (!email) return;
    $("gp-subject").textContent = email.subject || "(No Subject)";
    $("gp-sender").textContent  = email.sender  || "";
    $("gp-date").textContent    = email.date    || "";
    $("gp-body").textContent    = email.body    || email.snippet || "";
    show(prev);
    btnAnalyze.disabled = false;
  });

  // Analyze selected email
  btnAnalyze.addEventListener("click", async () => {
    const id    = selectEl.value;
    const email = gmailEmails[id];
    if (!email) return;
    const text       = email.body || email.snippet || "";
    const autoLabel  = $("auto-label-toggle")?.checked;

    btnAnalyze.disabled = true;
    try {
      if (autoLabel) {
        // Use /api/gmail/classify which can auto-label
        const res = await apiFetch(API.gmailClassify, {
          method:  "POST",
          headers: { "Content-Type": "application/json" },
          body:    JSON.stringify({
            message_id: id, text, model: window.getSelectedModel(), auto_label: true
          }),
        });
        if (res.auto_labeled) {
          const s = $("gmail-status");
          s.className  = "gmail-status";
          s.textContent = "✅ Email automatically labelled as Spam in Gmail";
        }
      }

      // Show prediction in this page's own result panel
      window.setCurrentText(text);
      $("msg-display").textContent = text;
      hideId("token-section");
      hideId("result-area");
      window.resetXai();
      await window.runPredict(text);
      document.querySelector(".content").scrollTo({ top: 0, behavior: "smooth" });
    } finally {
      btnAnalyze.disabled = false;
    }
  });
}
