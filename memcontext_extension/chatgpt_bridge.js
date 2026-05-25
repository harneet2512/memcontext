const MC_API = "http://localhost:8100";
let lastResults = [];

function createBridge() {
  if (document.getElementById("mc-bridge")) return;

  const el = document.createElement("div");
  el.id = "mc-bridge";
  el.innerHTML = `
    <div id="mc-panel">
      <div id="mc-panel-header">
        <div id="mc-panel-logo">M</div>
        <span id="mc-panel-title">MemContext</span>
        <span id="mc-panel-status">Connecting...</span>
      </div>
      <div id="mc-search">
        <input id="mc-query" type="text" placeholder="Query your memory..." />
        <button id="mc-search-btn">Search</button>
      </div>
      <div id="mc-results">
        <div id="mc-empty">Type a query to search your memory across all sessions and tools</div>
      </div>
      <div id="mc-inject-all" style="display:none">
        <button id="mc-inject-btn">Inject Context into Chat</button>
      </div>
    </div>
    <button id="mc-bridge-btn">M</button>
  `;
  document.body.appendChild(el);

  // Check server connection
  fetch(MC_API + "/health")
    .then((r) => r.json())
    .then(() => {
      document.getElementById("mc-panel-status").textContent = "Connected";
      document.getElementById("mc-panel-status").style.color = "#22C55E";
    })
    .catch(() => {
      document.getElementById("mc-panel-status").textContent = "Server offline";
      document.getElementById("mc-panel-status").style.color = "#EF4444";
    });

  // Toggle panel
  document.getElementById("mc-bridge-btn").addEventListener("click", () => {
    const panel = document.getElementById("mc-panel");
    const btn = document.getElementById("mc-bridge-btn");
    panel.classList.toggle("open");
    btn.classList.toggle("active");
    if (panel.classList.contains("open")) {
      document.getElementById("mc-query").focus();
    }
  });

  // Search
  document.getElementById("mc-search-btn").addEventListener("click", doSearch);
  document.getElementById("mc-query").addEventListener("keydown", (e) => {
    if (e.key === "Enter") doSearch();
  });

  // Inject all
  document.getElementById("mc-inject-btn").addEventListener("click", injectAll);
}

function doSearch() {
  const query = document.getElementById("mc-query").value.trim();
  if (!query) return;

  const results = document.getElementById("mc-results");
  results.innerHTML = '<div id="mc-empty">Searching...</div>';

  fetch(MC_API + "/api/memory/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: 8 }),
  })
    .then((r) => r.json())
    .then((data) => {
      lastResults = data.claims || [];
      const relevant = lastResults.filter((c) => c.score > 0);
      if (relevant.length === 0) {
        results.innerHTML =
          '<div id="mc-empty">No relevant context found</div>';
        document.getElementById("mc-inject-all").style.display = "none";
        return;
      }

      results.innerHTML = relevant
        .map(
          (c, i) => `
        <div class="mc-claim" data-idx="${i}" title="Click to copy">
          <div class="mc-claim-subject">${c.subject}</div>
          <div class="mc-claim-value">${c.value}</div>
          <div class="mc-claim-score">confidence: ${c.confidence} · score: ${c.score}</div>
        </div>`
        )
        .join("");

      document.getElementById("mc-inject-all").style.display = "block";

      // Click individual claim to copy
      results.querySelectorAll(".mc-claim").forEach((el) => {
        el.addEventListener("click", () => {
          const idx = parseInt(el.dataset.idx);
          const claim = relevant[idx];
          navigator.clipboard.writeText(
            `[${claim.subject}] ${claim.value}`
          );
          el.style.background = "rgba(34,197,94,0.1)";
          setTimeout(
            () => (el.style.background = ""),
            500
          );
        });
      });
    })
    .catch(() => {
      results.innerHTML =
        '<div id="mc-empty">Failed to query — is MemContext server running?</div>';
    });
}

function injectAll() {
  const relevant = lastResults.filter((c) => c.score > 0);
  if (relevant.length === 0) return;

  // Build context block
  const context = relevant
    .map((c) => `- [${c.subject}]: ${c.value}`)
    .join("\n");

  const block = `Here is relevant context from MemContext (my cross-tool memory):\n\n${context}\n\nBased on this context, `;

  // Find ChatGPT's input textarea and inject
  const textarea =
    document.querySelector("#prompt-textarea") ||
    document.querySelector("textarea") ||
    document.querySelector('[contenteditable="true"]');

  if (textarea) {
    if (textarea.tagName === "TEXTAREA") {
      textarea.value = block;
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
    } else {
      // contenteditable (ProseMirror)
      textarea.innerHTML = `<p>${block.replace(/\n/g, "<br>")}</p>`;
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
    }
    // Flash green on inject button
    const btn = document.getElementById("mc-inject-btn");
    btn.textContent = "Injected!";
    btn.style.background = "#16A34A";
    setTimeout(() => {
      btn.textContent = "Inject Context into Chat";
      btn.style.background = "#22C55E";
    }, 1500);
  }
}

// Initialize when page is ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", createBridge);
} else {
  createBridge();
}

// Re-inject on SPA navigation
const observer = new MutationObserver(() => {
  if (!document.getElementById("mc-bridge")) createBridge();
});
observer.observe(document.body, { childList: true, subtree: false });
