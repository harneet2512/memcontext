const API = "http://localhost:8100/api/sessions/export";

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.action === "export_sessions") {
    chrome.cookies.getAll({}, (cookies) => {
      fetch(API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cookies }),
      })
        .then((r) => r.json())
        .then((data) => sendResponse({ ok: true, count: cookies.length, ...data }))
        .catch((e) => sendResponse({ ok: false, error: e.message }));
    });
    return true;
  }
});
