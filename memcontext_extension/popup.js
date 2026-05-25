const btn = document.getElementById("btn");
const status = document.getElementById("status");

btn.addEventListener("click", () => {
  btn.disabled = true;
  btn.textContent = "Exporting...";
  status.style.display = "none";

  chrome.runtime.sendMessage({ action: "export_sessions" }, (res) => {
    status.style.display = "block";
    if (res && res.ok) {
      status.className = "ok";
      status.textContent = `Exported ${res.count} cookies to Agent Browser`;
      btn.textContent = "Exported ✓";
      setTimeout(() => {
        btn.disabled = false;
        btn.textContent = "Export Sessions to Agent";
      }, 3000);
    } else {
      status.className = "err";
      status.textContent = `Failed: ${res ? res.error : "Agent server not running on :8100"}`;
      btn.disabled = false;
      btn.textContent = "Export Sessions to Agent";
    }
  });
});
