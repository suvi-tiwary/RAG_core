/* Local command builders. These generate text only; they never execute commands. */
(function () {
  "use strict";

  function quote(value, shell) {
    return "'" + (shell === "powershell" ? value.replace(/'/g, "''") :
      value.replace(/'/g, "'\"'\"'")) + "'";
  }

  function initialize(root, number) {
    if (root.dataset.ready) return;
    root.dataset.ready = "true";
    const prefix = "cb-command-" + number;
    root.innerHTML = `
      <p class="cb-eyebrow">YOUR COMMAND · GENERATED LOCALLY</p>
      <h3>Choose your files. Copy. Run in your terminal.</h3>
      <p>Paths are relative to the folder your terminal is in. This builder does not run Python.</p>
      <div class="cb-fields">
        <label>Terminal<select name="shell"><option value="bash">macOS / Linux — Bash or zsh</option>
          <option value="powershell">Windows — PowerShell</option></select></label>
        <label>Input data file<input name="input" type="text" spellcheck="false"></label>
        <label>Output report file<input name="output" type="text" spellcheck="false"></label>
        <label>K — results to score<input name="k" type="number" min="1" max="1000" step="1" value="3"></label>
      </div>
      <label class="cb-toggle"><input name="compare" type="checkbox"> Compare with a saved baseline report</label>
      <div class="cb-baseline" hidden>
        <label>Baseline report file<input name="baseline" type="text" value="baseline.json" spellcheck="false"></label>
        <label class="cb-toggle"><input name="fail" type="checkbox" checked> Exit with code 1 if metrics regress</label>
      </div>
      <p class="cb-error" role="alert"></p>
      <pre tabindex="0" aria-label="Generated CLI command"><code></code></pre>
      <div class="cb-actions"><button type="button" class="cb-primary">Copy command</button>
        <span class="cb-copy-status" role="status"></span></div>
      <p class="cb-output-note"></p>`;
    const get = name => root.querySelector(`[name="${name}"]`);
    root.querySelectorAll("input, select").forEach((field, i) => {
      field.id = prefix + "-" + i;
      field.closest("label").htmlFor = field.id;
    });
    get("input").value = root.dataset.input || "sample.json";
    get("output").value = root.dataset.output || "baseline.json";
    get("compare").checked = root.dataset.compare === "true";
    const code = root.querySelector("code");
    const copy = root.querySelector("button");
    const status = root.querySelector(".cb-copy-status");

    function update() {
      const comparing = get("compare").checked;
      root.querySelector(".cb-baseline").hidden = !comparing;
      const names = ["input", "output", ...(comparing ? ["baseline"] : [])];
      const paths = names.map(name => get(name).value.trim());
      const k = Number(get("k").value);
      let error = paths.some(path => !path || /[\r\n\0]/.test(path)) ?
        "Enter a non-empty file path on one line for each file." : "";
      if (!Number.isInteger(k) || k < 1 || k > 1000) error = "Choose a whole-number K from 1 to 1000.";
      if (new Set(paths).size !== paths.length) error = "Use different paths for input, output and baseline so files are preserved.";
      if (!/\.json$/i.test(get("output").value.trim())) error = "End the output filename with .json; a .md file is also saved.";
      root.querySelector(".cb-error").textContent = error;
      copy.disabled = Boolean(error);
      status.textContent = "";
      if (error) { code.textContent = "Fix the fields above to generate a command."; return; }
      const q = value => quote(value.trim(), get("shell").value);
      let command = `rag-titan evaluate --input ${q(get("input").value)} --output ${q(get("output").value)} --k ${k}`;
      if (comparing) command += ` --baseline ${q(get("baseline").value)}`;
      if (comparing && get("fail").checked) command += " --fail-on-regression";
      code.textContent = command;
      root.querySelector(".cb-output-note").textContent = "Creates " +
        get("output").value.trim() + " and " + get("output").value.trim().replace(/\.json$/i, ".md") +
        ". K changes how many logged results are scored; it does not run retrieval again.";
    }
    root.addEventListener("input", update);
    copy.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(code.textContent);
        status.textContent = "Copied. Paste into your terminal.";
      } catch (_) {
        const range = document.createRange();
        range.selectNodeContents(code);
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        status.textContent = "Command selected. Press Ctrl+C or ⌘C to copy.";
      }
    });
    update();
  }

  function mount() { document.querySelectorAll(".cb-command").forEach(initialize); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount);
  else mount();
})();
