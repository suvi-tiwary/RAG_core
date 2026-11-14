// RAG playground (docs/rag-playground.md): pick chunkings, RAG types and a question, then
// watch each setup run on its own live flow diagram. Python runs in rag-worker.js;
// the cards and the report are drawn by rag-cards.js.

(() => {
  const app = document.getElementById("rag-playground-app");
  if (!app) return;
  const { el, choices, documentsBox, card, reportTable } = RagCards;

  const here = document.currentScript.src;   // .../assets/javascripts/rag-playground.js?v=...
  const assets = new URL("../", here).href;
  const worker = new Worker(new URL(`rag-worker.js${new URL(here).search}`, here), { type: "module" });
  const waiting = {};
  let calls = 0;
  const status = el("p", "rp-status", "Starting…");
  app.replaceChildren(status);

  worker.onmessage = ({ data }) => {
    if (data.status) { status.textContent = data.status; return; }
    const { resolve, reject } = waiting[data.id];
    delete waiting[data.id];
    if (data.error) reject(new Error(data.error)); else resolve(data.result);
  };
  // a worker that fails to load (no network, old browser) must not leave "Starting…" forever
  worker.onerror = (event) => {
    status.textContent = `The playground could not start: ${event.message || "the Python worker failed to load"}`;
  };
  const python = (call, ...args) => new Promise((resolve, reject) => {
    waiting[++calls] = { resolve, reject };
    worker.postMessage({ id: calls, call, args, assets });
  });

  // live mode: your Ollama, any question, the sample documents or your own text
  function liveControls() {
    const box = el("div", "rp-live");
    const address = Object.assign(el("input", "rp-input"), { value: "http://localhost:11434" });
    const connect = el("button", "md-button", "Connect");
    const model = el("select", "rp-input"), embedModel = el("select", "rp-input");
    const own = Object.assign(el("textarea", "rp-input rp-own"),
      { placeholder: "Optional: paste your own text here. Empty = the sample documents." });
    const help = el("p", "rp-help", "Work in progress: on the public website your browser may block "
      + "access to your Ollama. Start Ollama so this page may use it: OLLAMA_ORIGINS="
      + `${location.origin} ollama serve (on a Mac with the Ollama app: launchctl setenv `
      + `OLLAMA_ORIGINS ${location.origin}, then restart the app).`);
    const base = () => address.value.trim().replace(/\/$/, "");
    connect.onclick = async () => {
      help.textContent = "Connecting…";
      try {
        const names = await python("models", base());
        const pick = (select, list, preferred) => select.replaceChildren(...list.map((n) =>
          Object.assign(el("option", "", n), { value: n, selected: n.startsWith(preferred) })));
        pick(model, names.filter((n) => !n.includes("embed")), "gemma3:4b");
        pick(embedModel, names.filter((n) => n.includes("embed")), "nomic-embed-text");
        help.textContent = `Connected: ${names.length} models.`;
      } catch (error) { help.textContent = error.message; }
    };
    box.append(el("label", "rp-label", "Your Ollama"), address, connect, help,
      el("label", "rp-label", "Chat model"), model, el("label", "rp-label", "Embedding model"),
      embedModel, el("label", "rp-label", "Documents"), own);
    return { box, settings: () => [base(), model.value, embedModel.value],
             documents: () => (own.value.trim() ? [{ source: "your-text.md", text: own.value }] : null),
             ready: () => Boolean(model.value && embedModel.value) };
  }

  function build(info) {
    const mode = el("select", "rp-input rp-mode");
    mode.append(Object.assign(el("option", "", `Recorded runs (${info.model}, instant)`), { value: "recorded" }),
      Object.assign(el("option", "", "Live with your Ollama (work in progress)"), { value: "live" }));
    const live = liveControls();
    const question = Object.assign(el("input", "rp-input rp-question"),
      { value: info.questions[0].question, placeholder: "Ask a question" });
    const presets = Object.assign(el("datalist"), { id: "rp-presets" });
    info.questions.forEach((q) => presets.append(Object.assign(el("option"), { value: q.question })));
    question.setAttribute("list", "rp-presets");
    const chunkings = choices("Ingestion (chunker)", Object.keys(info.chunkings), (name) => {
      const s = info.chunkings[name];
      return ` ${s.chunk_size} ${name === "word" ? "words" : "characters"}, overlap ${s.chunk_overlap}`;
    }, ["word", "paragraph"]);
    const ragTypes = choices("RAG type", info.rag_types, () => "", info.rag_types);
    const runButton = el("button", "md-button md-button--primary", "Run");
    const evalButton = el("button", "md-button", "Evaluate all questions");
    const results = el("div", "rp-results");

    const ready = `Ready: real ${info.model} replies and ${info.embed_model} vectors, `
      + "replayed by the real rag-titan code running in your browser.";
    mode.onchange = () => {
      live.box.hidden = mode.value !== "live";
      evalButton.hidden = mode.value === "live";   // live evaluation would ask the model ~50 times
      status.textContent = mode.value === "live"
        ? "Live: the real rag-titan code in your browser, asking your own Ollama." : ready;
    };
    runButton.onclick = () => runAll(info, mode.value, live, question.value, chunkings.picked(),
                                     ragTypes.picked(), results);
    evalButton.onclick = async () => {
      results.replaceChildren(el("p", "rp-status", "Running every question on every setup…"));
      results.replaceChildren(...reportTable(await python("evaluate", chunkings.picked(), ragTypes.picked())));
    };
    mode.onchange();
    const controls = el("div", "rp-controls");
    controls.append(chunkings.box, ragTypes.box);
    app.append(documentsBox(info.documents), el("label", "rp-label", "Mode"), mode, live.box,
               el("label", "rp-label", "Question"), question, presets, controls,
               runButton, evalButton, results);
    runButton.onclick();
  }

  async function runAll(info, mode, live, question, chunkings, ragTypes, results) {
    const preset = info.questions.find((q) => q.question === question.trim());
    if (mode === "recorded" && !preset) {
      results.replaceChildren(el("p", "rp-status", "Recorded mode only knows the listed "
        + "questions: pick one, or switch to Live with your Ollama to ask anything."));
      return;
    }
    if (mode === "live" && !live.ready()) {
      results.replaceChildren(el("p", "rp-status", "Connect to your Ollama first."));
      return;
    }
    const cards = [];
    try {
      for (const chunking of chunkings) {
        for (const ragType of ragTypes) {
          results.replaceChildren(el("p", "rp-status", `Running ${ragType} + ${chunking}`
            + `${mode === "live" ? " with your Ollama (this can take a while)" : ""}…`));
          const run = mode === "live"
            ? await python("live_run", ...live.settings(), chunking, ragType, question, live.documents())
            : await python("run", chunking, ragType, preset.id);
          cards.push(card(info.flows[ragType], `${ragType} + ${chunking}`, run));
        }
      }
    } catch (error) {
      results.replaceChildren(el("p", "rp-status", `Could not run: ${error.message}`));
      return;
    }
    results.replaceChildren(...cards.map((c) => c.box));
    cards.forEach((c) => c.play());
  }

  python("info").then(build).catch((error) => {
    status.textContent = `The playground could not start: ${error.message}`;
  });
})();
