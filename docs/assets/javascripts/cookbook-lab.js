/* A visual walk through real, precomputed rag-titan reports. No model runs here. */
(function () {
  "use strict";
  const stages = [
    ["chunking", "Chunk", "Split documents into pieces", "The source is split into c1–c4. c1 holds the leave policy; c4 holds the remote-work policy."],
    ["embedding", "Embed", "Turn text into search vectors", "The teaching fixture uses tiny two-dimensional vectors. In a live run, your embedding model supplies the vectors."],
    ["retrieval", "Retrieve", "Find the evidence", "Check whether c1 appears in the first K retrieved results for the leave question."],
    ["reranking", "Rerank", "Put useful evidence first", "Compare the order before and after reranking. A reranker can reorder candidates; it cannot recover evidence absent from the candidate set."],
    ["augmentation", "Context", "Pack the prompt", "These are the chunks actually given to the language model. Retrieved evidence can still be lost while packing the context."],
    ["generation", "Answer", "Generate and cite an answer", "Compare the actual answer with the human-written reference. Word overlap and citation labels are useful checks, not proofs of factual correctness."],
    ["end_to_end", "Report", "Review the complete run", "Read coverage and individual metrics alongside the overall score. Missing measurements and a tiny question set limit what you can conclude."],
  ];
  const notes = {
    "sample": "Both answers match their references. For q1, the correct chunk starts at rank 2 and the reranker moves it to rank 1.",
    "missed-evidence": "q1 never retrieves c1. The context lacks the leave policy and the answer abstains. q2 still succeeds, so mean retrieval recall at K=3 is 0.5.",
    "wrong-answer": "Both questions receive the right evidence, but their answers are wrong and cite c3. Look at generation rather than assuming search failed.",
    "missing-labels": "Answers still exist, but their reference answers and several human labels were removed. Answer F1 is not measured; the score is provisional, not evidence of correctness.",
  };
  const ns = "http://www.w3.org/2000/svg";

  function diagram(root, select) {
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", "0 0 840 106");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Seven RAG stages, from chunking to the final report. Use the buttons below to inspect a stage.");
    stages.forEach((stage, index) => {
      const x = index * 120 + 4;
      const group = document.createElementNS(ns, "g");
      group.dataset.stage = stage[0];
      const rect = document.createElementNS(ns, "rect");
      Object.entries({x, y: 16, width: 108, height: 66, rx: 12}).forEach(([k, v]) => rect.setAttribute(k, v));
      group.append(rect);
      for (const [label, y, className] of [[String(index + 1).padStart(2, "0"), 39, "cb-svg-number"], [stage[1], 63, "cb-svg-label"]]) {
        const text = document.createElementNS(ns, "text");
        text.setAttribute("x", x + 54); text.setAttribute("y", y);
        text.setAttribute("class", className); text.textContent = label; group.append(text);
      }
      svg.append(group);
      if (index < 6) {
        const path = document.createElementNS(ns, "path");
        path.setAttribute("d", `M${x + 109} 49 h10 l-3 -3 m3 3 l-3 3`);
        svg.append(path);
      }
    });
    root.querySelector(".cb-diagram").append(svg);
    stages.forEach((stage, index) => {
      const button = document.createElement("button");
      button.type = "button"; button.textContent = stage[1];
      button.addEventListener("click", () => select(index));
      root.querySelector(".cb-stage-buttons").append(button);
    });
  }

  async function mount(root) {
    if (root.dataset.ready) return;
    root.dataset.ready = "true";
    try {
      const response = await fetch(new URL(root.dataset.source, document.baseURI));
      if (!response.ok) throw new Error("Snapshots unavailable");
      const data = await response.json();
      root.innerHTML = `
        <p class="cb-eyebrow">FOLLOW THE EVIDENCE</p><h3>Where did the answer go wrong?</h3>
        <div class="cb-fields"><label>Example run<select class="cb-case">
          <option value="sample">Correct answers</option><option value="missed-evidence">Search misses the evidence</option>
          <option value="wrong-answer">Right evidence, wrong answer</option><option value="missing-labels">Missing answer labels</option>
        </select></label><label>Score the first K results<select class="cb-k"><option>1</option><option>2</option><option selected>3</option></select></label></div>
        <p class="cb-scenario-note" aria-live="polite"></p>
        <div class="cb-diagram"></div><div class="cb-stage-buttons" role="group" aria-label="Inspect a RAG stage"></div>
        <div class="cb-step"><p class="cb-eyebrow cb-step-count"></p><h4></h4><p class="cb-explanation"></p><pre class="cb-evidence"></pre></div>
        <div class="cb-actions"><button type="button" class="cb-play cb-primary">Play walkthrough</button>
          <button type="button" class="cb-next">Next stage →</button><span class="cb-play-status" role="status"></span></div>
        <div class="cb-metrics" aria-live="polite"></div><p class="cb-overall"></p>
        <p class="cb-provenance"></p>`;
      root.querySelector(".cb-provenance").textContent = `Computed with rag-titan ${data.version} from the downloadable teaching data. These are saved results, not a live model run. No data leaves this page.`;
      let step = 0, timer = null;
      const choose = () => data.cases[root.querySelector(".cb-case").value];
      const run = () => choose().runs[root.querySelector(".cb-k").value];
      const play = root.querySelector(".cb-play");
      function stop() {
        clearInterval(timer); timer = null; play.textContent = "Play walkthrough";
        root.classList.remove("cb-playing");
      }
      function render() {
        const selected = stages[step], query = choose().query, result = run();
        root.querySelectorAll("svg g").forEach((g, i) => {
          g.classList.toggle("cb-active", i === step);
          g.classList.toggle("cb-failed", stages[i][0] === result.root_causes.first_failing_stage);
        });
        root.querySelectorAll(".cb-stage-buttons button").forEach((button, i) => button.setAttribute("aria-pressed", String(i === step)));
        root.querySelector(".cb-step-count").textContent = `STEP ${step + 1} OF 7 · ${selected[1]}`;
        root.querySelector(".cb-step h4").textContent = selected[2];
        root.querySelector(".cb-explanation").textContent = selected[3];
        const evidence = {
          chunking: "c1: Employees get 25 days of annual leave.\nc2: Sick pay is full salary for 10 days.\nc3: Travel needs manager approval.\nc4: Remote work is allowed 3 days a week.",
          embedding: "c1 vector: [0.95, 0.05]\nq1 vector: [1.0, 0.0]\nToy vectors provided in sample.json, not model-generated.",
          retrieval: `q1 expected: c1\nRetrieved: ${query.retrieved_ids.join(" → ")}\nK = ${root.querySelector(".cb-k").value} (only the first K are scored)`,
          reranking: `Before: ${query.retrieved_ids.join(" → ")}\nAfter:  ${query.reranked_ids.join(" → ")}`,
          augmentation: `q1 prompt contains: ${query.context_ids.join(", ")}\nGold evidence: c1`,
          generation: `Actual: ${query.answer}\nReference: ${query.expected_answer || "not supplied"}\nCitations: ${query.citations.join(", ") || "none"}`,
          end_to_end: `First failing stage: ${result.root_causes.first_failing_stage || "none"}\nStatus: ${result.overall.status}\nHints are threshold-based, not proof of a cause.`,
        };
        root.querySelector(".cb-evidence").textContent = evidence[selected[0]];
        root.querySelector(".cb-scenario-note").textContent = notes[root.querySelector(".cb-case").value];
        const metrics = [["Retrieval recall", result.summaries.retrieval.recall_at_k],
          ["Answer F1", result.summaries.generation.answer_f1], ["Citation precision", result.summaries.generation.citation_precision]];
        root.querySelector(".cb-metrics").replaceChildren(...metrics.map(([label, value]) => {
          const tile = document.createElement("div"), name = document.createElement("span"), score = document.createElement("strong");
          name.textContent = label; score.textContent = value == null ? "Not measured" : value.toFixed(2);
          tile.append(name, score); return tile;
        }));
        root.querySelector(".cb-overall").textContent = `Overall: ${result.overall.score.toFixed(1)}/100 · Coverage: ${(result.overall.evaluation_coverage * 100).toFixed(0)}% · ${result.overall.status}. Metrics above are means across both questions.`;
      }
      diagram(root, index => { stop(); step = index; render(); });
      root.querySelectorAll("select").forEach(select => select.addEventListener("change", () => { stop(); render(); }));
      root.querySelector(".cb-next").addEventListener("click", () => { stop(); step = (step + 1) % 7; render(); });
      play.addEventListener("click", () => {
        if (timer) { stop(); return; }
        step = 0; render(); play.textContent = "Pause walkthrough"; root.classList.add("cb-playing");
        root.querySelector(".cb-play-status").textContent = "Advances every 4 seconds. Pause or select any stage to read at your own pace.";
        timer = setInterval(() => { if (step === 6) { stop(); return; } step++; render(); }, 4000);
      });
      document.addEventListener("visibilitychange", () => { if (document.hidden) stop(); });
      render();
    } catch (_) {
      const message = document.createElement("p");
      message.textContent = "The interactive example could not load. Use the downloadable data and static walkthrough below.";
      root.append(message);
    }
  }
  function start() { document.querySelectorAll(".cb-lab").forEach(mount); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
