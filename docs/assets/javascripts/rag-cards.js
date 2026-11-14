// Building blocks of the RAG playground page: controls, one card per run, the evaluate report.
// Text from documents and models is always set with textContent, never as HTML.

const RagCards = (() => {
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  // a group of checkboxes; picked() returns the ticked values
  function choices(title, names, describe, checked) {
    const box = el("fieldset", "rp-choices");
    box.append(el("legend", "", title));
    const inputs = names.map((name) => {
      const input = Object.assign(el("input"), { type: "checkbox", value: name,
                                                 checked: checked.includes(name) });
      const label = el("label", "rp-choice");
      label.append(input, ` ${name}`, el("span", "", describe(name)));
      box.append(label);
      return input;
    });
    return { box, picked: () => inputs.filter((i) => i.checked).map((i) => i.value) };
  }

  function documentsBox(documents) {
    const box = el("details", "rp-documents");
    box.append(el("summary", "", `The ${documents.length} sample documents`));
    documents.forEach((doc) => box.append(el("h4", "", doc.source), el("pre", "", doc.text)));
    return box;
  }

  function stats(run) {
    const evidence = run.recall === null ? "evidence not scored (your own question or text)"
      : `found ${Math.round(run.recall * 100)}% of the evidence`;
    const files = [...new Set(run.sources)].join(", ") || "none";
    return `${evidence} · ${run.steps.length} steps · ${Math.round(run.ms)} ms · files given to the LLM: ${files}`;
  }

  // one setup's run: title, live diagram, answer, numbers, and the details of a clicked step
  function card(flow, title, run) {
    const box = el("div", "rp-card");
    const holder = el("div"), details = el("div", "rp-details");
    const label = (id) => flow.nodes.find((node) => node.id === id).label;
    box.append(el("h3", "", title), holder,
      el("p", run.answered ? "rp-answer" : "rp-answer rp-none", run.answer),
      el("p", "rp-stats", stats(run)), details);
    const diagram = RagFlow.render(holder, flow, {
      label: title,
      onSelect: (id, visits) => details.replaceChildren(RagSteps.details(label(id), visits)),
    });
    return { box, play: () => diagram.play(run.steps, 650).then(() => diagram.select("retrieve")) };
  }

  function reportTable(report) {
    const table = el("table", "rp-report");
    const row = (cells, tag) => {
      const tr = el("tr");
      cells.forEach((cell) => tr.append(el(tag, "", cell)));
      table.append(tr);
    };
    row(["setup", "Recall", "MRR", "nDCG", "Answered", "Steps"], "th");
    report.results.forEach((r) => row([r.name, r.recall.toFixed(3), r.mrr.toFixed(3),
      r.ndcg.toFixed(3), `${Math.round(r.answered * 100)}%`, r.steps.toFixed(1)], "td"));
    const weak = report.results.flatMap((r) => r.rows.filter((q) => q.recall < 1 || !q.answered)
      .map((q) => `${r.name}: ${q.id} · recall ${q.recall.toFixed(2)}`
        + `${q.answered ? "" : " · not answered"} · found ${[...new Set(q.sources)].join(", ") || "nothing"}`));
    const list = el("ul", "rp-weak");
    (weak.length ? weak : ["Every question found all its evidence."])
      .forEach((line) => list.append(el("li", "", line)));
    return [table, el("p", "rp-stats", "Questions to look at:"), list];
  }

  return { el, choices, documentsBox, card, reportTable };
})();
