// Readable details of RAG steps (the {"step", "input", "output", "ms"} dicts of run()).
// Like describe_steps() in Python: chunks are shown by id, file and score.
// Everything is set with textContent, so text from documents is never run as HTML.

const RagSteps = (() => {
  const LONG = 220;   // characters of a text to show

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function isChunks(value) {
    return Array.isArray(value) && value.every((item) => item && typeof item === "object" && "id" in item);
  }

  function short(text) {
    const flat = String(text).replace(/\s+/g, " ").trim();
    return flat.length > LONG ? `${flat.slice(0, LONG - 1)}…` : flat;
  }

  function chunkLine(chunk) {
    const score = "rerank_score" in chunk ? `rerank ${chunk.rerank_score.toFixed(3)}`
      : "score" in chunk ? chunk.score.toFixed(3) : "";
    return `${chunk.id} ${chunk.source || ""} ${score ? `(${score})` : ""}\n   “${short(chunk.text || "")}”`;
  }

  // One value: chunks one per line, yes/no for booleans, quoted text otherwise.
  function describe(value, brief) {
    if (isChunks(value)) {
      const count = `${value.length} chunk${value.length === 1 ? "" : "s"}`;
      return brief ? count : [count, ...value.map(chunkLine)].join("\n");
    }
    if (typeof value === "boolean") return value ? "yes" : "no";
    if (typeof value === "string") return `“${short(value)}”`;
    return short(JSON.stringify(value));
  }

  function io(label, text) {
    const line = el("p", "rf-io");
    line.append(el("b", "", `${label} `), document.createTextNode(text));
    return line;
  }

  // One step: "#6 retrieve · 0.01 ms", its inputs (chunk lists only counted) and its output.
  function step(visit) {
    const box = el("div", "rf-step");
    const head = el("div", "rf-step-head", `#${visit.number} ${visit.step.step} `);
    head.append(el("span", "", `· ${Number(visit.step.ms).toFixed(2)} ms`));
    const inputs = Object.entries(visit.step.input || {})
      .map(([key, value]) => `${key}=${describe(value, true)}`).join(", ");
    box.append(head, io("in", inputs || "–"), io("out", describe(visit.step.output, false)));
    return box;
  }

  // All the visits of one node, e.g. retrieve was steps #2 and #6 after a rewrite.
  function details(label, visits) {
    const box = el("div", "rf-details");
    if (!visits.length) {
      box.append(el("p", "rf-io", `${label}: not used in this run.`));
      return box;
    }
    visits.forEach((visit) => box.append(step(visit)));
    return box;
  }

  return { details, describe };
})();
