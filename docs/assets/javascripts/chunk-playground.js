// Chunk playground (docs/playground.md): the controls and the live output.
// The chunking rules live in chunk-core.js (ChunkCore), which mirrors the Python package.

const PLAYGROUND = {
  // example texts, loaded with one click; the first one is shown at the start
  presets: {
    "Leave policy": "Leave policy\n\nStaff get twenty days. Unused days carry over. "
      + "Ask HR for the form. Sick leave is separate.",
    "Markdown with headings": "# Leave policy\n\n## Annual leave\n\nStaff get twenty days. "
      + "Unused days carry over.\n\n## Sick leave\n\nTell your manager before nine. "
      + "A doctor's note is needed after three days.",
    "Tricky sentences": "Dr. Smith met J. K. Rowling at 3.5 p.m. on Monday. She said: \"Hi!\" "
      + "Then they left... Steps: 1. Open the form. 2. Sign it. Fruit, e.g. apples, is fine. "
      + "The U.S. Army came. 東京は大きい。大阪も大きい。",
    "One long paragraph": "Every member of staff gets twenty days of paid leave each year, "
      + "and the days are added on the first of January. Days that are not used by the end "
      + "of the year move to the next year, up to a limit of five days, and anything above "
      + "that limit is lost unless your manager agrees in writing before the end of March. "
      + "Ask HR for the carry-over form if you need it.",
  },
  // per mode: [default size, default overlap, max size]
  modes: {
    recursive: [50, 25, 400],
    sentence: [50, 25, 400],
    paragraph: [60, 0, 400],
    character: [18, 4, 400],
    word: [4, 1, 60],
  },
};

function el(tag, props, children) {
  const node = Object.assign(document.createElement(tag), props || {});
  (children || []).forEach((child) => node.append(child));
  return node;
}

function slider(label, value, max) {
  const input = el("input", { type: "range", min: 0, max, value });
  const shown = el("output", { textContent: value });
  input.addEventListener("input", () => { shown.textContent = input.value; });
  return { input, shown, row: el("label", { className: "cp-field" }, [label, " ", shown, input]) };
}

function chunkBlock(chunk, index, unit) {
  const units = Array.from(chunk.text);
  const shared = el("mark", { textContent: units.slice(0, chunk.overlap).join("") });
  const rest = document.createTextNode(units.slice(chunk.overlap).join(""));
  const head = el("div", { className: "cp-head", textContent: `C${index} · ${unit.count(chunk.text)} ${unit.name}` });
  const body = el("div", { className: "cp-text" }, chunk.overlap ? [shared, rest] : [rest]);
  return el("div", { className: `cp-chunk cp-c${index % 6}` }, [head, body]);
}

function statsText(mode, text, size, overlap, chunks) {
  const unit = ChunkCore.UNITS[mode];
  const n = unit.count(text);
  const k = chunks.length;
  const h = size - overlap;
  const lines = [`N = ${n} ${unit.name} · S = ${size} · O = ${overlap} · H = S − O = ${h} · K = ${k} chunks`];
  if (unit.packed) {
    const stored = chunks.reduce((sum, c) => sum + Array.from(c.text).length, 0);
    lines.push(`stored ${stored} characters for ${n} (overlap repeats whole ${unit.packed} only)`);
  } else if (n > size) {
    const plain = ChunkCore.countWindows(n, size, 0);
    lines.push(`K = 1 + ⌈(N − S) / H⌉ = 1 + ⌈(${n} − ${size}) / ${h}⌉ = ${k}`);
    lines.push(`without overlap: ${plain} chunks, so overlap adds ${Math.round((k / plain - 1) * 100)}%`);
  }
  return lines.join("\n");
}

function presetButtons(text, render) {
  const buttons = Object.entries(PLAYGROUND.presets).map(([name, value]) => {
    const button = el("button", { type: "button", className: "cp-preset", textContent: name });
    button.addEventListener("click", () => { text.value = value; render(); });
    return button;
  });
  return el("div", { className: "cp-presets" }, ["examples: ", ...buttons]);
}

function buildPlayground(root) {
  root.dataset.ready = "1";
  const mode = el("select", {}, Object.keys(PLAYGROUND.modes).map(
    (name) => el("option", { value: name, textContent: name })));
  const size = slider("size", 50, 400);
  const overlap = slider("overlap", 25, 49);
  const firstPreset = Object.values(PLAYGROUND.presets)[0];
  const text = el("textarea", { rows: 6, value: firstPreset, spellcheck: false });
  const stats = el("pre", { className: "cp-stats" });
  stats.setAttribute("aria-live", "polite");
  const output = el("div", { className: "cp-chunks" });

  function render() {
    size.input.min = 1;
    overlap.input.max = Number(size.input.value) - 1;
    overlap.shown.textContent = overlap.input.value;
    const s = Number(size.input.value);
    const o = Number(overlap.input.value);
    const chunks = ChunkCore.UNITS[mode.value].run(text.value, s, o);
    stats.textContent = statsText(mode.value, text.value, s, o, chunks);
    const unit = ChunkCore.UNITS[mode.value];
    output.replaceChildren(...chunks.map((chunk, i) => chunkBlock(chunk, i, unit)));
  }

  mode.addEventListener("change", () => {
    const [s, o, max] = PLAYGROUND.modes[mode.value];
    size.input.max = max;
    size.input.value = size.shown.textContent = s;
    overlap.input.max = s - 1;
    overlap.input.value = overlap.shown.textContent = o;
    render();
  });
  [size.input, overlap.input, text].forEach((input) => input.addEventListener("input", render));

  const controls = el("div", { className: "cp-controls" },
    [el("label", { className: "cp-field" }, ["chunker ", mode]), size.row, overlap.row]);
  root.replaceChildren(controls, presetButtons(text, render), text, stats, output);
  render();
}

if (typeof document !== "undefined") {
  const start = () => {
    const root = document.getElementById("chunk-playground-app");
    if (root && !root.dataset.ready) buildPlayground(root);
  };
  if (typeof document$ !== "undefined") document$.subscribe(start);
  else document.addEventListener("DOMContentLoaded", start);
}
