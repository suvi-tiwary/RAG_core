// RAG flow diagrams: draw any RAG type's FLOW, then light up the steps of a real run.
//   const diagram = RagFlow.render(element, AdaptiveRAG_FLOW, { onSelect });
//   diagram.play(result.steps);            // or diagram.show(steps) without animation
// The layout comes from FLOW alone, so a new RAG type needs no diagram code.

const RagFlow = (() => {
  const SIZE = { w: 150, h: 46, gapX: 26, gapY: 40, pad: 14, loop: 54 };
  const NS = "http://www.w3.org/2000/svg";
  let diagrams = 0;   // gives every diagram its own arrow-marker id

  function svg(tag, attrs, parent) {
    const node = document.createElementNS(NS, tag);
    Object.entries(attrs || {}).forEach(([key, value]) => node.setAttribute(key, value));
    if (parent) parent.append(node);
    return node;
  }

  // Edges that go back to an earlier step (loops, like rewrite -> retrieve).
  function backEdges(flow) {
    const next = {};
    flow.edges.forEach(([from, to]) => (next[from] = next[from] || []).push(to));
    const back = new Set();
    const state = {};   // 1 = on the current path, 2 = done
    const visit = (id) => {
      state[id] = 1;
      (next[id] || []).forEach((to) => {
        if (state[to] === 1) back.add(`${id}>${to}`);
        else if (!state[to]) visit(to);
      });
      state[id] = 2;
    };
    flow.nodes.forEach((node) => { if (!state[node.id]) visit(node.id); });
    return back;
  }

  // Row of every node: the longest path from the first step, loops ignored.
  function rows(flow, back) {
    const row = Object.fromEntries(flow.nodes.map((node) => [node.id, 0]));
    const forward = flow.edges.filter(([a, b]) => !back.has(`${a}>${b}`));
    flow.nodes.forEach(() => forward.forEach(([a, b]) => { row[b] = Math.max(row[b], row[a] + 1); }));
    return row;
  }

  function layout(flow) {
    const back = backEdges(flow);
    const row = rows(flow, back);
    const loopStarts = new Set([...back].map((edge) => edge.split(">")[0]));
    const byRow = [];
    flow.nodes.forEach((node) => (byRow[row[node.id]] = byRow[row[node.id]] || []).push(node.id));
    // a loop leaves from the right side, so its start goes last in its row
    byRow.forEach((ids) => ids.sort((a, b) => loopStarts.has(a) - loopStarts.has(b)));

    const widest = Math.max(...byRow.map((ids) => ids.length));
    const inner = widest * SIZE.w + (widest - 1) * SIZE.gapX;
    const pos = {};
    byRow.forEach((ids, r) => {
      const used = ids.length * SIZE.w + (ids.length - 1) * SIZE.gapX;
      ids.forEach((id, i) => {
        pos[id] = { x: SIZE.pad + (inner - used) / 2 + i * (SIZE.w + SIZE.gapX),
                    y: SIZE.pad + r * (SIZE.h + SIZE.gapY) };
      });
    });
    const height = SIZE.pad * 2 + byRow.length * SIZE.h + (byRow.length - 1) * SIZE.gapY;
    return { pos, back, width: SIZE.pad * 2 + inner + (back.size ? SIZE.loop : 0), height };
  }

  function edgePath(a, b, isLoop) {
    if (isLoop) {   // out of the right side, round, into the right side of the earlier step
      const x = Math.max(a.x, b.x) + SIZE.w + SIZE.loop - 8;
      return `M${a.x + SIZE.w} ${a.y + SIZE.h / 2} C${x} ${a.y + SIZE.h / 2} ${x} ${b.y + SIZE.h / 2} `
        + `${b.x + SIZE.w + 4} ${b.y + SIZE.h / 2}`;
    }
    const [x1, y1, x2, y2] = [a.x + SIZE.w / 2, a.y + SIZE.h, b.x + SIZE.w / 2, b.y - 4];
    const mid = (y1 + y2) / 2;
    return `M${x1} ${y1} C${x1} ${mid} ${x2} ${mid} ${x2} ${y2}`;
  }

  // one arrowhead per edge colour: plain, loop and taken
  function arrowheads(root, number) {
    const defs = svg("defs", {}, root);
    const ids = {};
    ["plain", "loop", "taken"].forEach((kind) => {
      ids[kind] = `rf-arrow-${number}-${kind}`;
      const head = svg("marker", { id: ids[kind], viewBox: "0 0 10 10", refX: 9, refY: 5,
                                   markerWidth: 7, markerHeight: 7, orient: "auto" }, defs);
      svg("path", { d: "M0 0 L10 5 L0 10 z", class: `rf-head rf-head-${kind}` }, head);
    });
    return ids;
  }

  function render(container, flow, options) {
    const opts = options || {};
    const { pos, back, width, height } = layout(flow);
    // natural size (width attribute), shrinking on narrow screens (CSS max-width)
    const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, width, class: "rf-diagram",
                              role: "img", "aria-label": opts.label || "RAG flow" });
    const marker = arrowheads(root, ++diagrams);

    const edges = {};
    flow.edges.forEach(([from, to]) => {
      const isLoop = back.has(`${from}>${to}`);
      const edge = svg("path", { d: edgePath(pos[from], pos[to], isLoop),
        class: `rf-edge${isLoop ? " rf-loop" : ""}`,
        "marker-end": `url(#${isLoop ? marker.loop : marker.plain})` }, root);
      edge.dataset.taken = `url(#${marker.taken})`;
      edges[`${from}>${to}`] = edge;
    });

    const nodes = {};
    flow.nodes.forEach((node) => {
      const group = svg("g", { class: `rf-node rf-${node.kind}`, tabindex: 0, role: "button",
        transform: `translate(${pos[node.id].x} ${pos[node.id].y})` }, root);
      svg("rect", { width: SIZE.w, height: SIZE.h, rx: 10 }, group);
      svg("text", { x: SIZE.w / 2, y: SIZE.h / 2 + 5, class: "rf-label" }, group).textContent = node.label;
      const badge = svg("text", { x: SIZE.w - 8, y: 13, class: "rf-badge" }, group);
      const pick = () => opts.onSelect && opts.onSelect(node.id, nodes[node.id].visits);
      group.addEventListener("click", pick);
      group.addEventListener("keydown", (event) => { if (event.key === "Enter") pick(); });
      nodes[node.id] = { group, badge, visits: [] };
    });

    container.replaceChildren(root);
    return diagramApi(nodes, edges, opts.onSelect);
  }

  function diagramApi(nodes, edges, onSelect) {
    let timer = null;
    const reset = () => {
      clearTimeout(timer);
      Object.values(nodes).forEach((n) => { n.visits = []; n.badge.textContent = "";
        n.group.classList.remove("rf-visited", "rf-current"); });
      Object.values(edges).forEach((edge) => {
        edge.classList.remove("rf-taken");
        if (edge.dataset.plain) edge.setAttribute("marker-end", edge.dataset.plain);
      });
    };
    const light = (steps, i) => {
      const step = steps[i];
      const node = nodes[step.step];
      if (!node) throw new Error(`Step "${step.step}" is not in this FLOW`);
      Object.values(nodes).forEach((n) => n.group.classList.remove("rf-current"));
      node.visits.push({ number: i + 1, step });
      node.badge.textContent = node.visits.map((v) => v.number).join(" · ");
      node.group.classList.add("rf-visited", "rf-current");
      const edge = i > 0 && edges[`${steps[i - 1].step}>${step.step}`];
      if (edge) {
        edge.dataset.plain = edge.dataset.plain || edge.getAttribute("marker-end");
        edge.classList.add("rf-taken");
        edge.setAttribute("marker-end", edge.dataset.taken);
      }
    };
    const still = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    return {
      reset,
      // open a node's details from code, as if it was clicked
      select(id) { if (onSelect && nodes[id]) onSelect(id, nodes[id].visits); },
      show(steps) { reset(); steps.forEach((_, i) => light(steps, i)); },
      play(steps, delay) {
        reset();
        if (still()) { steps.forEach((_, i) => light(steps, i)); return Promise.resolve(); }
        return new Promise((done) => {
          const next = (i) => {
            if (i === steps.length) { done(); return; }
            light(steps, i);
            timer = setTimeout(() => next(i + 1), delay || 650);
          };
          next(0);
        });
      },
    };
  }

  return { render, layout };
})();
