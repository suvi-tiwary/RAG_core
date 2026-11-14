"""
Write a test page that draws every RAG type's FLOW with rag-flow.js and plays
real recorded runs on it (replayed from docs/assets/playground/presets.json).

Usage (from the repo root, no Ollama needed):
    python tools/playground/make_flow_demo.py /tmp/flow-demo.html
Then open the page in a browser. Click a step to see its input and output.
"""

import json
import sys
from pathlib import Path

from rag_titan.factory import RAG_TYPES, build_rag
from rag_titan.playground import replay_parts

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs" / "assets"
PRESETS = ASSETS / "playground" / "presets.json"

# (title, chunking preset, RAG type, question id): one run per diagram
RUNS = [
    ("Simple RAG: one search, one answer", "paragraph", "simple", "q_001"),
    ("Adaptive RAG: rewrites, searches wider, then answers", "paragraph", "adaptive", "q_005"),
    ("Adaptive RAG: rewrites, still not enough, gives up", "word", "adaptive", "q_002"),
]

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>RAG flow demo</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="{css}">
<style>
  body {{ font: 14px system-ui, sans-serif; margin: 24px; color: #0f172a; background: #fff; }}
  .run {{ display: grid; grid-template-columns: minmax(0, 600px) minmax(0, 1fr); gap: 24px;
          border-top: 1px solid #e2e8f0; padding: 16px 0; }}
  .answer {{ margin: 8px 0; font-weight: 600; }}
  button {{ font: inherit; padding: 2px 10px; }}
</style></head><body>
<h1>RAG flow demo</h1>
<p>Real {model} runs, replayed. Click a step for its input and output.</p>
<div id="runs"></div>
<script src="{steps_js}"></script><script src="{flow_js}"></script>
<script>
const DATA = {data};
DATA.runs.forEach((run) => {{
  const box = document.createElement("section"); box.className = "run";
  const left = document.createElement("div"), right = document.createElement("div");
  const title = document.createElement("h3"); title.textContent = run.title;
  const question = document.createElement("p"); question.textContent = "Q: " + run.question;
  const answer = document.createElement("p"); answer.className = "answer";
  answer.textContent = "A: " + run.answer;
  const replay = document.createElement("button"); replay.textContent = "Play again";
  const holder = document.createElement("div");
  left.append(title, question, holder, replay, answer); box.append(left, right);
  document.getElementById("runs").append(box);
  const flow = DATA.flows[run.rag_type];
  const label = (id) => flow.nodes.find((n) => n.id === id).label;
  const diagram = RagFlow.render(holder, flow, {{
    label: run.title,
    onSelect: (id, visits) => right.replaceChildren(RagSteps.details(label(id), visits)),
  }});
  const play = () => diagram.play(run.steps, 700).then(() => diagram.select("retrieve"));
  replay.addEventListener("click", play); play();
}});
</script></body></html>
"""


def main(out):
    presets = json.loads(PRESETS.read_text(encoding="utf-8"))
    questions = {q["id"]: q["question"] for q in presets["questions"]}
    flows, runs = {}, []

    for title, chunking, rag_type, question_id in RUNS:
        rag = build_rag(rag_type, replay_parts(presets, chunking))
        result = rag.run(questions[question_id])
        flows[rag_type] = rag.FLOW
        runs.append({"title": title, "rag_type": rag_type, "question": questions[question_id],
                     "answer": result["answer"], "steps": result["steps"]})

    data = json.dumps({"flows": flows, "runs": runs}, default=float).replace("</", "<\\/")
    page = PAGE.format(model=presets["model"], data=data,
                       css=(ASSETS / "stylesheets" / "rag-flow.css").as_uri(),
                       steps_js=(ASSETS / "javascripts" / "rag-steps.js").as_uri(),
                       flow_js=(ASSETS / "javascripts" / "rag-flow.js").as_uri())
    Path(out).write_text(page, encoding="utf-8")
    print(f"Wrote {out}: {len(runs)} runs of {sorted(flows)} ({len(RAG_TYPES)} RAG types)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python tools/playground/make_flow_demo.py OUT.html")
    main(sys.argv[1])
