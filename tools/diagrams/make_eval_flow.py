"""Draw docs/assets/images/eval-flow.svg from a REAL evaluator run on missed-evidence.json.

Usage: python tools/diagrams/make_eval_flow.py [--check]   (--check: fail if out of date)
"""

import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from rag_titan.evaluation import RAGEvaluator

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "assets" / "cookbook" / "missed-evidence.json"
OUT = ROOT / "docs" / "assets" / "images" / "eval-flow.svg"
LOOP = 14.0   # seconds of one animation loop

STAGES = [("chunking", "Chunking"), ("embedding", "Embedding"), ("retrieval", "Search"),
          ("reranking", "Reranking"), ("augmentation", "Context"), ("generation", "Answer"),
          ("end_to_end", "End to end")]

BOXES = [   # the three boxes on top: (title, two lines of detail, css class)
    ("1. Your RAG", ["any framework, model", "or vector database"], "neutral"),
    ("2. my-data.json", ["your logged outputs +", "golden labels"], "search"),
    ("3. rag-titan evaluate", ["7 stage evaluators,", "no LLM judge needed"], "llm"),
]

COMMAND = "rag-titan evaluate --input my-data.json --output report.json --k 3"
FOOTNOTE = ("Real scores: the cookbook's missed-evidence.json, where search misses"
            " the evidence for one question.")
DESCRIPTION = ("Your RAG logs its outputs into one JSON file with golden labels. rag-titan"
               " evaluate scores every stage from 0 to 100 and names the first failing stage"
               " with a hint. Real scores from the cookbook's missed-evidence example:"
               " overall {score:.1f}, first failing stage {stage}.")

STYLE = """
.bg { fill: #ffffff; stroke: #e2e8f0; }
.title { font: 700 22px system-ui, -apple-system, "Segoe UI", sans-serif; fill: #0f172a; }
.sub { font: 12.5px ui-monospace, SFMono-Regular, Menlo, monospace; fill: #475569; }
.label { font: 600 15px system-ui, -apple-system, "Segoe UI", sans-serif; fill: #0f172a; }
.small { font: 12.5px system-ui, -apple-system, "Segoe UI", sans-serif; fill: #475569; }
.num { font: 700 13px ui-monospace, SFMono-Regular, Menlo, monospace; fill: #0f172a; }
.big { font: 700 34px system-ui, -apple-system, "Segoe UI", sans-serif; fill: #0f172a; }
.warn { font: 700 13px system-ui, -apple-system, "Segoe UI", sans-serif; fill: #c2410c; }
.neutral, .search, .llm { stroke-width: 1.5; }
.neutral { fill: #f8fafc; stroke: #94a3b8; } .search { fill: #ecfeff; stroke: #06b6d4; }
.llm { fill: #fffbeb; stroke: #f59e0b; }
.card { fill: #f8fafc; stroke: #cbd5e1; stroke-width: 1; }
.track { fill: #e2e8f0; }
.flow { fill: none; stroke: #64748b; stroke-width: 2; stroke-dasharray: 6 5;
        animation: dash 0.9s linear infinite; }
.bar { transform-box: fill-box; transform-origin: left center; }
@keyframes dash { to { stroke-dashoffset: -11; } }
@media (prefers-reduced-motion: reduce) {
  .flow, .anim { animation: none !important; opacity: 1 !important; transform: none !important; }
  .glow { display: none; }
}
"""


def evaluate():
    """Run the real evaluator and return (stage scores, overall card, first failure)."""
    report = RAGEvaluator(k=3).evaluate(**json.loads(DATA.read_text(encoding="utf-8")))
    stages = report["scorecard"]["stages"]
    scores = [(label, stages[key]["score"], key) for key, label in STAGES]
    failures = report["root_causes"]["failures"]
    if not failures:
        raise ValueError("missed-evidence.json no longer fails: pick another example")
    return scores, report["scorecard"]["overall"], failures[0]


def show_at(name, start, kind="fade"):
    """CSS that hides an element until start (seconds), then fades or grows it in."""
    begin, end = 100 * start / LOOP, 100 * (start + 0.6) / LOOP
    hidden, shown = ("opacity: 0", "opacity: 1") if kind == "fade" else \
        ("transform: scaleX(0)", "transform: scaleX(1)")
    return (f"@keyframes {name} {{ 0%, {begin:.1f}% {{ {hidden}; }} "
            f"{end:.1f}%, 95% {{ {shown}; }} 100% {{ {hidden}; }} }}\n"
            f".{name} {{ animation: {name} {LOOP}s infinite; }}\n")


def blink_at(name, start):
    """CSS that shows an element for about one second, starting at start (seconds)."""
    points = [100 * (start + offset) / LOOP for offset in (0, 0.2, 1.1, 1.4)]
    return (f"@keyframes {name} {{ 0%, {points[0]:.1f}% {{ opacity: 0; }} "
            f"{points[1]:.1f}%, {points[2]:.1f}% {{ opacity: 1; }} "
            f"{points[3]:.1f}%, 100% {{ opacity: 0; }} }}\n"
            f".{name} {{ animation: {name} {LOOP}s infinite; }}\n")


def draw_boxes():
    """Return (css, svg) for the three boxes on top, lit up one after another."""
    css, svg = "", []
    for i, (title, lines, kind) in enumerate(BOXES):
        x = 30 + i * 310
        svg.append(f'<rect class="{kind}" x="{x}" y="96" width="280" height="92" rx="12"/>')
        svg.append(f'<text class="label" x="{x + 18}" y="126">{escape(title)}</text>')
        for j, line in enumerate(lines):
            svg.append(f'<text class="small" x="{x + 18}" y="{150 + j * 18}">{escape(line)}</text>')
        css += blink_at(f"g{i}", 0.3 + i * 0.9)
        svg.append(f'<rect class="glow g{i}" x="{x - 3}" y="93" width="286" height="98" rx="14" '
                   f'fill="none" stroke="#f97316" stroke-width="3"/>')
        if i < 2:
            svg.append(f'<path class="flow" d="M{x + 284} 142 H{x + 306}" '
                       f'marker-end="url(#arrow)"/>')
    svg.append('<path class="flow" d="M790 190 V222 H310 V236" marker-end="url(#arrow)"/>')
    return css, svg


def bar_color(score):
    """Green for good, amber for average, orange for stages that need work."""
    return "#22c55e" if score >= 80 else "#f59e0b" if score >= 60 else "#f97316"


def draw_scorecard(scores, failing):
    """Return (css, svg) for the stage bars, growing one by one."""
    css = ""
    svg = ['<rect class="card" x="30" y="240" width="560" height="296" rx="12"/>',
           '<text class="label" x="50" y="270">Score of every stage (0 to 100)</text>']
    for i, (label, score, key) in enumerate(scores):
        y = 292 + i * 34
        start = 3.2 + i * 0.5
        svg.append(f'<text class="small" x="50" y="{y + 14}">{escape(label)}</text>')
        svg.append(f'<rect class="track" x="150" y="{y}" width="340" height="18" rx="4"/>')
        svg.append(f'<rect class="bar anim b{i}" x="150" y="{y}" width="{3.4 * score:.1f}" '
                   f'height="18" rx="4" fill="{bar_color(score)}"/>')
        svg.append(f'<text class="num anim s{i}" x="502" y="{y + 14}">{score:.0f}</text>')
        css += show_at(f"b{i}", start, "grow") + show_at(f"s{i}", start + 0.4)
        if key == failing:
            css += show_at("fail", 7.2)
            svg.append(f'<rect class="anim fail" x="40" y="{y - 6}" width="540" height="30" '
                       f'rx="6" fill="none" stroke="#f97316" stroke-width="2.5" '
                       f'stroke-dasharray="6 4"/>')
    return css, svg


def draw_result(overall, failure, failing_label):
    """Return (css, svg) for the result card on the right."""
    hint = failure["suggestion"]
    svg = ['<g class="anim res">',
           '<rect class="card" x="620" y="240" width="310" height="296" rx="12"/>',
           '<text class="label" x="640" y="270">Result</text>',
           f'<text class="big" x="640" y="318">{overall["score"]:.1f}<tspan class="small"> / 100'
           f'</tspan></text>',
           f'<text class="small" x="640" y="344">{escape(overall["rating"])} · '
           f'{escape(overall["status"])}</text>',
           '<text class="warn" x="640" y="384">First failing stage:</text>',
           f'<text class="warn" x="640" y="404">{escape(failing_label)}</text>',
           '<text class="label" x="640" y="444">What to try</text>']
    line, y = "", 466
    for word in hint.split():   # wrap the hint into short lines
        if len(line) + len(word) > 34:
            svg.append(f'<text class="small" x="640" y="{y}">{escape(line)}</text>')
            line, y = "", y + 18
        line = f"{line} {word}".strip()
    svg.append(f'<text class="small" x="640" y="{y}">{escape(line)}</text>')
    svg.append('</g>')
    return show_at("res", 7.8), svg


def render():
    """Return the full SVG text."""
    scores, overall, failure = evaluate()
    failing_label = dict((key, label) for key, label in STAGES)[failure["stage"]]
    parts = [draw_boxes(), draw_scorecard(scores, failure["stage"]),
             draw_result(overall, failure, failing_label)]
    css = STYLE + "".join(part[0] for part in parts)
    body = "\n  ".join(line for part in parts for line in part[1])
    description = DESCRIPTION.format(score=overall["score"], stage=failing_label)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 590"
     width="960" height="590" role="img" aria-labelledby="title desc">
  <title id="title">Evaluate any RAG with rag-titan</title>
  <desc id="desc">{escape(description)}</desc>
  <defs>
    <style>{css}</style>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
            orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#64748b"/></marker>
  </defs>
  <rect class="bg" x="0.5" y="0.5" width="959" height="589" rx="16"/>
  <text class="title" x="30" y="44">Evaluate any RAG: your logs in, the stage to fix out</text>
  <text class="sub" x="30" y="68">{COMMAND}</text>
  {body}
  <text class="small" x="30" y="568">{escape(FOOTNOTE)}</text>
</svg>
"""


def main():
    """Draw the image, or with --check fail if it is out of date."""
    svg = render()
    if "--check" in sys.argv:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != svg:
            sys.exit(f"Out of date: {OUT.name}. Run this script to redraw it.")
        return
    OUT.write_text(svg, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
