"""Draw docs/assets/images/eval-steps.svg: the complete evaluation guide as seven steps.

The two scores in the last step are read from the REAL run of the guide's example
(docs/assets/cookbook/your-rag/expected/complete-run.txt).

Usage: python tools/diagrams/make_eval_steps.py [--check]   (--check: fail if out of date)
"""

import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from make_eval_flow import STYLE, blink_at

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "docs" / "assets" / "cookbook" / "your-rag" / "expected" / "complete-run.txt"
OUT = ROOT / "docs" / "assets" / "images" / "eval-steps.svg"

ROW = 80          # height of one step, with the gap below it
TOP = 104         # y of the first step

# (who does it, title, two lines of detail, what comes out)
STEPS = [
    ("you", "Prepare your documents",
     ["profile_documents() finds empty, tiny and duplicated files",
      "keep the clean ones"], "clean documents"),
    ("you", "Write golden questions",
     ["question + correct answer + evidence quote",
      "rag-titan golden drafts them, you review every one"], "golden.json"),
    ("rag", "Ask YOUR RAG the same questions",
     ["any framework, model or vector database",
      "rag-titan is not involved here"], "answers"),
    ("rag", "Capture what it did",
     ["per question: retrieved ids, context ids, answer",
      "once: chunks + metadata, documents, vectors"], "my_rag_logs.json"),
    ("titan", "Build the data file",
     ["build_dataset() joins your logs with golden.json",
      "then add the chunking and embedding fields"], "eval-data.json"),
    ("titan", "rag-titan evaluate",
     ["scores 7 stages from 0 to 100, no LLM judge",
      "names the first failing stage"], "report.json + report.md"),
    ("you", "Read the scores, improve, run again",
     ["fix the first failing stage in your RAG",
      "--baseline shows what got better or worse"], "score {before} → {after}"),
]

SUBTITLE = "Left: each step and who does it. Right: what the step produces."
KINDS = {"you": ("neutral", "you"), "rag": ("search", "your RAG"), "titan": ("llm", "rag-titan")}

EXTRA_STYLE = """
.step { font: 700 15px system-ui, -apple-system, "Segoe UI", sans-serif; fill: #ffffff; }
.dot { fill: #4f46e5; }
.out { font: 600 13px ui-monospace, SFMono-Regular, Menlo, monospace; fill: #0f172a; }
.pill { fill: #eef2ff; stroke: #818cf8; stroke-width: 1.5; }
.who { font: 600 11px system-ui, -apple-system, "Segoe UI", sans-serif; fill: #64748b; }
.loop { fill: none; stroke: #4f46e5; stroke-width: 2; stroke-dasharray: 6 5;
        animation: dash 0.9s linear infinite; }
"""


def real_scores():
    """Return the overall scores of the baseline run and of the improved run."""
    scores = re.findall(r"^Overall score: ([0-9.]+)/100", RUN.read_text(encoding="utf-8"), re.M)
    if len(scores) < 2:
        raise ValueError(f"{RUN.name} has no two 'Overall score' lines: rebuild the cookbook")
    return scores[-2], scores[-1]


def draw_step(number, step, scores):
    """Return (css, svg lines) for one step: its box, its output and the arrow below."""
    who, title, lines, output = step
    kind, owner = KINDS[who]
    y = TOP + (number - 1) * ROW
    output = output.format(before=scores[0], after=scores[1])
    svg = [f'<rect class="{kind}" x="70" y="{y}" width="560" height="64" rx="12"/>',
           f'<circle class="dot" cx="102" cy="{y + 32}" r="15"/>',
           f'<text class="step" x="102" y="{y + 37}" text-anchor="middle">{number}</text>',
           f'<text class="label" x="130" y="{y + 22}">{escape(title)}</text>',
           f'<text class="who" x="614" y="{y + 22}" text-anchor="end">{owner}</text>']
    for index, line in enumerate(lines):
        svg.append(f'<text class="small" x="130" y="{y + 40 + index * 16}">{escape(line)}</text>')

    svg.append(f'<path class="flow" d="M634 {y + 32} H676" marker-end="url(#arrow)"/>')
    svg.append(f'<rect class="pill" x="680" y="{y + 12}" width="250" height="40" rx="20"/>')
    svg.append(f'<text class="out" x="805" y="{y + 37}" text-anchor="middle">'
               f'{escape(output)}</text>')
    if number < len(STEPS):
        svg.append(f'<path class="flow" d="M350 {y + 66} V{y + ROW - 4}" '
                   f'marker-end="url(#arrow)"/>')

    svg.append(f'<rect class="glow g{number}" x="67" y="{y - 3}" width="566" height="70" rx="14" '
               f'fill="none" stroke="#f97316" stroke-width="3"/>')
    return blink_at(f"g{number}", 0.4 + (number - 1) * 1.7), svg


def draw_loop():
    """Return the dashed arrow from the last step back to "Ask YOUR RAG"."""
    last = TOP + (len(STEPS) - 1) * ROW + 32
    ask = TOP + 2 * ROW + 32
    return [f'<path class="loop" d="M70 {last} H40 V{ask} H64" marker-end="url(#back)"/>',
            f'<text class="who" transform="translate(30 {(last + ask) // 2}) rotate(-90)" '
            f'text-anchor="middle">run again</text>']


def render():
    """Return the full SVG text."""
    scores = real_scores()
    css, body = STYLE + EXTRA_STYLE, []
    for number, step in enumerate(STEPS, start=1):
        step_css, lines = draw_step(number, step, scores)
        css += step_css
        body += lines
    body += draw_loop()
    height = TOP + len(STEPS) * ROW + 40
    description = ("Seven steps: prepare your documents, write golden questions, ask your RAG "
                   "the same questions, capture what it did, build the data file, run rag-titan "
                   f"evaluate, then read the scores, improve and run again. In the guide's "
                   f"example the overall score goes from {scores[0]} to {scores[1]}.")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 {height}"
     width="960" height="{height}" role="img" aria-labelledby="title desc">
  <title id="title">Evaluate your RAG with rag-titan, step by step</title>
  <desc id="desc">{escape(description)}</desc>
  <defs>
    <style>{css}</style>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
            orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#64748b"/></marker>
    <marker id="back" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
            orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#4f46e5"/></marker>
  </defs>
  <rect class="bg" x="0.5" y="0.5" width="959" height="{height - 1}" rx="16"/>
  <text class="title" x="30" y="44">Evaluate your RAG, step by step</text>
  <text class="small" x="30" y="70">{SUBTITLE}</text>
  {chr(10).join("  " + line for line in body).strip()}
  <text class="small" x="30" y="{height - 16}">Real scores from the guide's example: \
a small stand-in RAG before and after one improvement.</text>
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
