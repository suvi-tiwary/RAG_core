"""Draw the animated terminal demos of the evaluation cookbook.

It RUNS the real `rag-titan evaluate` and `inspect_report.py` on the cookbook
starter data, so every line in a demo is real output.

    python tools/diagrams/make_terminal_demos.py           # draw all demos
    python tools/diagrams/make_terminal_demos.py --check   # fail if any is out of date
    python tools/diagrams/make_terminal_demos.py --live    # redraw the Ollama demo (needs Ollama)
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from terminal_svg import render

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "assets" / "cookbook"
OUT = ROOT / "docs" / "assets" / "images" / "eval-terminal"

EVALUATE = "rag-titan evaluate --input {} --output {} --k {}"
REGRESSION = ("rag-titan evaluate --input {} --output {} --baseline baseline.json"
              " --k {} --fail-on-regression")

# file name -> (title, description, commands shown in the terminal)
DEMOS = {
    "first-evaluation": (
        "Your first evaluation",
        "Score the two logged handbook questions, then print the key metrics.",
        [EVALUATE.format("sample.json", "baseline.json", 3),
         "python inspect_report.py baseline.json"],
    ),
    "diagnose": (
        "Diagnose a bad answer",
        "Search missed the evidence: the report names retrieval as the first failing stage"
        " and suggests a fix.",
        [EVALUATE.format("missed-evidence.json", "missed-report.json", 3),
         "python inspect_report.py missed-report.json"],
    ),
    "regressions": (
        "Catch regressions",
        "Save a baseline, then compare a broken run against it; the command exits with code 1.",
        [EVALUATE.format("sample.json", "baseline.json", 3),
         REGRESSION.format("missed-evidence.json", "candidate.json", 3),
         "echo $?"],
    ),
    "own-rag": (
        "Evaluate your own RAG",
        "Score a minimal file of your own logs; missing measurements are reported as not measured.",
        [EVALUATE.format("minimal.json", "minimal-report.json", 1),
         "python inspect_report.py minimal-report.json"],
    ),
    "end-to-end": (
        "From golden questions to a scorecard",
        "Write golden questions, check them against the handbook, then run a live Ollama RAG"
        " that answers them; rag-titan scores every stage and names what to fix.",
        ["# 1. Write golden questions: question, reference answer, evidence quote",
         "head -n 8 golden.json",
         "# 2. Check them against your documents",
         "python check_golden.py",
         "# 3. Your RAG answers them; rag-titan scores every stage",
         "python live_evaluation.py",
         "# 4. See what to fix",
         "python inspect_report.py live-report.json"],
    ),
    "complete-guide": (
        "Evaluate your RAG, start to finish",
        "A small stand-in RAG answers five golden questions and logs what it did; the logs become"
        " one data file, and rag-titan scores every stage.",
        ["# Your RAG answers the golden questions and saves what it did",
         "python my_rag.py",
         "# Join the logs with golden.json, then add chunking and embedding inputs",
         "python make_data.py",
         "python add_stages.py",
         "# Score every stage",
         EVALUATE.format("eval-data.json", "baseline.json", 3)],
    ),
}

# demos that run in the complete guide's kit folder instead of the starter data
KIT = {"complete-guide": DATA / "your-rag"}

# demos that call Ollama: answers change between runs, so they are only
# redrawn with --live and never compared by --check
LIVE = {"end-to-end"}


def tool_path(name):
    """Find a command next to the running Python (the project's virtualenv)."""
    found = Path(sys.executable).parent / name
    if not found.exists():
        raise ValueError(f"'{name}' not found next to {sys.executable}; "
                         "run 'pip install -e .' first")
    return str(found)


def run_commands(commands, folder):
    """Run the commands in folder and return [(command, output lines)]."""
    results, last_code = [], 0
    for command in commands:
        if command.startswith("#") or command == "echo $?":
            results.append((command, [] if command.startswith("#") else [str(last_code)]))
            continue
        words = command.split()
        program = {"python": sys.executable, "head": "head"}.get(words[0]) or tool_path(words[0])
        done = subprocess.run([program] + words[1:], cwd=folder, capture_output=True, text=True)
        if done.stderr.strip():
            raise ValueError(f"'{command}' printed an error:\n{done.stderr}")
        last_code = done.returncode
        results.append((command, done.stdout.rstrip().splitlines()))
    return results


def build(name):
    """Run one demo in a fresh copy of the starter data and return its SVG text."""
    title, description, commands = DEMOS[name]
    with tempfile.TemporaryDirectory() as folder:
        if name in KIT:
            folder = shutil.copytree(KIT[name], Path(folder) / "kit",
                                     ignore=shutil.ignore_patterns("expected", "__pycache__"))
        else:
            extras = ["inspect_report.py", "check_golden.py", "live_evaluation.py", "handbook.md"]
            for path in list(DATA.glob("*.json")) + [DATA / extra for extra in extras]:
                shutil.copy(path, folder)
        demo = run_commands(commands, folder)
    return render(f"rag-titan: {title}", description, demo)


def main():
    """Draw every demo, or with --check report the ones that are out of date."""
    check = "--check" in sys.argv
    live = "--live" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    stale = []
    for name in DEMOS:
        if (name in LIVE) != live:
            continue
        path = OUT / f"{name}.svg"
        svg = build(name)
        if check:
            if not path.exists() or path.read_text(encoding="utf-8") != svg:
                stale.append(path.name)
            continue
        path.write_text(svg, encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")
    if stale:
        sys.exit(f"Out of date: {', '.join(stale)}. Run this script to redraw them.")


if __name__ == "__main__":
    main()
