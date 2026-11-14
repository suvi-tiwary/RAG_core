"""
Run the "complete guide" example for real and return its files.

The hand-written files live in docs/assets/cookbook/your-rag/ (the documents,
the questions, the reviewed golden.json and five small scripts). This module
runs every command of the guide on them, in a temporary folder, and saves
what the commands printed and wrote under your-rag/expected/. The docs page
quotes those files, so every line it shows is real output.

Used by build_examples.py; nothing here is part of the package.
"""

import contextlib
import io
import os
import runpy
import shutil
import sys
import tempfile
from pathlib import Path

from rag_titan.cli import main as rag_titan_cli

SOURCE = Path(__file__).resolve().parents[2] / "docs/assets/cookbook/your-rag"
EVALUATE = "rag-titan evaluate --input {} --output {} --k {}"

# the guide, command by command; a "# text" line is only a caption
FIRST_RUN = [
    "# 1. The minimum: one chunk, one question",
    EVALUATE.format("minimal.json", "minimal-report.json", 1),
    "# The complete data file: every input rag-titan accepts",
    EVALUATE.format("full-data.json", "full-report.json", 2),
    "# 2. Prepare your documents",
    "python prepare_docs.py",
    "# 3. Golden questions: rag-titan drafts, you review, then check",
    "rag-titan golden --questions questions.txt --files documents/travel_policy.md"
    " --output golden-draft.json",
    "python check_golden.py",
    "# 4. Your RAG answers the questions and saves what it did",
    "python my_rag.py",
    "# 5. Build the data file and evaluate",
    "python make_data.py",
    EVALUATE.format("my-data.json", "first-report.json", 3),
    "# 6. Add the chunking and embedding inputs, evaluate again",
    "python add_stages.py",
    EVALUATE.format("eval-data.json", "baseline.json", 3),
]
SECOND_RUN = [
    "# 7. Improve your RAG, then compare with the baseline",
    "python my_rag.py --improved",
    "python make_data.py",
    "python add_stages.py",
    EVALUATE.format("eval-data.json", "candidate.json", 3)
    + " --baseline baseline.json --fail-on-regression",
]
# files of the first run that the guide shows, saved so nobody has to run it to look
KEPT = ["golden-draft.json", "my_rag_logs.json", "my-data.json", "eval-data.json"]


def source_files():
    """Return {relative name: bytes} of the hand-written kit files."""
    files = {}
    for path in sorted(SOURCE.rglob("*")):
        name = path.relative_to(SOURCE).as_posix()
        if path.is_file() and not name.startswith(("expected/", "__pycache__/")):
            files[name] = path.read_bytes()
    return files


def run_command(command):
    """Run one guide command in this process and return what it printed."""
    words = command.split()
    output = io.StringIO()
    old_argv = sys.argv
    try:
        with contextlib.redirect_stdout(output):
            if words[0] == "rag-titan":
                rag_titan_cli(words[1:])
            else:
                sys.argv = words[1:]
                runpy.run_path(words[1], run_name="__main__")
    except SystemExit as stop:
        if stop.code not in (None, 0):
            raise ValueError(f"'{command}' failed: {stop.code}\n{output.getvalue()}") from stop
    finally:
        sys.argv = old_argv
    return output.getvalue()


def transcript(commands):
    """Run the commands in the current folder; return them with their output as text."""
    lines = []
    for command in commands:
        if command.startswith("#"):
            lines.append(f"\n{command}\n")
        else:
            lines.append(f"$ {command}\n{run_command(command)}")
    return "".join(lines)


def scorecard(report_markdown):
    """Return the scorecard table of a Markdown report (it has no date in it)."""
    lines = report_markdown.splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("**Overall"))
    end = next(index for index, line in enumerate(lines) if line.startswith("### "))
    return "\n".join(lines[start:end]).strip() + "\n"


def your_rag_files(minimal):
    """
    Return {"your-rag/<name>": bytes} for the kit: its sources and expected/ outputs.

    Args:
        minimal: bytes of the cookbook's minimal.json (the smallest valid input).
    """
    sources = dict(source_files(), **{"minimal.json": minimal})
    expected = {}
    home = os.getcwd()
    with tempfile.TemporaryDirectory() as folder:
        for name, content in sources.items():
            Path(folder, name).parent.mkdir(parents=True, exist_ok=True)
            Path(folder, name).write_bytes(content)
        os.chdir(folder)
        try:
            text = transcript(FIRST_RUN)
            for name in KEPT:
                expected[name] = Path(name).read_bytes()
            text += transcript(SECOND_RUN)
            expected["scorecard.md"] = scorecard(
                Path("candidate.md").read_text(encoding="utf-8")).encode()
        finally:
            os.chdir(home)
            shutil.rmtree(Path(folder, "__pycache__"), ignore_errors=True)
    expected["complete-run.txt"] = text.lstrip("\n").encode()

    files = {f"your-rag/{name}": content for name, content in sources.items()}
    files.update({f"your-rag/expected/{name}": content for name, content in expected.items()})
    return files
