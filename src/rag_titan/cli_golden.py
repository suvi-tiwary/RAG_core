"""
`rag-titan golden`: write a first draft of your golden questions.

    rag-titan golden --questions questions.txt --files policy.md --output golden.json

questions.txt holds one question per line (a JSON list of texts works too).
With --files, every question gets a suggested evidence sentence from them.
"""

import json
from pathlib import Path

from rag_titan.evaluation.golden_template import make_golden_template
from rag_titan.loaders import load_document


def add_golden_command(commands):
    """Add the `golden` command to the CLI's sub-commands."""
    golden = commands.add_parser("golden", help="write a golden questions template to fill in")
    golden.add_argument("--questions", required=True,
                        help="text file with one question per line, or a JSON list of texts")
    golden.add_argument("--files", nargs="+", help="your documents, to suggest evidence quotes")
    golden.add_argument("--output", default="golden.json", help="where to write the template")
    golden.add_argument("--force", action="store_true",
                        help="overwrite --output if it exists (you lose your edits)")
    golden.add_argument("--log-level", default="WARNING", help="DEBUG, INFO or WARNING")


def read_questions(path):
    """Return the question texts of a .txt file (one per line) or a JSON list."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"rag-titan golden: questions file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as error:
            raise SystemExit(f"rag-titan golden: {path} is not valid JSON: {error.msg}")
    return [line.strip() for line in text.splitlines() if line.strip()]


def run_golden(args):
    """Write the template and tell the user what to do next."""
    output = Path(args.output)
    if output.exists() and not args.force:
        # a reviewed golden file is hours of work: never overwrite it silently
        raise SystemExit(f"rag-titan golden: {output} already exists; "
                         "pick another --output or add --force")

    documents = None
    if args.files:
        loaded = [load_document(path) for path in args.files]
        documents = {document["source"]: document["text"] for document in loaded}

    try:
        records = make_golden_template(read_questions(args.questions), documents)
    except ValueError as error:
        raise SystemExit(f"rag-titan golden: {error}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    suggested = sum(bool(record["expected_evidence"]) for record in records)
    print(f"Wrote {output}: {len(records)} questions, {suggested} with a suggested evidence quote.")
    print("Next: for every question, check the quote, write expected_answer and "
          "required_facts, then set validated to true.")
