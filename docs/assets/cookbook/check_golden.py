"""Check the starter golden questions: python check_golden.py."""

from pathlib import Path

from rag_titan.evaluation import check_golden

here = Path(__file__).resolve().parent
result = check_golden(here / "golden.json", {
    "handbook.md": (here / "handbook.md").read_text(encoding="utf-8"),
})
print(f"Questions: {result['question_count']}")
print(f"Human-validated: {result['validated_count']}")
print(f"Ready: {result['ready']}")
for problem in result["problems"]:
    print(problem)
raise SystemExit(0 if result["ready"] else 1)
