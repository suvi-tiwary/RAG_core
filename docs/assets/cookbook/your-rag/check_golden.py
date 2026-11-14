"""Check golden.json against the document before you trust any score."""

from pathlib import Path

from rag_titan.evaluation import check_golden

document = Path("documents/travel_policy.md").read_text(encoding="utf-8")
result = check_golden("golden.json", {"travel_policy.md": document})

print(f"Questions: {result['question_count']}")
print(f"Human-validated: {result['validated_count']}")
print(f"Ready: {result['ready']}")
for problem in result["problems"]:
    print(problem)
raise SystemExit(0 if result["ready"] else 1)
