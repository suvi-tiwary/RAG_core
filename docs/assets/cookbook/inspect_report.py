"""Print a few useful metrics: python inspect_report.py baseline.json."""

import json
import sys
from pathlib import Path

METRICS = {
    "retrieval": ["recall_at_k", "mrr_at_k"],
    "augmentation": ["context_sufficient"],
    "generation": ["answer_f1", "citation_precision", "unsupported_number_rate"],
}


def main():
    """Show missing measurements explicitly, then print root-cause hints."""
    path = sys.argv[1] if len(sys.argv) > 1 else "baseline.json"
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    for stage, names in METRICS.items():
        for name in names:
            value = report.get(stage, {}).get("summary", {}).get(name)
            print(f"{stage}.{name}: {'not measured' if value is None else value}")
    print("First failing stage:", report["root_causes"]["first_failing_stage"] or "none")
    for failure in report["root_causes"]["failures"]:
        print(f"  {failure['stage']}: {failure['suggestion']}")


if __name__ == "__main__":
    main()
