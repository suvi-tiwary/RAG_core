"""Tests for `rag-titan evaluate --input`: score a data file, save reports, detect regressions."""

import json
from pathlib import Path

import pytest

from rag_titan import cli
from rag_titan.evaluation import evaluate_file
from rag_titan.evaluation.reports import summary_text

EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture
def data_file(rag_dataset, tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps(rag_dataset), encoding="utf-8")
    return path


def test_example_data_file_matches_the_test_dataset(rag_dataset):
    sample = json.loads((EXAMPLES / "rag_evaluation_sample.json").read_text(encoding="utf-8"))
    assert sample == rag_dataset


def test_evaluate_input_saves_both_reports(data_file, tmp_path, capsys):
    output = tmp_path / "out" / "report.json"
    cli.main(["evaluate", "--input", str(data_file), "--output", str(output), "--k", "3"])
    out = capsys.readouterr().out

    assert f"Reports saved: {output} and {output.with_suffix('.md')}" in out
    assert "Overall score: " in out and "(Excellent)" in out
    assert "Status: Evaluated" in out
    assert "First failing stage: none" in out
    assert "Regressions" not in out

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["retrieval"]["summary"]["recall_at_k"] == 1
    assert output.with_suffix(".md").read_text(encoding="utf-8").startswith("# RAG Pipeline")


def test_baseline_and_fail_on_regression(data_file, rag_dataset, tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    cli.main(["evaluate", "--input", str(data_file), "--output", str(baseline), "--k", "3"])

    # same data again: nothing got worse, exit code 0
    cli.main(["evaluate", "--input", str(data_file), "--output", str(tmp_path / "same.json"),
              "--k", "3", "--baseline", str(baseline), "--fail-on-regression"])
    assert "Regressions against the baseline: none" in capsys.readouterr().out

    rag_dataset["queries"][0]["retrieved_ids"] = ["c3"]        # the retriever now misses q1
    data_file.write_text(json.dumps(rag_dataset), encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        cli.main(["evaluate", "--input", str(data_file), "--output", str(tmp_path / "new.json"),
                  "--k", "3", "--baseline", str(baseline), "--fail-on-regression"])

    assert error.value.code == 1
    assert "retrieval.recall_at_k" in capsys.readouterr().out
    assert (tmp_path / "new.json").is_file()                   # the report is saved first


def test_evaluate_needs_input_or_files_and_questions():
    with pytest.raises(SystemExit, match="--input data.json, or --files and --questions"):
        cli.main(["evaluate", "--files", "a.md"])
    with pytest.raises(SystemExit, match="--fail-on-regression needs --baseline"):
        cli.main(["evaluate", "--input", "data.json", "--fail-on-regression"])


def test_evaluate_file_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="Data file not found"):
        evaluate_file(tmp_path / "missing.json")

    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must hold a JSON object"):
        evaluate_file(bad)

    bad.write_text('{"chunks": ["a"], "chunkz": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown input"):
        evaluate_file(bad, tmp_path / "report.json")


def test_summary_text_without_scores():
    report = {"scorecard": {"overall": {"score": None, "rating": "Not Evaluated",
                                        "status": "Not Evaluated", "evaluation_coverage": 0}},
              "root_causes": {"first_failing_stage": None}}

    assert summary_text(report).startswith("Overall score: N/A (Not Evaluated)")
