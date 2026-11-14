"""Tests for the golden template: make_golden_template() and `rag-titan golden`."""

import json

import pytest

from rag_titan import cli
from rag_titan.evaluation import check_golden, make_golden_template
from rag_titan.evaluation.golden_template import suggest_evidence

POLICY = ("# Travel policy\n\n"
          "Economy class is required for all flights under six hours. "
          "Business class is allowed for flights of six hours or longer.\n\n"
          "The daily meal allowance is 45 euros. Alcohol is never reimbursed.\n")
DOCS = {"policy.md": POLICY}


def test_one_record_per_question_with_empty_labels():
    records = make_golden_template(["How much is the meal allowance?", "  Who pays?  "])

    assert [r["id"] for r in records] == ["q1", "q2"]
    assert records[1]["question"] == "Who pays?"
    first = records[0]
    assert first["expected_answer"] == "" and first["expected_evidence"] == []
    assert first["required_facts"] == [] and first["answerable"] is True
    assert first["validated"] is False and first["_todo"]


def test_suggests_the_sentence_sharing_most_words():
    records = make_golden_template(["How much is the daily meal allowance?"], DOCS)
    assert records[0]["expected_evidence"] == ["The daily meal allowance is 45 euros."]


def test_suggestion_is_only_a_suggestion():
    # "business class ... flight" points at the wrong sentence: a person must review it
    quote = suggest_evidence("Can I fly business class on a four-hour flight?", DOCS)
    assert quote == "Business class is allowed for flights of six hours or longer."


def test_no_suggestion_when_no_word_is_shared():
    assert suggest_evidence("What is the gym budget?", DOCS) is None


def test_repeated_sentences_are_never_suggested():
    docs = {"a.md": "Meals cost 45 euros.", "b.md": "Meals cost 45 euros."}
    assert suggest_evidence("What do meals cost?", docs) is None


def test_an_unreviewed_template_is_never_ready():
    records = make_golden_template(["How much is the daily meal allowance?"], DOCS)
    result = check_golden(records, DOCS)

    assert result["ready"] is False
    assert result["validated_count"] == 0
    assert {"id": "q1", "problem": "no expected_answer"} in result["problems"]


def test_reviewed_template_becomes_ready():
    record = make_golden_template(["How much is the daily meal allowance?"], DOCS)[0]
    record.update(expected_answer="45 euros a day.", required_facts=["45"], validated=True)
    assert check_golden([record], DOCS)["ready"] is True


@pytest.mark.parametrize("questions", [[], "a question", ["ok", "  "], [None]])
def test_bad_questions_raise_clear_errors(questions):
    with pytest.raises(ValueError):
        make_golden_template(questions)


def test_documents_must_be_a_dict():
    with pytest.raises(ValueError, match="dict"):
        make_golden_template(["Q?"], ["not a dict"])


def test_cli_writes_the_template(tmp_path, capsys):
    (tmp_path / "questions.txt").write_text("How much is the daily meal allowance?\n\n"
                                            "What is the gym budget?\n", encoding="utf-8")
    (tmp_path / "policy.md").write_text(POLICY, encoding="utf-8")
    output = tmp_path / "golden.json"

    cli.main(["golden", "--questions", str(tmp_path / "questions.txt"),
              "--files", str(tmp_path / "policy.md"), "--output", str(output)])

    records = json.loads(output.read_text(encoding="utf-8"))
    assert [r["id"] for r in records] == ["q1", "q2"]
    assert records[1]["expected_evidence"] == []
    assert "2 questions, 1 with a suggested evidence quote" in capsys.readouterr().out


def test_cli_reads_a_json_list(tmp_path):
    (tmp_path / "questions.json").write_text('["Who pays?"]', encoding="utf-8")
    output = tmp_path / "golden.json"
    cli.main(["golden", "--questions", str(tmp_path / "questions.json"), "--output", str(output)])
    assert json.loads(output.read_text(encoding="utf-8"))[0]["question"] == "Who pays?"


def test_cli_never_overwrites_without_force(tmp_path):
    (tmp_path / "questions.txt").write_text("Who pays?\n", encoding="utf-8")
    output = tmp_path / "golden.json"
    output.write_text("my reviewed work", encoding="utf-8")
    args = ["golden", "--questions", str(tmp_path / "questions.txt"), "--output", str(output)]

    with pytest.raises(SystemExit, match="already exists"):
        cli.main(args)
    assert output.read_text(encoding="utf-8") == "my reviewed work"

    cli.main(args + ["--force"])
    assert json.loads(output.read_text(encoding="utf-8"))[0]["id"] == "q1"


def test_cli_missing_questions_file(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        cli.main(["golden", "--questions", str(tmp_path / "nope.txt"),
                  "--output", str(tmp_path / "g.json")])
