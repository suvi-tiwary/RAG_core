"""Tests for the scorecard: normalisation, stage scores, overall score, gates and status."""

import pytest

from rag_titan.evaluation import RAGEvaluator, ScoreAggregator
from rag_titan.evaluation.scoring import normalize, rating
from rag_titan.evaluation.scoring.explanations import weakest_metrics
from rag_titan.evaluation.scoring.policy import METRICS, QUALITY_GATES, STAGE_WEIGHTS


@pytest.mark.parametrize("value, direction, expected", [
    (0.8, "high", 0.8),              # recall
    (0.1, "low", 0.9),               # noise
    (0.25, "ratio", 0.8),            # redundancy
    (2, "fragmentation", 0.5),       # two chunks per evidence span
    (0.6, "cos_high", 0.8),
    (0, "cos_low", 0.5),
    (1, "cos_distance", 0.5),
    (0.25, "margin", 0.75),
    (0.9, "margin", 1),              # saturates
    (0, "gain", 0.5),                # no change is neutral
])
def test_normalize_table(value, direction, expected):
    assert normalize(value, direction) == pytest.approx(expected)


def test_normalize_targets():
    assert normalize(2000, "target", 1000) == 0.5
    assert normalize(800, "target", 1000) == 1
    assert normalize(300, "target", {"min": 150, "max": 400}) == 1
    assert normalize(75, "target", {"min": 150, "max": 400}) == 0.5
    assert normalize(800, "target", {"min": 150, "max": 400}) == 0.5


def test_normalize_errors():
    with pytest.raises(ValueError, match="outside the range"):
        normalize(1.2, "high")
    with pytest.raises(ValueError, match="not a finite number"):
        normalize(None, "high")
    with pytest.raises(ValueError, match="Unknown direction"):
        normalize(0.5, "sideways")
    with pytest.raises(ValueError, match="target must be"):
        normalize(5, "target", 0)
    with pytest.raises(ValueError, match="min <= max"):
        normalize(5, "target", {"min": 9, "max": 3})


def test_rating_bands():
    assert rating(95)[0] == "Excellent"
    assert rating(90)[0] == "Excellent"
    assert rating(89.9)[0] == "Very Good"
    assert rating(70)[0] == "Good"
    assert rating(65)[0] == "Average"
    assert rating(40)[0] == "Needs Improvement"
    assert rating(10)[0] == "Poor"
    assert rating(None)[0] == "Not Evaluated"


def test_policy_is_consistent():
    assert set(METRICS) == set(STAGE_WEIGHTS)
    assert sum(STAGE_WEIGHTS.values()) == 100
    for stage, gates in QUALITY_GATES.items():
        assert set(gates) <= set(METRICS[stage])


def test_stage_score_hand_worked():
    report = {"retrieval": {"summary": {"recall_at_k": 0.8, "precision_at_k": 0.5,
                                         "ndcg_at_k": None}}}
    card = ScoreAggregator(stages=["retrieval"]).evaluate(report)
    stage = card["stages"]["retrieval"]

    # recall has weight 4, precision weight 2: (4 * 0.8 + 2 * 0.5) / 6 = 0.7
    assert stage["score"] == 70
    assert stage["rating"] == "Good"
    assert stage["metrics_scored"] == 2
    assert stage["evaluation_coverage"] == pytest.approx(6 / 24)     # all retrieval weights: 24
    assert "ndcg_at_k" in stage["missing_metrics"]                   # missing, not scored as 0
    assert stage["improvement_areas"][0]["metric"] == "precision_at_k"

    overall = card["overall"]
    assert overall["score"] == 70
    assert overall["status"] == "Provisional"                        # only 25% was measured
    assert overall["is_provisional"] is True
    assert overall["quality_gate_failures"] == []


def test_quality_gate_gives_action_required():
    report = {"retrieval": {"summary": {"recall_at_k": 0.5, "hit_at_k": 1.0}}}
    overall = ScoreAggregator(stages=["retrieval"], min_coverage=0).evaluate(report)["overall"]

    assert overall["status"] == "Action Required"
    assert overall["quality_gate_failures"][0]["metric"] == "recall_at_k"
    assert overall["quality_gate_failures"][0]["minimum"] == 0.7

    no_gates = ScoreAggregator(stages=["retrieval"], min_coverage=0, quality_gates={})
    assert no_gates.evaluate(report)["overall"]["status"] == "Evaluated"


def test_overall_weights_stages_by_weight_and_coverage():
    report = {"retrieval": {"summary": {"recall_at_k": 1.0}},        # weight 20, coverage 4/24
              "end_to_end": {"task_success_rate": 0.5}}              # weight 15, coverage 5/10
    card = ScoreAggregator(stages=["retrieval", "end_to_end"], quality_gates={}).evaluate(report)

    retrieval, end_to_end = 20 * 4 / 24, 15 * 5 / 10
    expected = (retrieval * 1.0 + end_to_end * 0.5) / (retrieval + end_to_end)
    assert card["overall"]["score"] == pytest.approx(expected * 100, abs=0.01)
    assert card["overall"]["evaluation_coverage"] == pytest.approx(
        (retrieval + end_to_end) / 35, abs=0.0001)
    assert card["overall"]["areas_to_review"][0]["stage"] == "end_to_end"


def test_nothing_measured():
    card = ScoreAggregator().evaluate({})

    assert card["overall"]["score"] is None
    assert card["overall"]["status"] == "Not Evaluated"
    assert card["overall"]["evaluated_stages"] == 0
    assert card["stages"]["chunking"]["rating"] == "Not Evaluated"


def test_targets_and_custom_weights():
    report = {"end_to_end": {"task_success_rate": 1.0, "p95_latency_ms": 2000}}
    plain = ScoreAggregator(stages=["end_to_end"]).evaluate(report)
    assert "p95_latency_ms" not in plain["stages"]["end_to_end"]["metric_breakdown"]

    scorer = ScoreAggregator(stages=["end_to_end"],
                             targets={"end_to_end": {"p95_latency_ms": 1000}},
                             metric_weights={"end_to_end": {"task_success_rate": 2,
                                                            "answer_f1": 0}})
    stage = scorer.evaluate(report)["stages"]["end_to_end"]

    # success: weight 2, score 1; latency: weight 2, score 0.5 -> 75
    assert stage["score"] == 75
    assert "answer_f1" not in stage["missing_metrics"]              # weight 0 switches it off


def test_value_outside_its_range_is_a_warning_not_a_score():
    stage = ScoreAggregator(stages=["retrieval"]).evaluate(
        {"retrieval": {"summary": {"recall_at_k": 1.5}}})["stages"]["retrieval"]

    assert stage["score"] is None
    assert stage["warnings"] == ["recall_at_k: value 1.5 is outside the range of a 'high' metric"]


def test_weakest_metrics_skips_strong_and_repeated_reasons():
    details = {"context_precision": {"normalized_score": 0.4, "weight": 1.5},
               "context_noise": {"normalized_score": 0.4, "weight": 0.5},     # same reason
               "context_recall": {"normalized_score": 0.9, "weight": 3}}      # strong
    found = weakest_metrics(details)

    assert [item["metric"] for item in found] == ["context_precision"]
    assert found[0]["metric_score"] == 40


def test_scorecard_in_the_full_report(rag_dataset):
    card = RAGEvaluator(k=3).evaluate(**rag_dataset)["scorecard"]

    assert card["overall"]["status"] == "Evaluated"
    assert card["overall"]["rating"] == "Excellent"
    assert card["overall"]["evaluated_stages"] == 7
    assert all(not stage["warnings"] for stage in card["stages"].values())

    only_search = RAGEvaluator(k=3, scorer=ScoreAggregator(stages=["retrieval", "reranking"]))
    assert list(only_search.evaluate(**rag_dataset)["scorecard"]["stages"]) == [
        "retrieval", "reranking"]


def test_constructor_errors():
    with pytest.raises(ValueError, match="stages must be"):
        ScoreAggregator(stages=["retrieval", "search"])
    with pytest.raises(ValueError, match="stages must be"):
        ScoreAggregator(stages=[])
    with pytest.raises(ValueError, match="min_coverage"):
        ScoreAggregator(min_coverage=2)
    with pytest.raises(ValueError, match="Stage weight"):
        ScoreAggregator(stage_weights={"retrieval": 0})
    with pytest.raises(ValueError, match="Stage weight"):
        ScoreAggregator(stages=["retrieval"], stage_weights={"generation": 5})
    with pytest.raises(ValueError, match="Invalid metric weight for retrieval.recal"):
        ScoreAggregator(metric_weights={"retrieval": {"recal": 1}})
