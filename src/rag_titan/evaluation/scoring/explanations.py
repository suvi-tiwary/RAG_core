"""Plain-language reasons for a weak metric, shown in the scorecard and the reports."""

DEFAULT_REASON = "This metric is below the strongest-performing checks."

REASONS = {
    # chunking
    "coverage": "Some document content may have been omitted during chunking.",
    "size_compliance": "Several chunks fall outside the intended size range.",
    "size_cv": "Chunk sizes vary a lot; check whether this is expected for the document.",
    "overlap_ratio": "Duplicated chunk content increases indexing overhead.",
    "evidence_completeness": "Important facts were split across chunk boundaries.",
    "evidence_fragmentation": "Evidence spans often cross several chunk boundaries.",
    "structural_integrity": "Tables, lists, or other structured content were split.",
    "metadata_completeness": "Some chunks are missing metadata needed for traceability.",
    "metadata_accuracy": "Some chunk metadata disagrees with the expected source labels.",
    "boundary_f1": "Chunk boundaries differ from the expected split locations.",
    "lexical_cohesion": "Some chunks mix distinct vocabulary or concepts.",
    "lexical_boundary_clarity": "Some chunk boundaries do not clearly separate topics.",
    "lexical_chunk_stickiness": "Adjacent chunks share similar language and may be over-split.",
    "semantic_cohesion": "Some chunks contain sentences that are weakly related.",
    "semantic_boundary_clarity": "Some chunk boundaries cut through one topic.",
    "semantic_chunk_stickiness": "Adjacent chunks are very similar and may be over-split.",
    "mean_chunk_tokens": "Average chunk size is outside the configured target range.",
    "min_chunk_tokens": "The smallest chunk falls outside the target range.",
    "max_chunk_tokens": "The largest chunk falls outside the target range.",
    # embedding
    "zero_norm_rate": "Some embeddings are all zeros and cannot be searched.",
    "duplicate_vector_rate": "Some chunks have identical vectors: look for duplicated content.",
    "index_coverage": "Some vectors did not reach the search index.",
    "query_topk_recall": "The vectors do not reliably rank expected evidence near the top.",
    "ann_recall_at_k": "Approximate search misses some exact nearest neighbours.",
    "positive_similarity": "The expected relevant content is not very close to the queries.",
    "negative_similarity": "Irrelevant content is too similar to the queries.",
    "similarity_margin": "Relevant and irrelevant chunks are not clearly separated.",
    "mean_vector_norm": "Embedding magnitude falls outside the configured target range.",
    # retrieval and reranking
    "precision_at_k": "The search returns extra irrelevant chunks.",
    "recall_at_k": "The search misses expected relevant chunks.",
    "f1_at_k": "The search trades too much precision for recall, or the reverse.",
    "hit_at_k": "For some questions the search finds no relevant chunk at all.",
    "mrr_at_k": "Relevant results appear too far down the list.",
    "ndcg_at_k": "The most relevant chunks are not ranked high enough.",
    "average_precision_at_k": "Relevant chunks are spread out instead of ranked first.",
    "token_recall": "Some required evidence is absent from the retrieved chunks.",
    "token_precision": "A large share of the retrieved text is unrelated to the answer.",
    "token_iou": "Retrieved chunks contain unnecessary text or miss evidence.",
    "complete_evidence_recall": "Some required evidence spans cannot be recovered in full.",
    "ndcg_gain": "Reranking showed little improvement; the first ranking may already be strong.",
    "mrr_gain": "Reranking did not move the first relevant result up much.",
    "precision_gain": "Reranking did not bring more relevant chunks into the top results.",
    "candidate_set_preservation_rate": "The reranker dropped or added candidates.",
    # augmentation
    "context_precision": "The final prompt contains substantial irrelevant context.",
    "context_recall": "Some required evidence did not reach the LLM.",
    "context_noise": "The final prompt contains substantial irrelevant context.",
    "context_sufficient": "The final prompt lacks information required for a complete answer.",
    "duplicate_ratio": "Repeated context wastes the token budget.",
    "budget_compliant": "The context exceeded the token budget.",
    "evidence_edge_placement": "Evidence sits in the middle of the context, where LLMs miss it.",
    # generation and end to end
    "exact_match": "The generated answer does not exactly match the reference wording.",
    "answer_exact_match": "The generated answer does not exactly match the reference wording.",
    "answer_precision": "Answers contain wording that is not in the reference.",
    "answer_recall": "Answers leave out parts of the reference answer.",
    "answer_f1": "The response differs from the reference answer's wording or details.",
    "rouge_l_f1": "Answers do not follow the reference answer's phrasing.",
    "required_fact_coverage": "The response is missing some required facts.",
    "citation_precision": "Some cited sources are not in the expected citation set.",
    "citation_recall": "Some expected evidence sources were not cited.",
    "citation_f1": "Citations differ from the expected sources.",
    "annotated_claim_support": "Some claims failed the human-verified evidence checks.",
    "abstention_accuracy": "The system sometimes answers when it should abstain, or vice versa.",
    "context_grounding": "Answers use wording that is not in the supplied context.",
    "unsupported_number_rate": "Answers contain numbers that are not in the supplied context.",
    "task_success": "Some end-user tasks were marked unsuccessful.",
    "task_success_rate": "Some end-user tasks were marked unsuccessful.",
    "p50_latency_ms": "Typical response time exceeds its budget.",
    "p95_latency_ms": "The slowest requests exceed the response-time budget.",
    "mean_cost_usd": "Average request cost is above the configured budget.",
    "mean_total_tokens": "Average token usage is above the configured budget.",
}


def weakest_metrics(details, limit=3):
    """
    Return the measured metrics that cost a stage the most points, worst first.

    Args:
        details: {metric: {"normalized_score", "weight", ...}} of one stage.
        limit: how many to return at most.

    Metrics scoring 85% or more, or costing under 0.5% of the stage, are left out.
    """
    total_weight = sum(info["weight"] for info in details.values())

    def points_lost(name):
        return (1 - details[name]["normalized_score"]) * details[name]["weight"]

    found, seen = [], set()
    for name in sorted(details, key=lambda name: (-points_lost(name), name)):
        info = details[name]
        reason = REASONS.get(name, DEFAULT_REASON)
        small = info["normalized_score"] >= 0.85 or points_lost(name) < 0.005 * total_weight
        if small or reason in seen:
            continue
        seen.add(reason)
        found.append({"metric": name, "metric_score": round(info["normalized_score"] * 100, 1),
                      "reason": reason})
    return found[:limit]
