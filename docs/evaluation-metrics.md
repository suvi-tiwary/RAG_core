# Evaluation metrics

Every metric the [stage evaluators](stage-evaluation.md) compute, with its
formula and how to read it. The **key** is the name in the report and in the
[scorecard](stage-evaluation.md#the-scorecard).

Three rules hold for all of them:

- **No LLM judge.** Every number is computed from your data, so the same data
  always gives the same score.
- **Not measured is not zero.** A metric whose input you did not pass is
  `None`. It is left out of averages and scores.
- **A "token" is a word**: letters, digits and underscores, compared in
  lowercase. `"Hi, Bob!"` has the tokens `hi` and `bob`.

Query metrics are computed for each query, then averaged over the queries
that have the data.

## 1. Chunking

Evaluate right after the document is split, before indexing or retrieval.
Computed by `ChunkEvaluator`.

**Notation:** $C_i$ is chunk $i$ and $\lvert C_i \rvert$ its number of tokens,
$n$ the number of chunks, $N$ the number of source tokens, $U$ the set of
source-token positions covered by at least one chunk, and $G$ the annotated
gold spans.

| Metric | Key | Formula | How to use it |
|---|---|---|---|
| Document Coverage | `coverage` | $\dfrac{\lvert U \rvert}{N}$ | Does the chunker keep all source content? Target 100%. |
| Size Compliance | `size_compliance` | $\dfrac{\#\{C_i : L_{min} \le \lvert C_i \rvert \le L_{max}\}}{n}$ | Chunks inside your `size_limits`. Higher is better. |
| Mean Chunk Size | `mean_chunk_tokens` | $\mu = \dfrac{1}{n} \sum_i \lvert C_i \rvert$ | Compare average chunk length across chunkers. Also `min_chunk_tokens`, `max_chunk_tokens`. |
| Size Variation (CV) | `size_cv` | $CV = \dfrac{\sigma}{\mu}$ | How evenly sized the chunks are. Lower is not always better: sections differ in length. |
| Redundancy Ratio | `overlap_ratio` | $\dfrac{\sum_i \lvert C_i \rvert - \lvert U \rvert}{\lvert U \rvert}$ | Duplication from overlapping chunks. Lower means a smaller index. |
| Evidence Completeness | `evidence_completeness` | $\dfrac{\#\text{gold spans intact in one chunk}}{\lvert G \rvert}$ | Is a required fact split across chunks? Higher is better. |
| Evidence Fragmentation | `evidence_fragmentation` | $\dfrac{1}{\lvert G \rvert} \sum_{g \in G} \#\{C_i : C_i \cap g \ne \emptyset\}$ | Average number of chunks a gold span touches. 1 is the best. |
| Structural Integrity | `structural_integrity` | $\dfrac{\#\text{intact structural blocks}}{\#\text{blocks}}$ | Do tables, code, lists and headings stay whole? Higher is better. |
| Boundary Alignment F1 | `boundary_f1` | $\dfrac{2 P_b R_b}{P_b + R_b}$ | Compare split points with the ideal ones in `expected_chunks`. Higher is better. |
| Semantic Cohesion | `semantic_cohesion` | $\dfrac{2}{m(m-1)} \sum_{i<j} \cos(e_i, e_j)$ | Do the $m$ sentences of a chunk talk about one thing? Needs `sentence_vectors`. Higher is generally better. |
| Boundary Clarity | `semantic_boundary_clarity` | $1 - \cos(e_{left}, e_{right})$ | Is the topic different on both sides of a boundary? Needs `sentence_vectors`. Higher suggests a clearer split. |
| Chunk Stickiness | `semantic_chunk_stickiness` | $\dfrac{1}{n-1} \sum_i \cos(e_{C_i}, e_{C_{i+1}})$ | Neighbouring chunks that may have been split without need. Needs `chunk_vectors`. Read it together with the document structure. |
| Metadata Completeness | `metadata_completeness` | $\dfrac{\#\text{chunks with every required key}}{n}$ | Can every chunk be traced to its source? |
| Metadata Accuracy | `metadata_accuracy` | $\dfrac{\#\text{correct metadata values}}{\#\text{expected values}}$ | Are page numbers and sources right? |

For Boundary F1, $P_b$ is the share of predicted boundaries that match an
ideal one and $R_b$ the share of ideal boundaries that were found. The end of
a document is not a boundary. `boundary_tolerance` allows a match a few
characters away.

For Semantic Cohesion the mean is taken inside each chunk first, then over
the chunks. For Boundary Clarity, $e_{left}$ is the last sentence of a chunk
and $e_{right}$ the first sentence of the next one.

Without vectors, three **word-overlap proxies** are always available. They
are rougher, but need nothing more than the chunk texts:

| Key | Formula | Stands in for |
|---|---|---|
| `lexical_cohesion` | mean Jaccard of the sentence pairs inside a chunk | Semantic Cohesion |
| `lexical_boundary_clarity` | $1 - J(\text{last 12 words left}, \text{first 12 words right})$ | Boundary Clarity |
| `lexical_chunk_stickiness` | mean $J(C_i, C_{i+1})$ | Chunk Stickiness |

Here $J(A, B) = \dfrac{\lvert A \cap B \rvert}{\lvert A \cup B \rvert}$ is the Jaccard overlap of two word sets.

## 2. Embedding and indexing

Evaluate after the chunks are embedded and stored. Computed by
`EmbeddingEvaluator` from vectors you pass: no model is called.

**Notation:** $V$ is the set of chunk vectors, $q$ a query vector, $G_q$ the
gold chunks of query $q$, and $E_q^K$ the exact cosine top $K$ for $q$.

| Metric | Key | Formula | How to use it |
|---|---|---|---|
| Zero-Norm Rate | `zero_norm_rate` | $\dfrac{\#\{v \in V : \lVert v \rVert = 0\}}{\lvert V \rvert}$ | All-zero vectors cannot be searched. Target 0. |
| Duplicate Vector Rate | `duplicate_vector_rate` | $\dfrac{\lvert V \rvert - \#\text{distinct vectors}}{\lvert V \rvert}$ | Identical vectors usually mean duplicated content. Lower is better. |
| Mean Vector Norm | `mean_vector_norm` | $\dfrac{1}{\lvert V \rvert} \sum_v \lVert v \rVert$ | About 1 for normalised embeddings. A change signals a model or pipeline change. |
| Index Coverage | `index_coverage` | $\dfrac{\lvert V \cap \text{indexed ids} \rvert}{\lvert V \rvert}$ | Did every vector reach the index? Target 100%. |
| Query Top-K Recall | `query_topk_recall` | $\dfrac{\lvert E_q^K \cap G_q \rvert}{\lvert G_q \rvert}$ | Can the embedding model itself find the gold chunks? |
| ANN Recall@K | `ann_recall_at_k` | $\dfrac{\lvert \text{index}_q^K \cap E_q^K \rvert}{\lvert E_q^K \rvert}$ | What the approximate index loses against exact search. |
| Positive Similarity | `positive_similarity` | mean $\cos(q, c)$ for $c \in G_q$ | How close gold chunks are to their queries. Higher is better. |
| Negative Similarity | `negative_similarity` | mean $\cos(q, c)$ for $c \notin G_q$ | How close the other chunks are. Lower is better. |
| Similarity Margin | `similarity_margin` | positive $-$ negative | How easily relevant and irrelevant chunks are told apart. Higher is better. |

## 3. Retrieval

Evaluate after the vector database or retriever returns ranked chunks.
Computed by `RetrievalEvaluator`.

**Notation:** $R$ is the set of source-token positions in the retrieved top
$K$ chunks, $G$ the gold evidence token positions, $rel_i$ the gold grade of
the chunk at rank $i$ (0 = not relevant), and $Q$ the number of queries.

| Metric | Key | Formula | How to use it |
|---|---|---|---|
| Precision@K | `precision_at_k` | $\dfrac{\#\text{relevant results in top } K}{K}$ | How many of the first K results are useful. |
| Recall@K | `recall_at_k` | $\dfrac{\#\text{relevant results in top } K}{\#\text{relevant results}}$ | Does the retriever find the required evidence? |
| F1@K | `f1_at_k` | $\dfrac{2PR}{P+R}$ | Balance retrieval precision and recall. |
| MRR@K | `mrr_at_k` | $\dfrac{1}{Q} \sum_q \dfrac{1}{\text{rank}_q}$ | How early the first relevant result appears. A miss scores 0. |
| nDCG@K | `ndcg_at_k` | $\dfrac{\sum_{i=1}^{K} \dfrac{2^{rel_i} - 1}{\log_2(i+1)}}{IDCG@K}$ | Ranking quality when chunks have different relevance grades. |
| Hit Rate@K | `hit_at_k` | $\dfrac{\#\text{queries with a relevant top-}K\text{ hit}}{Q}$ | How often retrieval finds at least one useful result. |
| MAP@K | `average_precision_at_k` | $\dfrac{\sum_{i \in \text{relevant ranks}} P@i}{\min(\#\text{relevant}, K)}$ | Are the relevant chunks ranked first, not spread out? |
| Token Recall | `token_recall` | $\dfrac{\lvert R \cap G \rvert}{\lvert G \rvert}$ | What share of the required evidence tokens was retrieved. |
| Token Precision | `token_precision` | $\dfrac{\lvert R \cap G \rvert}{\lvert R \rvert}$ | What share of the retrieved text is evidence. |
| Token IoU | `token_iou` | $\dfrac{\lvert R \cap G \rvert}{\lvert R \cup G \rvert}$ | Penalises missing evidence and excess context together. |
| Complete Evidence Recall | `complete_evidence_recall` | $\dfrac{\#\text{fully recovered gold spans}}{\#\text{gold spans}}$ | Is every evidence span retrievable in one piece? |
| Retrieval P95 Latency | `p95_latency_ms` | $P_{95}(t_1, \ldots, t_Q)$ | Slow-query performance, not just the mean. Also `p50_latency_ms`. |

$IDCG@K$ is the DCG of the best possible order of the gold chunks. Without
graded labels every gold chunk has grade 1.

The chunk metrics need gold chunk ids; the token metrics need
`expected_evidence` and the source text, and **do not depend on chunk ids**,
so they stay comparable when you change the chunker. A chunk id returned
twice counts once, at its first rank.

## 4. Reranking

Evaluate after the reranker reorders the retrieved chunks. Computed by
`RerankingEvaluator` from `retrieved_ids` (before) and `reranked_ids` (after).

| Metric | Key | Formula | How to use it |
|---|---|---|---|
| nDCG@K | `ndcg_at_k` | nDCG@K of the reranked list | Final ordering quality. |
| MRR@K | `mrr_at_k` | MRR@K of the reranked list | Is a relevant chunk at the very top? |
| nDCG Gain | `ndcg_gain` | $nDCG_{after} - nDCG_{before}$ | Above 0 the reranker helps, below 0 it hurts. |
| MRR Gain | `mrr_gain` | $MRR_{after} - MRR_{before}$ | Did the first relevant chunk move up? |
| Precision Gain | `precision_gain` | $P@K_{after} - P@K_{before}$ | Did reranking bring more relevant chunks into the top K? |
| First Relevant Rank Change | `first_relevant_rank_change` | $\text{rank}_{before} - \text{rank}_{after}$ | Places the first relevant chunk moved up, over the whole lists. Not scored. |
| Candidate Set Preservation | `candidate_set_preservation_rate` | $\dfrac{\#\text{queries with the same chunk set}}{Q}$ | 1 when the reranker only reorders. A reranker that cuts the list gives 0, which can be intended. |

## 5. Augmentation (context)

Evaluate the chunks that are packed into the prompt. Computed by
`AugmentationEvaluator` from `context_ids`.

**Notation:** $C$ is the set of source-token positions in the context and
$G$ the gold evidence token positions.

| Metric | Key | Formula | How to use it |
|---|---|---|---|
| Context Precision | `context_precision` | $\dfrac{\lvert C \cap G \rvert}{\lvert C \rvert}$ | How much of the prompt context is evidence. |
| Context Recall | `context_recall` | $\dfrac{\lvert C \cap G \rvert}{\lvert G \rvert}$ | How much of the evidence reached the LLM. |
| Context Noise | `context_noise` | $1 - \text{Context Precision}$ | Irrelevant text the LLM must ignore. Lower is better. |
| Context Sufficiency | `context_sufficient` | $1[\text{every gold span is whole in a context chunk}]$ | Can the question be answered from the context at all? |
| Duplicate Ratio | `duplicate_ratio` | $\dfrac{\text{context tokens} - \lvert C \rvert}{\text{context tokens}}$ | Repeated text wastes the token budget. Lower is better. |
| Budget Compliance | `budget_compliant` | $1[\text{context tokens} \le \text{max\_context\_tokens}]$ | Does the context fit the budget? |
| Evidence Edge Placement | `evidence_edge_placement` | mean $\lvert 2p - 1 \rvert$, $p = \dfrac{\text{position}}{\text{chunks} - 1}$ | 1 when evidence is first or last in the context, 0 in the middle, where LLMs often overlook it. |

## 6. Generation

Compare the generated answer with the validated ground truth. Computed by
`GenerationEvaluator`.

**Notation:** $A$ is the generated answer, $Y$ the reference answer, and
$t(A)$ and $t(Y)$ their normalised token sequences.

| Metric | Key | Formula | How to use it |
|---|---|---|---|
| Exact Match (EM) | `exact_match` | $1[norm(A) = norm(Y)]$ | Best for factual answers, dates, IDs and numbers. |
| Answer Token Precision | `answer_precision` | $\dfrac{\#\text{matched token occurrences}}{\lvert t(A) \rvert}$ | How much of the answer matches the reference vocabulary. |
| Answer Token Recall | `answer_recall` | $\dfrac{\#\text{matched token occurrences}}{\lvert t(Y) \rvert}$ | How much of the reference answer is represented. |
| Answer F1 | `answer_f1` | $\dfrac{2 P_a R_a}{P_a + R_a}$ | Balance answer token precision and recall. |
| ROUGE-L F1 | `rouge_l_f1` | $\dfrac{2 P_{LCS} R_{LCS}}{P_{LCS} + R_{LCS}}$ | Longer answers: overlap by longest common subsequence, so word order counts. |
| Required Fact Coverage | `required_fact_coverage` | $\dfrac{\#\text{required facts in the answer}}{\#\text{required facts}}$ | Are the mandatory facts present? Matched as phrases, ignoring case. |
| Citation Precision | `citation_precision` | $\dfrac{\#\text{correct cited ids}}{\#\text{cited ids}}$ | Do the citations point to the right sources? |
| Citation Recall | `citation_recall` | $\dfrac{\#\text{correct cited ids}}{\#\text{expected ids}}$ | Is every expected source cited? |
| Citation F1 | `citation_f1` | $\dfrac{2PR}{P+R}$ | Balance citation precision and recall. |
| Claim Support | `annotated_claim_support` | $\dfrac{\#\text{claims marked supported}}{\#\text{annotated claims}}$ | Faithfulness, from **your** human labels in `verified_claims`. |
| Abstention Accuracy | `abstention_accuracy` | $\dfrac{\#\text{correct answer/abstain decisions}}{Q}$ | Does the system refuse unanswerable questions, and only those? |
| Context Grounding | `context_grounding` | $\dfrac{\#\{w \in t(A) : w \in \text{context}\}}{\lvert t(A) \rvert}$ | Share of the answer's words found in the context. A low value hints at invented content. |
| Unsupported Number Rate | `unsupported_number_rate` | $\dfrac{\#\text{numbers in } A \text{ not in the context}}{\#\text{numbers in } A}$ | Invented numbers are an easy to detect hallucination. Lower is better. |
| Task Success | `task_success` | $\dfrac{\#\text{successful tasks}}{\#\text{labelled tasks}}$ | Your own verdict per query, in `task_success`. |
| Generation P95 Latency | `p95_latency_ms` | $P_{95}(t_{generation})$ | LLM time alone, from `generation_latency_ms`. |

$P_{LCS} = LCS(A, Y) / \lvert t(A) \rvert$ and $R_{LCS} = LCS(A, Y) / \lvert t(Y) \rvert$,
where $LCS$ is the length of the longest common subsequence of the two token lists.

Context Grounding and Unsupported Number Rate are **hints, not proof**: an
answer can reuse the context's words and still be wrong. For a hard check of
faithfulness, label the claims yourself (`verified_claims`).

**Refusals are scored by Abstention Accuracy only.** A question the documents
cannot answer (`"answerable": false`) has no reference answer, so Exact Match,
Answer F1 and ROUGE-L are *not measured* for it. A refusal (`"abstained": true`)
makes no claims, so Context Grounding and Unsupported Number Rate are *not
measured* for it. An answer that should have been a refusal still fails: its
Abstention Accuracy is 0 and its grounding is measured.

Exact Match and token F1 are the standard question-answering metrics of the
SQuAD evaluation.

## 7. End to end

What the user experienced, from request to answer. Computed by
`EndToEndEvaluator`.

| Metric | Key | Formula | How to use it |
|---|---|---|---|
| Exact Match | `answer_exact_match` | $1[norm(A) = norm(Y)]$ | Final-answer correctness for short factual answers. |
| Answer F1 | `answer_f1` | $\dfrac{2 P_a R_a}{P_a + R_a}$ | Final-answer overlap with the reference. |
| Task Success Rate | `task_success_rate` | $\dfrac{\#\text{successful tasks}}{\#\text{labelled tasks}}$ | Did the user get what they needed? |
| End-to-End P95 Latency | `p95_latency_ms` | $P_{95}(t_{\text{request to answer}})$ | Real user-facing response time. Also `p50_latency_ms`. |
| Mean Cost | `mean_cost_usd` | $\dfrac{1}{Q} \sum_q \text{cost}_q$ | Cost per request. |
| Mean Tokens | `mean_total_tokens` | $\dfrac{1}{Q} \sum_q \text{tokens}_q$ | LLM token usage per request. |

Percentiles use linear interpolation: with latencies 100 ms and 300 ms,
$P_{50} = 200$ and $P_{95} = 290$.
