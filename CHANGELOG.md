# Changelog

All notable changes to this project are written here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.8.5] - 2026-11-13

### Added
- Docs: [Evaluate your RAG: the complete guide](https://aion7891.github.io/rag-titan/cookbook/complete-guide/),
  one page from the smallest input file to a report on all seven stages: prepare
  documents, golden questions, capture your RAG's logs, `build_dataset()`, chunking
  and embedding inputs, read the report, improve and compare with a baseline. With a
  step diagram, a cheat sheet of what to pass, a complete data file that uses every
  input (`full-data.json`, 100% measurement coverage) with every key explained,
  and a runnable example
  (`your-rag/` in the starter ZIP) that needs no model. Every command, output and
  score on the page is checked against a real run by the tests.

### Fixed
- A correct refusal is no longer scored as a wrong answer. For a question the
  documents cannot answer (`"answerable": false`) Exact Match, Answer F1 and ROUGE-L
  are now *not measured*, in the generation and end-to-end stages: Abstention
  Accuracy scores the reply. Before, "I don't know" got Answer F1 0 against the empty
  reference and was listed under "Queries to look at".
- A refusal (`"abstained": true`) is no longer checked against the context: Context
  Grounding and Unsupported Number Rate are *not measured* for it. Before, a RAG that
  learned to refuse correctly lost Context Grounding, and `--fail-on-regression`
  reported a regression for an improvement.

### Changed
- Scores can move for data with refusals or unanswerable questions. In the cookbook's
  `missed-evidence.json` (one question is refused) the overall score goes from 62.5
  to 63.2. Reports saved with an older version may show a difference in
  `context_grounding` or `answer_f1` when used as a `--baseline`: save a new baseline.

## [0.8.4] - 2026-11-12

### Fixed
- Windows: `RAGTitan(...)` no longer crashes with "DLL load failed while importing
  _pydantic_core: An Application Control policy has blocked this file". rag-titan
  now talks to Ollama over plain HTTP with the Python standard library, so it no
  longer needs pydantic and its compiled `_pydantic_core` file, which Smart App
  Control and company application-control policies block.
- Clearer errors when Ollama is not running ("Could not reach Ollama at ... Is
  Ollama running?"), when Ollama returns an error (its own message, like "model not
  found", is shown), and when it sends a different number of embeddings than texts.

### Changed
- `OllamaLLM` and `OllamaEmbedder` send exactly the same requests as before
  (same JSON, same tool descriptions, same host rules, `OLLAMA_HOST` and
  `OLLAMA_API_KEY` work as before). Their `.client` attribute is gone; use `.host`
  to see the server address.
- `OllamaLLM.chat()["message"]` is now a plain dict instead of an `ollama.Message`
  object. `message["content"]` and the other keys work as before; attribute access
  like `message.content` does not.

### Removed
- The `ollama` dependency, and with it `pydantic`, `pydantic-core`, `httpx`,
  `httpcore`, `anyio`, `h11` and `typing-inspection`.

## [0.8.3] - 2026-11-12

### Fixed
- Windows: `python -m rag_titan ...` now runs the command line tool, with the same
  commands and options as `rag-titan`. Use it when Smart App Control or an
  application-control policy blocks the unsigned `rag-titan.exe` launcher that pip
  creates. Before, `python -m rag_titan` failed with "is a package and cannot be
  directly executed".
- Docs: Getting started, the cookbook's Windows install steps and the Command line
  page use `python -m rag_titan` instead of `rag-titan.exe`.

## [0.8.2] - 2026-11-12

Helpers for users who already have a RAG, from a new-user test. Nothing existing
was removed or renamed.

### Added
- `rag-titan golden --questions questions.txt --files docs.md` and
  `make_golden_template()`: a first draft of your golden questions, with a
  suggested evidence sentence per question. Every record starts unvalidated, so
  `check_golden()` is not ready until a person has reviewed it. Never overwrites
  an existing file without `--force`.
- `build_dataset(chunks, runs, golden, documents)`: turns the logs of any RAG and
  your golden questions into an evaluation data file. It places chunks in their
  documents, takes labels only from the golden questions, finds gold chunk ids
  from evidence quotes and turns `[p1]` markers into citations. On the new-user
  test data, measurement coverage went from 32% to 59%.
- Docs: "Prepare golden data step by step", "Why is a value null?", "A regression
  test in plain words" with a diagram, and `build_dataset()` in "Evaluate your own RAG".

### Changed
- The `rag-titan evaluate --input` summary also prints "Queries to look at" (each
  failed query and its stage) and "Not measured" (what input would measure each
  missing part).
- Regression checks ignore latency noise: a latency counts as worse only when it is
  more than 10% and more than 5 ms slower. `find_regressions()` and
  `ReportManager.regressions()` take `latency_share=0, latency_min_ms=0` for the
  old strict rule.

## [0.8.1] - 2026-11-12

Docs-only release: no change to the Python API.

### Added
- Beginner evaluation cookbook: six runnable recipes, downloadable teaching data and
  live Ollama scripts, interactive stage walkthroughs using real evaluator results,
  and Bash/PowerShell command builders for scoring and regression checks.
- Animated terminal demos of the evaluation recipes, generated from real
  `rag-titan evaluate` output, including an end-to-end demo from golden questions
  to a scorecard.
- Animated evaluation flow image: your logs in, the score of every stage and the
  first failing stage out, drawn from a real evaluator run.
- "Choose your path" on the cookbook start page, with six steps for users who
  already have a RAG and only want to evaluate it; buttons on the Home page.

### Changed
- Docs sidebar regrouped into collapsible sections: Get started, Evaluate your RAG,
  Evaluation reference, Build a RAG, Playgrounds and Advanced.

## [0.8.0] - 2026-11-12

### Added
- Stage-by-stage evaluation for any RAG system from logged outputs and gold labels,
  without an LLM judge or new dependencies: `ChunkEvaluator`, `EmbeddingEvaluator`,
  `RetrievalEvaluator`, `RerankingEvaluator`, `AugmentationEvaluator`,
  `GenerationEvaluator` and `EndToEndEvaluator` from `rag_titan.evaluation`.
  Metrics cover chunk quality, vectors, ranking, context, answers and citations,
  latency, cost and tokens, with multi-document support.
- `RAGEvaluator`, `ScoreAggregator` and `RootCauseAnalyzer`: a weighted 0–100
  scorecard with quality gates, measurement coverage, provisional status and
  root-cause hints. Missing measurements stay unmeasured rather than scoring zero.
- `ReportManager` and `evaluate_file()`: JSON and Markdown reports, comparisons
  against a baseline and trends, with regression direction taken from the score policy.
- `rag-titan evaluate --input data.json` with `--output`, `--baseline`, `--k` and
  `--fail-on-regression` (non-zero exit on regression). Existing
  `--files` / `--questions` evaluation remains supported.
- `collect_dataset()` and `RAGTitan.evaluate_stages()` to collect and evaluate
  rag-titan runs, plus `load_golden()` for gold labels and expected evidence.
- Dataset helpers: `profile_documents()` for empty, short and duplicate documents,
  `sample_documents()` for random, stratified or risk-based samples, and
  `check_golden()` for golden-set checks and human-validation flags.
- Stage evaluation guide, metric formula reference, architecture diagram and
  runnable examples with sample data and golden questions.

## [0.7.0] - 2026-11-11

### Added
- **RAG playground** on the docs site: the real package runs in your browser
  (Pyodide 314.0.7 in a Web Worker). Pick chunkers and RAG types, ask a question,
  and every setup runs side by side on its own live flow diagram: steps light up
  in order with step numbers, loops are drawn, and clicking a step shows its input
  and output (chunks with files and scores, rewritten queries, evidence checks).
  - Recorded mode: real `gemma3:4b` replies and `nomic-embed-text` vectors,
    replayed instantly; "Evaluate all questions" gives the same report as
    `rag-titan evaluate`.
  - Live mode: your own Ollama (`OLLAMA_ORIGINS`), any question, the sample
    documents or your own pasted text.
- `rag-flow.js`: draws any RAG type from its `FLOW` (automatic layout, loops),
  so new RAG types get a diagram without JavaScript.
- `HashEmbedder` (`EMBED_PROVIDER=hash`): word-hash vectors, no model needed.
- `FunctionLLM` and `FunctionEmbedder`: any function becomes a model.
- `RecordingLLM` / `ReplayLLM` and `RecordingEmbedder` / `ReplayEmbedder`:
  record real model replies and vectors once, replay them anywhere.
- `rag_titan.playground`: `Playground`, `LivePlayground`, `record_presets()`,
  `check_presets()`; `tools/playground/record_traces.py` records the examples
  (`--check` replays them, and a test fails when they are out of date).
- Docs: Ollama Cloud setup (`OLLAMA_HOST=https://ollama.com` + `OLLAMA_API_KEY`).

### Changed
- `ollama`, `pypdf`, `python-docx` and `python-dotenv` are imported only when
  used, so the package also loads without them (for example in the browser).
  Nothing changes when they are installed.

## [0.6.1] - 2026-11-11

### Fixed
- `pip install rag-titan` no longer installs pytest: it was listed as a runtime
  dependency by mistake. It stays in the `dev` extra.

### Changed
- README shows the 0.6 features: `rag-titan evaluate` with a real report,
  `run()` / `--trace` step logs, and the Evaluating RAG page.

## [0.6.0] - 2026-11-11

### Added
- End-to-end evaluation (RAG handbook step 0): compare any ingestion with any
  RAG type on your own test questions.
  - `rag-titan evaluate --files ... --questions questions.json --rag-type simple
    adaptive --chunker word paragraph` runs every combination and prints a
    report (Recall, MRR, nDCG, answered rate, steps, p50/p95 time) and the
    questions that went wrong. `--retrieval-only --top-k 5` scores search alone.
  - `compare_setups(files, questions, setups)`, `evaluate_rag()` and
    `format_rag_report()`. Evidence is scored on the chunks given to the LLM;
    setups with the same settings share one index.
  - Retrieval metrics: `recall_at_k()`, `reciprocal_rank()`, `ndcg_at_k()`,
    `evaluate_retriever()`, `compare_retrievers()`, `format_report()`.
  - Questions file loader `load_questions()`. Evidence ids name a file
    (`"hr.pdf"`) or a page (`"hr.pdf#p4"`), so labels keep working when the
    chunker changes; each evidence counts once.
- `run(question)` on every RAG type: the answer, the chunks used (`sources`) and
  every step with its input, output and time. `describe_steps()` prints them as
  a readable log; `rag-titan ask ... --trace` shows it in the terminal.
- `FLOW` on every RAG type: its steps and the arrows between them, for flow
  diagrams. A test checks that every run follows its FLOW.
- `RAGTitan.run()`, `RAGTitan.evaluate()` and `RAGTitan.evaluate_retrieval()`.
- `BaseRetriever`, so any retriever can be used by every RAG type, and
  `DenseRetriever` (the retriever from before; `Retriever` still works).
  Results also say which retriever found them: `"retriever"` and
  `"score_trace"`, e.g. `{"dense": 0.82}`.
- `BaseRAG` and `Trace` for writing new RAG types; `CHUNKERS` in the factory.
- Example `examples/evaluate_retrieval.py` with sample policies and questions.
- Docs: "Evaluating RAG" page with an animated end-to-end flow diagram.

### Changed
- `retrieve()` raises `ValueError` for a `top_k` that is not a whole number of
  at least 1 (it returned an empty list for 0 before).
- Search results have two more keys, `"retriever"` and `"score_trace"`. Code
  that compares whole result dicts must ignore them.

## [0.5.0] - 2026-11-11

### Added
- Search results no longer hold near-copies (chunking handbook 2.4). With
  overlap, neighbouring chunks share text, so a search could return the same
  passage twice. The retriever now fetches twice as many results and drops one
  that overlaps a better result from the same file by at least half of the
  shorter chunk, using the chunk offsets.
- Setting `DEDUPE_OVERLAP` (default `0.5`; `off` keeps every result), and
  `Retriever(..., dedupe_overlap=0.5)`.
- `rag_titan.retrieval.overlap_fraction()` and `remove_duplicates()`.

### Changed
- Retrieval returns different results than 0.4.x when chunks overlap: fewer
  near-copies, more different passages. Set `DEDUPE_OVERLAP=off` for the old
  behaviour.

### Fixed
- Docs site: the 0.4.0 docs were not published, because a docstring
  (`BaseChunker.chunk`) was read as a broken link in the API reference.

## [0.4.0] - 2026-11-11

### Added
- Chunk offsets (chunking handbook section 1): every chunk gets
  `start_offset` and `end_offset`, so `document["text"][start:end]` is where it
  comes from.
  - Exact for the character, word, recursive, sentence, paragraph, heading and
    page chunkers, even in very repetitive text.
  - `TokenChunker` and chunkers you write yourself get offsets by searching for
    each chunk in order (`None` when a chunk cannot be found).
- `page_numbers` on chunks of documents with pages (PDFs), and `section_path`
  on chunks of documents with headings, with any chunker.
- The answer context shows where each chunk comes from, for example
  `[chunk_4] Source: handbook.pdf, page 3, section: Leave policy > Sick leave`.
- `BaseChunker.split_spans(text)`: override it in your own chunker to report
  exact positions.
- New modules: `rag_titan.chunking.offsets`, `rag_titan.chunking.locations` and
  `rag_titan.chunking.headings`.

### Changed
- The answer prompt lets the LLM cite pages, such as `[chunk_0, page 3]`.
- The Markdown heading rules moved to `rag_titan.chunking.headings`; importing
  them from `heading_chunker` still works. Chunk text is unchanged.

## [0.3.1] - 2026-11-10

### Added
- Docs: a "Chunk playground" page. Try the chunkers on your own text in the
  browser, load example texts with one click, and follow "things to try".
- Docs: an animated diagram for each of the 8 chunkers, showing it run on a
  real text one chunk at a time, with the overlap highlighted.
- `tools/diagrams/`: draws those diagrams from the chunkers' real output, and
  `--check` fails when a diagram no longer matches the code. Not part of the
  package.

### Changed
- README: links to the chunk playground and the chunking guide.
- Docs: the chunking page links to the playground instead of embedding it.

## [0.3.0] - 2026-11-10

### Added
- Structure-aware chunking (chunking handbook section 3), in
  `rag_titan.chunking.structure_aware`:
  - `SentenceChunker`: packs whole sentences.
  - `ParagraphChunker`: packs whole paragraphs; a long one falls back to
    sentences, then words.
  - `HeadingChunker`: never mixes two sections and starts each chunk with its
    heading path, for example "Leave policy > Sick leave" (`include_path=False`
    keeps the original heading lines instead).
  - `PageChunker`: never puts two PDF pages in one chunk.
  - `split_sentences()` and `split_paragraphs()`: no library needed; handle
    abbreviations, initials, list numbers, decimals and CJK/Hindi end marks,
    and never lose a character.
- Settings: `CHUNKER=sentence`, `paragraph`, `heading` or `page`.
- `TextLoader` for `.txt`, `.md` and `.markdown` files.
- `rag_titan.chunking.packing.pack()`: the shared "whole pieces up to a size"
  step, now also used by `RecursiveChunker` (same output as before).
- Docs: a "Structure-aware chunkers" section on the Chunking methods page, and
  sentence and paragraph modes in the chunk playground.

### Changed
- Loaders keep the document's structure. DOCX paragraphs are separated by a
  blank line and Word headings become Markdown headings (`#`, `##`, ...); PDF
  pages are separated by a page break (`\f`) instead of a space.
  - `WordChunker` gives the same chunks as before, except that DOCX headings now
    add `#` marks.
  - `CharacterChunker`, `TokenChunker` and `RecursiveChunker` see the real
    paragraphs, so their chunks of PDF and DOCX files differ from 0.2.x.
    Re-index your files after upgrading.

## [0.2.1] - 2026-11-10

### Added
- Docs: a "Chunking methods" page that explains every chunker with
  formulas (stride, chunk count, overlap cost), runnable examples with real
  output, three animated diagrams, and an interactive chunk playground that
  runs in the browser.

### Changed
- Docs: "Building blocks" links to the new page instead of repeating it; the
  docs site can now show maths.

### Fixed
- README: the PyPI version badge showed an old cached version.

## [0.2.0] - 2026-11-10

### Added
- Fixed-length chunking (chunking handbook section 2):
  - `CharacterChunker`: fixed windows of characters, text kept exactly.
  - `TokenChunker`: fixed windows of model tokens. Uses tiktoken
    (`pip install "rag-titan[tokens]"`) or your own tokenizer with
    `encode()` / `decode()`.
  - `RecursiveChunker`: splits on paragraphs, lines, sentences and words
    before cutting, with whole-piece overlap; counts characters or tokens.
  - `fixed_windows()`: the shared sliding-window helper.
  - `count_windows()`: how many chunks a text makes, to see the cost of overlap.
- Settings: `CHUNKER=character`, `token` or `recursive`, and `CHUNK_ENCODING`.
- Optional install extra `tokens` (tiktoken).

### Changed
- Chunkers are grouped by family: fixed-size chunkers live in
  `rag_titan.chunking.fixed_size`. `from rag_titan.chunking import ...` and the
  old `rag_titan.chunking.word_chunker` path keep working.
- `WordChunker` is built on `fixed_windows()`; its output is unchanged.
- Docs: fixed-length chunkers, overlap cost, settings for every chunker, and
  where new chunkers go.
- README: banner header, a short summary, and links to the docs site.
- README and docs: a clear "Supported RAG types" table (simple, adaptive) and a
  separate list of planned RAG types.
- Docs: logo and favicon, banner on the home page.
- Fixed the number of LLM calls for adaptive RAG in the docs and diagram (2 to 5).

## [0.1.0] - 2026-11-10

### Added
- First release of `rag-titan`.
- RAG types: `SimpleRAG` and `AdaptiveRAG`.
- `RAGTitan` class: index files and ask questions in two lines.
- `rag-titan` command line tool (`ask` and `chat`).
- Building blocks: PDF/DOCX loaders, word chunker, Ollama LLM and embedder,
  in-memory vector store, keyword reranker, grounded generator.
- Tool-calling `BaseAgent` with an example `MathAgent`.
- Tests that run without Ollama, docs site, and PyPI publish workflow.
- Animated flow diagrams (SVG) for every capability, with worked examples.
