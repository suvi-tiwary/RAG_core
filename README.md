<p align="center">
  <a href="https://aion7891.github.io/rag-titan/"><img src="https://cdn.jsdelivr.net/gh/aion7891/rag-titan@main/docs/assets/images/banner.svg" alt="rag-titan: a Swiss Army knife for RAG. Many RAG types, one simple API, runs locally." width="100%"></a>
</p>

<p align="center">
  <a href="https://pypi.org/project/rag-titan/"><img src="https://img.shields.io/pypi/v/rag-titan?label=pypi" alt="PyPI version"></a>
  <a href="https://pypi.org/project/rag-titan/"><img src="https://img.shields.io/pypi/pyversions/rag-titan" alt="Python versions"></a>
  <a href="https://github.com/aion7891/rag-titan/actions/workflows/tests.yml"><img src="https://github.com/aion7891/rag-titan/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://aion7891.github.io/rag-titan/"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg" alt="Documentation"></a>
  <a href="https://github.com/aion7891/rag-titan/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
</p>

<p align="center">
  <a href="https://aion7891.github.io/rag-titan/"><b>📖 Documentation</b></a> ·
  <a href="https://aion7891.github.io/rag-titan/getting-started/">Getting started</a> ·
  <a href="https://aion7891.github.io/rag-titan/rag-types/">Supported RAG types</a> ·
  <a href="https://aion7891.github.io/rag-titan/api/core/">API reference</a> ·
  <a href="https://pypi.org/project/rag-titan/">PyPI</a>
</p>

<p align="center">
  <a href="https://aion7891.github.io/rag-titan/"><img src="https://cdn.jsdelivr.net/gh/aion7891/rag-titan@main/docs/assets/images/overview.svg" alt="rag-titan overview: index your files once, then ask questions" width="100%"></a>
</p>

## Why rag-titan?

- **Many RAG types, one API.** Two today (`simple`, `adaptive`), with more on the way. Switch with one word: `rag_type="adaptive"`.
- **Runs 100% locally** with [Ollama](https://ollama.com), so your documents never leave your machine.
- **Grounded answers.** The model answers only from your files and cites the chunks it used.
- **Chunking your way.** Eight chunkers: by words, characters or model tokens (tiktoken or your own tokenizer), recursive separators, and structure-aware chunkers that keep sentences, paragraphs, heading sections or pages whole. [Try them live in the chunk playground →](https://aion7891.github.io/rag-titan/playground/)
- **Watch every step, in your browser.** The [RAG playground →](https://aion7891.github.io/rag-titan/rag-playground/) runs the real package (Pyodide) side by side for any chunker and RAG type, and lights up each step on a live flow diagram: click a step to see the chunks it found. Recorded answers work instantly; a live mode with your own Ollama is work in progress.
- **PDF, Word, Markdown and text files**, with paragraphs, headings and pages kept for the chunkers.
- **Measure, don't guess.** Compare any chunker with any RAG type on your own questions in one command (`rag-titan evaluate`): recall, MRR, nDCG, answered rate and p95 time, plus every step of a run with `--trace`. [Evaluating RAG →](https://aion7891.github.io/rag-titan/evaluation/)
- **Find the stage that fails, in any RAG system.** Pass the logged outputs of your pipeline (rag-titan or not) and get 70+ metrics for chunking, embedding, retrieval, reranking, context, answers and end to end, a 0 to 100 scorecard, root-cause hints and regression checks. No LLM judge. [Evaluating every stage →](https://aion7891.github.io/rag-titan/stage-evaluation/)
- **Easy to read, easy to change.** Short files, plain Python, and a docstring on every class. Swap any part by inheriting one base class.

## Install

```bash
pip install rag-titan
ollama pull gemma3:4b && ollama pull nomic-embed-text
```

For token-based chunking, add the extra: `pip install "rag-titan[tokens]"`.

Needs Python 3.10+ and [Ollama](https://ollama.com/download). → [Full setup guide](https://aion7891.github.io/rag-titan/getting-started/)

## Quick start

```python
from rag_titan import RAGTitan

rag = RAGTitan(rag_type="simple")            # or "adaptive"
rag.index(["handbook.pdf", "policy.docx"])

print(rag.ask("How many leave days do I get?"))
# You get 20 paid leave days every year [chunk_0].

print(rag.run("How many leave days do I get?")["log"])   # every step it took
rag.evaluate("questions.json")                           # score it on your own test questions
```

Or from the terminal:

```bash
rag-titan ask --files handbook.pdf --question "How many leave days do I get?"
rag-titan ask --files handbook.pdf --question "How many leave days do I get?" --trace
```

## Evaluate on your own questions

Which chunker and which RAG type work best **for your documents**? Write a few
test questions, each naming the file (or page) that holds the answer, and compare
every combination in one command:

```bash
rag-titan evaluate --files docs/*.md --questions questions.json \
    --rag-type simple adaptive --chunker word paragraph
```

```text
setup                   Recall       MRR      nDCG  Answered     Steps     p50 s     p95 s
simple + word            1.000     1.000     1.000      100%       3.0      0.92      1.26
adaptive + word          1.000     1.000     1.000      100%       5.0      1.67      2.02
simple + paragraph       0.917     1.000     0.936      100%       3.0      0.64      0.81
adaptive + paragraph     1.000     1.000     0.987      100%       6.3      0.96      2.91

Questions to look at:
  simple + paragraph: q_006  recall 0.50  found ['remote_work.md']
```

Then see **why** one question went wrong. Here search found the right file, so
the problem is the answer step, not the chunker:

```text
$ rag-titan ask --files docs/*.md --question "Can I work from Spain for a month?" --trace
1. retrieve     7.47 ms  -> 3 chunks: chunk_2 remote_work.md (0.615), chunk_0 expenses.md (0.540), ...
2. rerank       0.04 ms  -> 2 chunks: chunk_2 remote_work.md (rerank 0.815), chunk_0 expenses.md (rerank 0.690)
3. generate   567.39 ms  -> "I don't know based on the available documents."
```

<p align="center">
  <a href="https://aion7891.github.io/rag-titan/evaluation/"><img src="https://cdn.jsdelivr.net/gh/aion7891/rag-titan@main/docs/assets/images/evaluation-flow.svg" alt="End-to-end evaluation: one index per chunker, every RAG type runs every question, the chunks given to the LLM are scored" width="100%"></a>
</p>

Real runs with `gemma3:4b` on the sample policies in `examples/sample_docs/`. Search-only scoring
(`--retrieval-only`, no LLM) and the Python API are in **[Evaluating RAG →](https://aion7891.github.io/rag-titan/evaluation/)**

## Evaluate every stage of any RAG system

A wrong answer does not say **which part** failed. Pass what each stage of your pipeline
produced, plus the right answers, and every stage is scored on its own. It works for any RAG
system, uses no LLM judge and needs no extra dependencies:

<p align="center">
  <a href="https://aion7891.github.io/rag-titan/cookbook/"><img src="https://cdn.jsdelivr.net/gh/aion7891/rag-titan@main/docs/assets/images/eval-flow.svg" alt="Evaluate any RAG: your logs and golden labels go into rag-titan evaluate, which scores every stage and names the first failing stage" width="100%"></a>
</p>

**Already have a RAG and only want to evaluate it?** Follow the 6 steps in the
**[Evaluation cookbook →](https://aion7891.github.io/rag-titan/cookbook/#choose-your-path)**

```python
from rag_titan.evaluation import RAGEvaluator

report = RAGEvaluator(k=5).evaluate(
    chunks=chunks,        # what your chunker produced
    queries=queries,      # per question: gold labels + retrieved ids, context, answer, latency
    document_text=text,   # optional: more metrics when given
)
print(report["scorecard"]["overall"]["score"])          # 0 to 100
print(report["root_causes"]["first_failing_stage"])     # e.g. "retrieval"
```

```bash
rag-titan evaluate --input data.json --baseline last_report.json --fail-on-regression
```

```text
| Pipeline stage       | Score / 100 | Rating    | Measurement coverage |
|----------------------|------------:|-----------|---------------------:|
| Document preparation |        97.6 | Excellent |                  74% |
| Semantic indexing    |        98.4 | Excellent |                 100% |
| Information search   |        79.4 | Good      |                 100% |
| Search ordering      |        89.7 | Very Good |                 100% |
| Answer context       |        92.3 | Excellent |                 100% |
| Answer quality       |       100.0 | Excellent |                 100% |
| User experience      |       100.0 | Excellent |                 100% |
```

<p align="center">
  <a href="https://aion7891.github.io/rag-titan/stage-evaluation/"><img src="https://cdn.jsdelivr.net/gh/aion7891/rag-titan@main/docs/assets/images/evaluation-architecture.svg" alt="Evaluation architecture: your data, RAGEvaluator with one evaluator per stage, the scorecard and the reports" width="100%"></a>
</p>

Using rag-titan itself? `rag.evaluate_stages(files, "golden.json")` collects the data for you.
Your own RAG? `rag-titan golden` drafts your golden questions and `build_dataset()` turns your
logs into the data file, see **[Evaluate your own RAG →](https://aion7891.github.io/rag-titan/cookbook/any-rag/)**.
Every formula is on the **[Evaluation metrics →](https://aion7891.github.io/rag-titan/evaluation-metrics/)** page,
and the guide is **[Evaluating every stage →](https://aion7891.github.io/rag-titan/stage-evaluation/)**

## Supported RAG types

**rag-titan supports 2 RAG types.** Pick one with `rag_type`. Every type has `answer()`, and
`run()` that also returns every step it took ([see the steps](https://aion7891.github.io/rag-titan/rag-types/#seeing-every-step-run)):

| Status | RAG type | How it works | Best for | LLM calls |
|---|---|---|---|---|
| ✅ Available | [**Simple RAG**](https://aion7891.github.io/rag-titan/rag-types/#simple-rag-simple)<br>`rag_type="simple"` | Retrieve → Rerank → Generate | Fast answers to clear questions | 1 |
| ✅ Available | [**Adaptive RAG**](https://aion7891.github.io/rag-titan/rag-types/#adaptive-rag-adaptive)<br>`rag_type="adaptive"` | Route → Retrieve → Check evidence → Rewrite & retry | Mixed questions, fewer wrong answers | 2–5 |

```python
from rag_titan import RAG_TYPES
print(RAG_TYPES)        # ['simple', 'adaptive']
```

**Coming next** (not available yet): Corrective RAG · Multi-query RAG · HyDE · Hybrid RAG · Self-RAG · Parent-document RAG · Graph RAG · Agentic RAG. → [Roadmap](https://aion7891.github.io/rag-titan/roadmap/)

> Also included: a tool-calling [agent](https://aion7891.github.io/rag-titan/agents/) loop. It's a separate feature, not a `rag_type`.

<p align="center">
  <a href="https://aion7891.github.io/rag-titan/rag-types/"><img src="https://cdn.jsdelivr.net/gh/aion7891/rag-titan@main/docs/assets/images/simple-rag.svg" alt="Simple RAG: retrieve, rerank, generate, with example scores" width="100%"></a>
</p>

<p align="center">
  <a href="https://aion7891.github.io/rag-titan/rag-types/"><img src="https://cdn.jsdelivr.net/gh/aion7891/rag-titan@main/docs/assets/images/adaptive-rag.svg" alt="Adaptive RAG: route, check evidence, rewrite and retry" width="100%"></a>
</p>

## Explore the docs

Click any picture to open its page.

<table>
  <tr>
    <td width="50%" valign="top">
      <a href="https://aion7891.github.io/rag-titan/rag-playground/"><img src="https://cdn.jsdelivr.net/gh/aion7891/rag-titan@main/docs/assets/images/rag-playground.png" alt="RAG playground: the same question with paragraph and word chunks, side by side, on live flow diagrams"></a>
      <p><b><a href="https://aion7891.github.io/rag-titan/rag-playground/">RAG playground</a></b><br>Run the RAG types in your browser and watch every step light up.</p>
    </td>
    <td width="50%" valign="top">
      <a href="https://aion7891.github.io/rag-titan/playground/"><img src="https://cdn.jsdelivr.net/gh/aion7891/rag-titan@main/docs/assets/images/playground.png" alt="Chunk playground: tricky sentences split by the sentence chunker"></a>
      <p><b><a href="https://aion7891.github.io/rag-titan/playground/">Chunk playground</a></b><br>Paste your text and watch the chunkers cut it, live in the browser.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <a href="https://aion7891.github.io/rag-titan/evaluation/"><img src="https://cdn.jsdelivr.net/gh/aion7891/rag-titan@main/docs/assets/images/evaluation-flow.svg" alt="End-to-end evaluation flow"></a>
      <p><b><a href="https://aion7891.github.io/rag-titan/evaluation/">Evaluating RAG</a></b><br>Compare any chunker with any RAG type on your own questions.</p>
    </td>
    <td width="50%" valign="top">
      <a href="https://aion7891.github.io/rag-titan/chunking/"><img src="https://cdn.jsdelivr.net/gh/aion7891/rag-titan@main/docs/assets/images/fixed-chunkers.svg" alt="One sentence chunked by characters, words and tokens"></a>
      <p><b><a href="https://aion7891.github.io/rag-titan/chunking/">Chunking methods</a></b><br>All 8 chunkers explained, with formulas and animated diagrams.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <a href="https://aion7891.github.io/rag-titan/building-blocks/"><img src="https://cdn.jsdelivr.net/gh/aion7891/rag-titan@main/docs/assets/images/indexing.svg" alt="Indexing pipeline"></a>
      <p><b><a href="https://aion7891.github.io/rag-titan/building-blocks/">Building blocks</a></b><br>Load → chunk → embed → store, with the data at every step.</p>
    </td>
    <td width="50%" valign="top">
      <a href="https://aion7891.github.io/rag-titan/extending/"><img src="https://cdn.jsdelivr.net/gh/aion7891/rag-titan@main/docs/assets/images/architecture.svg" alt="Swappable architecture"></a>
      <p><b><a href="https://aion7891.github.io/rag-titan/extending/">Extending rag-titan</a></b><br>Add your own LLM, vector store, chunker or RAG type.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <a href="https://aion7891.github.io/rag-titan/agents/"><img src="https://cdn.jsdelivr.net/gh/aion7891/rag-titan@main/docs/assets/images/agent.svg" alt="Agent loop"></a>
      <p><b><a href="https://aion7891.github.io/rag-titan/agents/">Agents</a></b><br>Let the LLM call your Python functions as tools.</p>
    </td>
    <td width="50%" valign="top">
      <a href="https://aion7891.github.io/rag-titan/cli/"><img src="https://cdn.jsdelivr.net/gh/aion7891/rag-titan@main/docs/assets/images/cli.svg" alt="Command line demo"></a>
      <p><b><a href="https://aion7891.github.io/rag-titan/cli/">Command line</a></b><br><code>rag-titan ask</code> (with <code>--trace</code>), <code>chat</code> and <code>evaluate</code>.</p>
    </td>
  </tr>
</table>

| I want to... | Read |
|---|---|
| Install it and ask my first question | [Getting started](https://aion7891.github.io/rag-titan/getting-started/) |
| Pick the right RAG type | [RAG types](https://aion7891.github.io/rag-titan/rag-types/) |
| Pick the right chunker | [Chunking methods](https://aion7891.github.io/rag-titan/chunking/) |
| Run RAG types in my browser and watch each step | [RAG playground](https://aion7891.github.io/rag-titan/rag-playground/) |
| Try the chunkers on my own text | [Chunk playground](https://aion7891.github.io/rag-titan/playground/) |
| Compare chunkers and RAG types on my questions | [Evaluating RAG](https://aion7891.github.io/rag-titan/evaluation/) |
| Evaluate the RAG I already built, step by step | [Evaluation cookbook](https://aion7891.github.io/rag-titan/cookbook/) |
| Score every stage of any RAG pipeline | [Evaluating every stage](https://aion7891.github.io/rag-titan/stage-evaluation/) |
| Look up a metric and its formula | [Evaluation metrics](https://aion7891.github.io/rag-titan/evaluation-metrics/) |
| See why an answer went wrong | [Seeing every step](https://aion7891.github.io/rag-titan/rag-types/#seeing-every-step-run) |
| Change models, chunk size or reranker | [Configuration](https://aion7891.github.io/rag-titan/configuration/) |
| See every class and method | [API reference](https://aion7891.github.io/rag-titan/api/core/) |

## Contributing

Contributions are welcome! Read [CONTRIBUTING.md](https://github.com/aion7891/rag-titan/blob/main/CONTRIBUTING.md) to get set up. The tests run without Ollama.

## License

[MIT](https://github.com/aion7891/rag-titan/blob/main/LICENSE) © Kumar Nishant
