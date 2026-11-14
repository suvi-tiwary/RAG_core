---
hide:
  - toc
---

# RAG playground

Run the RAG types **in your browser** and watch every step. Pick one or more
chunkers and RAG types, choose a question, and each setup runs side by side on
its own flow diagram: the steps light up in the order they ran, with step
numbers, and the path taken turns orange. **Click any step** to see what went in
and what came out (the chunks found, with their files and scores, the rewritten
query, the evidence check).

<div id="rag-playground-app"><p><em>The playground needs JavaScript.</em></p></div>

## How it works

- The **real rag-titan code** runs in your browser with
  [Pyodide](https://pyodide.org) (Python compiled to WebAssembly). The first
  visit downloads about 10 MB; later visits use your browser's cache.
- In the default **recorded** mode, the model replies and embedding vectors are
  **real**, recorded once with
  `gemma3:4b` and `nomic-embed-text` on the three sample policies in
  `examples/sample_docs/`, then replayed. Search, reranking, the RAG types, the
  scores and the diagrams are all computed live.
- **Evaluate all questions** runs every chosen setup on the six test questions in
  `examples/questions.json` and scores them like
  [`rag-titan evaluate`](evaluation.md): did the chunks given to the model contain
  the evidence, and was the question answered?

## Things to try

1. **Chunking changes the answer.** Ask *When do I need a note from the doctor?*
   with `adaptive` on `word` and on `paragraph`. With word chunks the evidence
   check says "no" twice and it gives up after 9 steps; with paragraphs it
   answers in 5. Click **Retrieve** in the failed run: the right sentence *was*
   found, the model just did not trust it.
2. **A retry that works.** Ask *What is the maximum hotel cost per night?* with
   `adaptive` + `paragraph`: the first evidence check fails, the query is
   rewritten, and the wider search (`top_k` 4 → 8) succeeds.
3. **Two files needed.** Ask the furniture question with `simple`: it keeps only
   two chunks, so the second file may never reach the model.
4. **Compare everything.** Tick all chunkers and RAG types and press
   **Evaluate all questions**.

## Live mode: your Ollama, any question, your own text

!!! warning "Work in progress"
    Live mode does not work on this website yet: Chrome blocks a public
    website from reaching the Ollama on your computer (its new *local network
    access* protection), and the fix is still being made. It does work when you
    run the docs on your own machine (`mkdocs serve`). The recorded runs above
    work everywhere.

Switch **Mode** to *Live with your Ollama* to ask **any** question, on the sample
documents or on text you paste, with the models on your own machine. The page
talks to Ollama on your computer; nothing is sent anywhere else.

1. Ollama must allow this page. Start it with:

    ```bash
    OLLAMA_ORIGINS=https://aion7891.github.io ollama serve
    ```

    With the Ollama app on a Mac, run
    `launchctl setenv OLLAMA_ORIGINS https://aion7891.github.io` once, then quit
    and reopen the app.

2. Have a chat model and an embedding model, for example
   `ollama pull gemma3:4b` and `ollama pull nomic-embed-text`. Cloud models
   (`gpt-oss:20b-cloud`, ...) work too after `ollama signin`, if your Ollama
   account has credits.
3. Press **Connect**, pick the models, and **Run**.

Your browser may ask whether this page may **access devices on your local
network**: allow it, that is your own Ollama. Live runs take a few seconds per
setup, and the evidence is only scored for the listed questions on the sample
documents. **Evaluate all questions** is only offered for recorded runs: live it
would ask your model about 50 times.
