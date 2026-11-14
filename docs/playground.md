# Chunk playground

Try the chunkers on your own text, right here in the browser. Pick a chunker,
move the **size** and **overlap** sliders, and watch the chunks change.

- Each coloured block is one chunk. The <mark>highlighted</mark> start of a
  chunk is the **overlap**, repeated from the chunk before.
- The grey box shows the numbers behind it: $N$ units, size $S$, overlap $O$,
  stride $H = S - O$ and the number of chunks $K$. For the sliding-window
  chunkers it also shows the formula and how much the overlap costs.
- Click an **example** to load a ready-made text, or type your own.

<div id="chunk-playground-app"><p><em>The playground needs JavaScript.</em></p></div>

## Things to try

1. **The cost of overlap.** Pick `character`, keep the size and move overlap
   from 0 up. "overlap adds …%" grows like $1 / (1 - O/S)$: half the size as
   overlap doubles the number of chunks.
2. **Sentences that are not sentences.** Load *Tricky sentences* and pick
   `sentence`. `Dr.`, `J. K.`, `3.5`, `1.`, `e.g.` and `U.S.` do not end a
   sentence, and the Japanese `。` does. Then switch to `character` and see
   words cut in half.
3. **Paragraph vs sentence.** Load *Markdown with headings*. `paragraph` keeps
   each heading with its text when they fit; `sentence` can join the end of one
   paragraph with the start of the next.
4. **When a unit is too big.** Load *One long paragraph*, pick `paragraph` and
   make the size smaller than the paragraph. It falls back to whole sentences;
   make it smaller than a sentence and it falls back to words.
5. **Whole-piece overlap.** With `recursive`, `sentence` or `paragraph`, set the
   overlap smaller than a sentence. Nothing is repeated, because these
   chunkers never cut a sentence to fill the overlap.

## What is (and is not) here

| Chunker     | In the playground | Why not                                        |
|-------------|-------------------|------------------------------------------------|
| `recursive` | yes               |                                                |
| `sentence`  | yes               |                                                |
| `paragraph` | yes               |                                                |
| `character` | yes               |                                                |
| `word`      | yes               |                                                |
| `token`     | no                | needs tiktoken, which does not run in a browser |
| `heading`   | no                | needs a whole document with headings           |
| `page`      | no                | needs a PDF with page breaks                   |

The playground uses the same rules as the Python package. It is checked
against the package on 4,000 random texts full of abbreviations, decimals and
other traps, so what you see here is what `rag_titan.chunking` returns.

How each chunker works, with formulas and diagrams:
**[Chunking methods](chunking.md)**.
