// Chunking rules for the playground, mirroring the Python package:
// rag_titan.chunking.fixed_size, packing.py and structure_aware/sentences.py.
// Text is handled as characters (code points) to match Python string lengths.

const ChunkCore = (() => {
  const SEPARATORS = ["\n\n", "\n", ". ", " ", ""];
  const PARAGRAPH_BREAK = /\n[ \t]*\n\s*/g;
  const SENTENCE_END = /[.!?…]+["'”’)\]]*\s+|[。！？।]+["'”’」』)）\]]*\s*/g;
  const ABBREVIATIONS = new Set(["mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs",
    "etc", "fig", "approx", "inc", "ltd", "dept", "est", "no", "vol", "p", "pp", "cf"]);
  const chars = (text) => Array.from(text);
  const isLower = (c) => c !== "" && c === c.toLowerCase() && c !== c.toUpperCase();

  function fixedWindows(units, size, overlap) {
    const step = size - overlap;
    const windows = [];
    for (let start = 0; start < units.length; start += step) {
      windows.push(units.slice(start, start + size));
      if (start + size >= units.length) break;   // tail is not repeated
    }
    return windows;
  }

  function countWindows(n, size, overlap) {
    if (n <= size) return Math.min(n, 1);
    return 1 + Math.ceil((n - size) / (size - overlap));
  }

  function byCharacter(text, size, overlap) {
    if (!text.trim()) return [];
    return fixedWindows(chars(text), size, overlap).map((w, i) => ({
      text: w.join(""), overlap: i === 0 ? 0 : overlap,
    }));
  }

  function byWord(text, size, overlap) {
    const words = text.split(/\s+/).filter(Boolean);
    return fixedWindows(words, size, overlap).map((w, i) => ({
      text: w.join(" "),
      // shared words plus the space after them, in characters
      overlap: i === 0 ? 0 : chars(w.slice(0, overlap).join(" ")).length + 1,
    }));
  }

  function splitKeepSeparator(text, sep) {
    const parts = text.split(sep);
    const pieces = parts.slice(0, -1).map((p) => p + sep).concat(parts[parts.length - 1]);
    return pieces.filter(Boolean);
  }

  function recursiveSplit(text, separators, size) {
    if (chars(text).length <= size) return [text];
    for (let i = 0; i < separators.length; i++) {
      const sep = separators[i];
      if (sep === "") break;
      if (text.includes(sep)) {
        return splitKeepSeparator(text, sep)
          .flatMap((part) => recursiveSplit(part, separators.slice(i + 1), size));
      }
    }
    return fixedWindows(chars(text), size, 0).map((w) => w.join(""));   // hard cut
  }

  // packing.py: whole pieces up to size, whole-piece overlap up to `overlap`
  function pack(sizes, size, overlap) {
    const total = (ids) => ids.reduce((sum, i) => sum + sizes[i], 0);
    const groups = [];
    let current = [];
    let carried = 0;
    sizes.forEach((s, index) => {
      if (current.length && total(current) + s > size) {
        groups.push({ ids: current, carried });
        let tail = [];
        for (const i of [...current].reverse()) {   // last whole pieces <= overlap
          if (total(tail) + sizes[i] > overlap) break;
          tail.unshift(i);
        }
        while (tail.length && total(tail) + s > size) tail.shift();
        current = tail;
        carried = tail.length;
      }
      current.push(index);
    });
    if (current.length) groups.push({ ids: current, carried });
    return groups;
  }

  function packPieces(pieces, size, overlap) {
    return pack(pieces.map((p) => chars(p).length), size, overlap).map(({ ids, carried }) => {
      const raw = ids.map((i) => pieces[i]).join("");
      const lead = chars(raw).length - chars(raw.trimStart()).length;
      const repeated = chars(ids.slice(0, carried).map((i) => pieces[i]).join("")).length;
      const text = raw.trim();
      // trimming can remove a whitespace-only new piece, so keep the highlight inside the chunk
      return { text, overlap: Math.min(Math.max(0, repeated - lead), chars(text).length) };
    }).filter((chunk) => chunk.text);
  }

  // structure_aware/sentences.py
  function cutAfter(text, ends) {
    const pieces = [];
    let start = 0;
    for (const end of ends) {
      if (end > start) { pieces.push(text.slice(start, end)); start = end; }
    }
    if (start < text.length) pieces.push(text.slice(start));
    return pieces;
  }

  const splitParagraphs = (text) =>
    cutAfter(text, [...text.matchAll(PARAGRAPH_BREAK)].map((m) => m.index + m[0].length));

  function isSentenceEnd(text, match) {
    const end = match.index + match[0].length;
    if (isLower(chars(text.slice(end, end + 2))[0] || "")) return false;
    if (match[0][0] === "." && !match[0].startsWith("..")) {
      const found = text.slice(0, match.index).match(/[^\s(\["'“‘]+$/);
      const word = found ? found[0].toLowerCase() : "";
      if (ABBREVIATIONS.has(word) || /^\p{Nd}+$/u.test(word)) return false;
      if (/^(?:[a-z]|(?:[a-z]\.)+[a-z])$/.test(word)) return false;
    }
    return true;
  }

  const splitSentences = (text) => splitParagraphs(text).flatMap((para) => cutAfter(para,
    [...para.matchAll(SENTENCE_END)].filter((m) => isSentenceEnd(para, m))
      .map((m) => m.index + m[0].length)));

  // structure_aware: cut too-big pieces smaller, then pack whole pieces
  function fit(pieces, size, smaller) {
    return pieces.flatMap((p) => (chars(p).length > size ? smaller(p) : [p]));
  }

  function byStructure(cut) {
    return (text, size, overlap) => {
      if (!text.trim()) return [];
      const words = (p) => recursiveSplit(p, [" ", ""], size);
      return packPieces(cut(text, size, words), size, overlap);
    };
  }

  const toSentences = (text, size, words) => fit(splitSentences(text), size, words);
  const toParagraphs = (text, size, words) =>
    fit(splitParagraphs(text), size, (p) => toSentences(p, size, words));
  const byRecursive = (text, size, overlap) => (text.trim()
    ? packPieces(recursiveSplit(text, SEPARATORS, size), size, overlap) : []);
  const count = (t) => chars(t).length;

  const UNITS = {
    // packed: whole pieces are packed, so the sliding-window formula does not apply
    recursive: { run: byRecursive, count, name: "characters", packed: "pieces" },
    sentence: { run: byStructure(toSentences), count, name: "characters", packed: "sentences" },
    paragraph: { run: byStructure(toParagraphs), count, name: "characters", packed: "paragraphs" },
    character: { run: byCharacter, count, name: "characters" },
    word: { run: byWord, count: (t) => t.split(/\s+/).filter(Boolean).length, name: "words" },
  };

  return { fixedWindows, countWindows, splitSentences, splitParagraphs, UNITS };
})();

if (typeof module !== "undefined") module.exports = ChunkCore;   // for tests in Node
