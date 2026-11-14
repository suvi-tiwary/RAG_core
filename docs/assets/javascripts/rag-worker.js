// Web Worker (a module worker) for the RAG playground: runs the real rag_titan package with Pyodide,
// away from the page, so the page stays responsive while Python loads and runs.
// Messages in:  {id, call, args, assets}; calls: info, run, evaluate (recorded),
//               models (list local Ollama models), live_run (your Ollama)
// Messages out: {id, result} or {id, error}, plus {status: "..."} while loading.

import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";

const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/";
let ready = null;   // a promise for the Python dispatcher

async function fetchOk(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${url} (${response.status})`);
  return response;
}

// Calls to local Ollama are synchronous: allowed in workers, and the Python code
// (RAG types, retriever, ...) stays the same plain synchronous code as everywhere else.
function ollama(base, path, body) {
  const request = new XMLHttpRequest();
  request.open(body ? "POST" : "GET", `${base}${path}`, false);
  // text/plain: Ollama reads the JSON anyway, and the browser needs no extra CORS request
  if (body) request.setRequestHeader("Content-Type", "text/plain");
  try {
    request.send(body ? JSON.stringify(body) : null);
  } catch (error) {
    throw new Error(`Could not reach Ollama at ${base}. Is it running, and does OLLAMA_ORIGINS `
      + `allow ${self.location.origin}? (${error.message})`);
  }
  if (request.status === 403) {
    throw new Error(`Ollama refused this page: start it with OLLAMA_ORIGINS=${self.location.origin}`);
  }
  if (request.status !== 200) throw new Error(`Ollama ${path}: ${request.status} ${request.responseText}`);
  return JSON.parse(request.responseText);
}
self.ollamaChat = (base, model, messagesJson) => ollama(base, "/api/chat",
  { model, messages: JSON.parse(messagesJson), stream: false }).message.content;
self.ollamaEmbed = (base, model, textsJson) => JSON.stringify(ollama(base, "/api/embed",
  { model, input: JSON.parse(textsJson) }).embeddings);

const DISPATCH = `
import json, sys
sys.path.insert(0, "/home/pyodide/lib")
import js
from rag_titan.playground import LivePlayground, Playground

PRESETS = json.loads(PRESETS_TEXT)
PLAYGROUND = Playground(PRESETS)
LIVE = {}

def live(base, model, embed_model):
    key = (base, model, embed_model)
    if key not in LIVE:
        chat = lambda messages: js.ollamaChat(base, model, json.dumps(messages))
        embed = lambda texts: json.loads(js.ollamaEmbed(base, embed_model, json.dumps(texts)))
        LIVE[key] = LivePlayground(PRESETS, chat, embed, model, embed_model)
    return LIVE[key]

def handle(call, args):
    if call == "live_run":
        return live(*args[:3]).run(*args[3:])
    if call in ("info", "run", "evaluate"):
        return getattr(PLAYGROUND, call)(*args)
    raise ValueError(f"Unknown call {call}")
`;

async function start(assets) {
  postMessage({ status: "Loading Python (Pyodide, about 10 MB the first time)…" });
  const pyodide = await loadPyodide({ indexURL: PYODIDE });
  postMessage({ status: "Loading numpy…" });
  await pyodide.loadPackage("numpy");
  postMessage({ status: "Loading rag-titan and the recorded runs…" });
  const [bundle, presets] = await Promise.all([
    fetchOk(`${assets}playground/rag_titan.zip`).then((r) => r.arrayBuffer()),
    fetchOk(`${assets}playground/presets.json`).then((r) => r.text()),
  ]);
  pyodide.unpackArchive(bundle, "zip", { extractDir: "/home/pyodide/lib" });
  pyodide.globals.set("PRESETS_TEXT", presets);
  pyodide.runPython(DISPATCH);
  const handle = pyodide.globals.get("handle");
  return (call, args) => JSON.parse(pyodide.globals.get("json").dumps(handle(call, pyodide.toPy(args))));
}

// "Traceback ... ValueError: Please enter a question" -> "Please enter a question"
function shortError(error) {
  const lines = String(error.message || error).trim().split("\n");
  return lines[lines.length - 1].replace(/^[A-Za-z.]*(Error|Exception): /, "");
}

onmessage = async (event) => {
  const { id, call, args, assets } = event.data;
  try {
    if (call === "models") {   // pure JavaScript: no need to wait for Python
      postMessage({ id, result: ollama(args[0], "/api/tags").models.map((m) => m.name) });
      return;
    }
    if (!ready) ready = start(assets);
    postMessage({ id, result: (await ready)(call, args || []) });
  } catch (error) {
    postMessage({ id, error: shortError(error) });
  }
};
