"""Add the chunking and embedding inputs to my-data.json and save eval-data.json."""

import json
from pathlib import Path

logs = json.loads(Path("my_rag_logs.json").read_text(encoding="utf-8"))
data = json.loads(Path("my-data.json").read_text(encoding="utf-8"))
queries = data["queries"]

# ---- chunking: your rules, and one vector per sentence -----------------------
data["required_metadata"] = ["source", "page"]     # keys every chunk must carry
data["size_limits"] = [5, 60]                       # a valid chunk has 5 to 60 words
data["max_context_tokens"] = 80                     # your prompt budget, in words
data["sentence_vectors"] = logs["sentence_vectors"]

# ---- embedding: the vectors and what your vector index returned --------------
data["embeddings"] = {
    "vectors": logs["vectors"],                                         # chunk id -> vector
    "query_vectors": {query["id"]: run["query_vector"]
                      for query, run in zip(queries, logs["runs"])},    # query id -> vector
    "relevant_ids": {query["id"]: query.get("expected_chunk_ids", []) for query in queries},
    "indexed_ids": list(logs["vectors"]),                               # ids in your index
    "index_results": {query["id"]: query["retrieved_ids"] for query in queries},
}

Path("eval-data.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
print("Wrote eval-data.json: chunking and embedding inputs added")
