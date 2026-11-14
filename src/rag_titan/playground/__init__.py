"""
The RAG playground on the docs site: the real package runs in the browser (Pyodide)
and replays model runs recorded with tools/playground/record_traces.py.
"""

from rag_titan.playground.live import LivePlayground
from rag_titan.playground.presets import (
    PRESET_CHUNKING,
    PRESET_KEYS,
    build_parts,
    check_presets,
    index_documents,
    preset_settings,
    record_presets,
    replay_parts,
    summarize_run,
)
from rag_titan.playground.session import Playground

__all__ = [
    "LivePlayground",
    "PRESET_CHUNKING",
    "PRESET_KEYS",
    "Playground",
    "build_parts",
    "check_presets",
    "index_documents",
    "preset_settings",
    "record_presets",
    "replay_parts",
    "summarize_run",
]
