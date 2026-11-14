"""
All settings in one place.

Values are read from environment variables (or a `.env` file in the
folder where you run your code). Every setting has a safe default, so
the package works without any configuration.

Example:
    from rag_titan.config import get_settings

    settings = get_settings({"llm_model": "llama3.2", "chunk_size": 200})
"""

import os

try:
    from dotenv import load_dotenv
except ImportError:
    # python-dotenv is missing in the browser (Pyodide): settings then come only
    # from environment variables and overrides
    load_dotenv = None


def _get_int(name, default):
    """Read a whole number from the environment, or use the default."""
    value = os.getenv(name, str(default))

    try:
        return int(value)
    except ValueError:
        raise ValueError(f"Setting {name} must be a whole number, got '{value}'")


def _get_bool(name, default):
    """Read a yes/no value from the environment, or use the default."""
    value = os.getenv(name, str(default)).strip().lower()
    return value in ("1", "true", "yes", "on")


def _get_overlap(name, default):
    """Read a share between 0 and 1 (like 0.5), or "off" for None."""
    value = os.getenv(name, str(default)).strip().lower()
    if value in ("off", "none", "false", "no"):
        return None

    try:
        share = float(value)
    except ValueError:
        raise ValueError(f"Setting {name} must be a number like 0.5, or off; got '{value}'")
    if not 0 < share <= 1:
        raise ValueError(f"Setting {name} must be more than 0 and at most 1, or off")
    return share


def get_settings(overrides=None):
    """
    Build the settings dictionary.

    Args:
        overrides: optional dict of settings that win over the
            environment, for example {"chunk_size": 200}.

    Returns:
        A dict with every setting the package uses.
    """
    # read the .env file only when settings are requested, not on import
    if load_dotenv is not None:
        load_dotenv()

    settings = {
        # LLM used by RAG
        "llm_provider": os.getenv("LLM_PROVIDER", "ollama"),
        "llm_model": os.getenv("LLM_MODEL", "gemma3:4b"),
        "ollama_host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),

        # LLM used by agents (needs tool-calling support)
        "agent_model": os.getenv("AGENT_MODEL", "qwen3:4b"),
        "agent_think": _get_bool("AGENT_THINK", True),
        "agent_max_steps": _get_int("AGENT_MAX_STEPS", 5),

        # Embeddings
        "embed_provider": os.getenv("EMBED_PROVIDER", "ollama"),
        "embed_model": os.getenv("EMBED_MODEL", "nomic-embed-text"),

        # Chunking
        "chunker": os.getenv("CHUNKER", "word"),
        "chunk_size": _get_int("CHUNK_SIZE", 150),
        "chunk_overlap": _get_int("CHUNK_OVERLAP", 30),
        "chunk_encoding": os.getenv("CHUNK_ENCODING", "cl100k_base"),  # token chunker only

        # Storage, retrieval and generation
        "vector_store": os.getenv("VECTOR_STORE", "memory"),
        # drop a search result that overlaps a better one by this share ("off" keeps all)
        "dedupe_overlap": _get_overlap("DEDUPE_OVERLAP", 0.5),
        "reranker": os.getenv("RERANKER", "keyword"),
        "generator": os.getenv("GENERATOR", "grounded"),
    }

    if overrides:
        settings.update(overrides)

    return settings
