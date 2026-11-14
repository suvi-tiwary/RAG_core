# Contributing to rag-titan

Thanks for helping! This project is meant to be easy to read for beginners,
so simple code is more important than clever code.

## Set up

```bash
git clone https://github.com/aion7891/rag-titan.git
cd rag-titan
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,docs]"
```

## Before you open a pull request

```bash
pytest            # tests run without Ollama
ruff check .      # lint
mkdocs serve      # preview the docs at http://127.0.0.1:8000
```

## Code style

- functions under about 50 lines, files under about 200 lines
- no type hints, use plain dicts and lists
- every class and public function has a docstring (Google style: `Args:`, `Returns:`)
- comments explain **why** something is done
- new parts inherit from the matching base class (see `docs/extending.md`)
- add tests using `FakeLLM` / `FakeEmbedder` from `tests/conftest.py`
- add a line to `CHANGELOG.md`

## Releasing

Maintainers: see [RELEASING.md](RELEASING.md) for how to publish to PyPI.
