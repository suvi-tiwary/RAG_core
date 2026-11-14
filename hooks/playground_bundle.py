"""
MkDocs hook: put the rag_titan package into the built site as a zip file.

The RAG playground page downloads assets/playground/rag_titan.zip and runs it
in the browser with Pyodide, so the page always runs the same code as the docs
were built from. Only .py files go in; caches and compiled files stay out.
"""

import zipfile
from pathlib import Path

PACKAGE = "src/rag_titan"
TARGET = "assets/playground/rag_titan.zip"


def on_post_build(config):
    """Runs once after MkDocs has written the site."""
    root = Path(config.config_file_path).parent
    package = root / PACKAGE
    target = Path(config.site_dir) / TARGET
    target.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(package.rglob("*.py")):
            # stored as rag_titan/..., so unpacking it gives an importable package
            bundle.write(path, path.relative_to(package.parent))
