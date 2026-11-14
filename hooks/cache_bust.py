"""
MkDocs hook: add ?v=<package version> to the site's own CSS and JS files.

Browsers keep a stylesheet for a while. After a docs update they could mix an
old extra.css with new JavaScript. A new address for every release makes them
fetch the new files. Files from other sites (https://...) are left alone.
"""

import re
from pathlib import Path

VERSION_FILE = "src/rag_titan/__init__.py"


def read_version(config):
    """Return __version__ from the package, without importing it."""
    root = Path(config.config_file_path).parent
    text = (root / VERSION_FILE).read_text(encoding="utf-8")
    return re.search(r'__version__ = "([^"]+)"', text).group(1)


def with_version(path, version):
    """Add ?v=<version> to a local file path; leave web addresses unchanged."""
    if "://" in path or path.startswith("//") or "?" in path:
        return path
    return f"{path}?v={version}"


def on_config(config):
    """Runs once when MkDocs loads mkdocs.yml."""
    version = read_version(config)
    config.extra_css = [with_version(path, version) for path in config.extra_css]

    for index, script in enumerate(config.extra_javascript):
        # an entry is a plain string, or an object with .path when it has options
        if isinstance(script, str):
            config.extra_javascript[index] = with_version(script, version)
        else:
            script.path = with_version(script.path, version)

    return config
