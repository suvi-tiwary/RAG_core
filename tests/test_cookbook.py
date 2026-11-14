"""Tests that the evaluation cookbook downloads match what the evaluator really computes."""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

from rag_titan import __version__

ROOT = Path(__file__).parent.parent
BUILD_SCRIPT = ROOT / "tools" / "cookbook" / "build_examples.py"
COOKBOOK = ROOT / "docs" / "cookbook"

# the source package on PyPI has the tests but not docs/ and tools/
pytestmark = pytest.mark.skipif(not BUILD_SCRIPT.is_file(),
                                reason="the cookbook is only in the repository")


def load_build_script():
    """Import tools/cookbook/build_examples.py (tools/ is not a package)."""
    spec = importlib.util.spec_from_file_location("build_examples", BUILD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cookbook_assets_are_up_to_date(monkeypatch, capsys):
    """If this fails, the evaluator or the sample changed: run
    `python tools/cookbook/build_examples.py` to build the assets again."""
    monkeypatch.setattr(sys, "argv", ["build_examples.py", "--check"])
    load_build_script().main()

    assert "Cookbook assets verified" in capsys.readouterr().out


def test_recipes_pin_the_current_version():
    """The recipes promise exact numbers for one version: update them with every release."""
    pinned = set()
    for page in COOKBOOK.glob("*.md"):
        pinned |= set(re.findall(r"rag-titan==([0-9.]+[0-9])", page.read_text(encoding="utf-8")))

    assert pinned == {__version__}


def test_recipe_summaries_match_the_expected_reports():
    """Every "Overall score" line quoted in a recipe is the real summary of a kit file."""
    build = load_build_script()
    real = {(build.DEST / "expected" / f"{name}-summary.txt").read_text(encoding="utf-8")
            .splitlines()[0] for name in build.scenarios()}
    quoted = set()
    for page in ("first-evaluation.md", "diagnose.md"):
        quoted |= set(re.findall(r"^Overall score: .+$", (COOKBOOK / page).read_text(
            encoding="utf-8"), re.M))

    assert quoted and quoted <= real
