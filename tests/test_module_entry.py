"""Tests for `python -m rag_titan`: the CLI without the rag-titan(.exe) launcher."""

import importlib
import subprocess
import sys

from rag_titan import __version__


def run_module(*args):
    return subprocess.run([sys.executable, "-m", "rag_titan", *args],
                          capture_output=True, text=True, timeout=60)


def test_version_through_python_m():
    done = run_module("--version")
    assert done.returncode == 0
    assert done.stdout.strip() == f"rag-titan {__version__}"


def test_sub_commands_are_the_same(tmp_path):
    (tmp_path / "questions.txt").write_text("Who pays?\n", encoding="utf-8")
    output = tmp_path / "golden.json"
    done = run_module("golden", "--questions", str(tmp_path / "questions.txt"),
                      "--output", str(output))
    assert done.returncode == 0, done.stderr
    assert output.is_file()


def test_bad_command_shows_usage():
    done = run_module("nope")
    assert done.returncode == 2 and "invalid choice" in done.stderr


def test_importing_does_not_run_the_cli():
    module = importlib.import_module("rag_titan.__main__")
    assert callable(module.main)
