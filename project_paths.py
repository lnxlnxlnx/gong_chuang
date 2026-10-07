"""Locate project resources without depending on depth or working directory.

Before the ``modules/`` layout, each loader computed the repository root with
``Path(__file__).resolve().parents[N]``.  That silently points somewhere else as
soon as the file moves one level deeper, and the tests that write to
``output/`` depended on the current working directory instead.

Two rules keep paths honest from here on:

* project-wide resources (``models/``, ``output/``) come from
  :data:`PROJECT_ROOT`, which this file's own location pins to the repository
  root no matter which module or directory asks;
* a resource that belongs to one module (its own ``config.json``) is resolved
  relative to that module's ``__file__``, so it stays correct at any depth.

``models/`` and ``output/`` are git-ignored and created on demand, so nothing
here asserts their contents.
"""

from __future__ import annotations

from pathlib import Path

#: Repository root: the directory holding ``main.py``, ``test.py`` and ``modules/``.
PROJECT_ROOT = Path(__file__).resolve().parent


def project_path(*parts: str) -> Path:
    """Return ``PROJECT_ROOT/parts...`` regardless of the current directory."""
    return PROJECT_ROOT.joinpath(*parts)


def models_path(*parts: str) -> Path:
    """Return a path under the git-ignored ``models/`` directory."""
    return project_path("models", *parts)


def output_path(*parts: str) -> Path:
    """Return a path under the git-ignored ``output/`` directory.

    Callers create the directory they need; nothing here writes to disk.
    """
    return project_path("output", *parts)
