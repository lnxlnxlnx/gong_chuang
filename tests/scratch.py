"""Throwaway directories for tests that need to write real files.

``tempfile.TemporaryDirectory`` creates its directory with mode ``0o700``; on
Windows that becomes a restrictive security descriptor that confined processes
(for example an agent sandbox) cannot write into.  Creating the directory with
the default mode keeps the suite runnable in those environments too.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def scratch_directory() -> Iterator[Path]:
    """Yield a fresh, empty directory that is writable in this environment."""
    path = Path(tempfile.gettempdir()) / f"gong-chuang-test-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
