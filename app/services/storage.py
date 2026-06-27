"""File storage for uploaded documents.

Every write goes through `save_upload`. Phase 1 writes to the local filesystem;
Phase 2 swaps the body of this one function for an S3 put, with no change to any
caller. (DECISIONS: D-6b)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

# A scratch directory for raw uploads. We only need the bytes long enough to
# validate and parse them — the chunks (not the original file) are the system of
# record in Phase 1.
UPLOAD_DIR = Path(tempfile.gettempdir()) / "rag_uploads"


def save_upload(filename: str, content: bytes) -> str:
    """Persist the uploaded bytes and return the path they were written to."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / filename
    path.write_bytes(content)
    return str(path)
