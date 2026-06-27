"""Validate and parse uploaded documents.

The order matters (DECISIONS: D-2): we validate the *raw bytes* first — empty,
size, MIME, duplicate content, and (for PDFs) password/scanned — and only parse a
file once it has passed. A file that fails validation is never handed to a parser.

Parsing uses a dedicated library per format so the documented per-format edge
cases are explicit and testable. Rejections raise `DocumentError`, whose `code`
maps onto the API's RFC-7807 error types.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Optional

import chardet
import magic
import pandas as pd
import yaml
from docx import Document as DocxDocument
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.config import get_settings

SUPPORTED_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".csv", ".xlsx", ".xls", ".json", ".yaml", ".yml", ".docx",
}

_TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".yaml", ".yml"}


class DocumentError(Exception):
    """A document that cannot be ingested. `code` matches an API error type."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def content_hash(content: bytes) -> str:
    """SHA-256 of the raw bytes, first 32 hex chars — used for dedup. (C-C)"""
    return hashlib.sha256(content).hexdigest()[:32]


def load_document(
    filename: str,
    content: bytes,
    existing_hashes: Optional[dict[str, str]] = None,
) -> dict:
    """Validate then parse. Returns a dict with the extracted text + metadata.

    `existing_hashes` maps content_hash -> filename for already-indexed docs, used
    to reject identical content uploaded under a different name. Raises
    DocumentError on any rejection.
    """
    existing_hashes = existing_hashes or {}
    ext = Path(filename).suffix.lower()
    settings = get_settings()

    if len(content) == 0:
        raise DocumentError("empty-file", "The uploaded file is empty.")

    limit = settings.upload_size_limit_mb * 1024 * 1024
    if len(content) > limit:
        raise DocumentError(
            "file-too-large",
            f"File size exceeds the {settings.upload_size_limit_mb}MB limit. "
            "Please upload a smaller file or split the document.",
        )

    if ext not in SUPPORTED_EXTENSIONS:
        raise DocumentError("unsupported-file-type", f"Unsupported file type '{ext}'.")

    _validate_mime(ext, content)

    digest = content_hash(content)
    twin = existing_hashes.get(digest)
    if twin and twin != filename:
        # Same content under a different name → reject. Same filename = replace (allowed).
        raise DocumentError(
            "duplicate-content",
            f"This document appears to be identical to an already indexed file: {twin}. "
            "Uploading duplicate content degrades answer quality.",
        )

    text, warnings = _parse(ext, content)
    if not text.strip():
        raise DocumentError(
            "empty-file", "No readable text could be extracted from the document."
        )

    return {
        "filename": filename,
        "content_hash": digest,
        "size_bytes": len(content),
        "text": text,
        "warnings": warnings,
    }


# ── validation ────────────────────────────────────────────────────────────────

def _validate_mime(ext: str, content: bytes) -> None:
    """Best-effort check that the bytes match the declared extension.

    Catches the "a .pdf that isn't really a PDF" case. Text formats all look like
    text/* to libmagic, so the check is necessarily loose for those.
    """
    mime = magic.from_buffer(content, mime=True)

    # libmagic is vague on Office files (zip- or OLE-based): a docx may come back
    # as octet-stream, an xlsx as application/zip. Accept that family here; a truly
    # corrupt Office file is caught later when its parser fails.
    office_ok = "openxmlformats" in mime or mime in {
        "application/zip", "application/octet-stream", "application/x-ole-storage",
        "application/vnd.ms-excel", "application/CDFV2",
    }

    if ext == ".pdf":
        ok = mime == "application/pdf"
    elif ext in (".docx", ".xlsx", ".xls"):
        ok = office_ok
    elif ext in _TEXT_EXTENSIONS:
        ok = mime.startswith("text/") or mime in ("application/json", "application/csv")
    else:
        ok = True

    if not ok:
        raise DocumentError(
            "unsupported-file-type", "File content does not match the declared file type."
        )


# ── parsing (one helper per format) ─────────────────────────────────────────────

def _parse(ext: str, content: bytes) -> tuple[str, list[str]]:
    if ext == ".pdf":
        return _parse_pdf(content)
    if ext in (".txt", ".md"):
        return _parse_text(content)
    if ext == ".csv":
        return _parse_csv(content)
    if ext in (".xlsx", ".xls"):
        return _parse_excel(content)
    if ext == ".json":
        return _flatten_structured(json.loads(content.decode("utf-8", "replace"))), []
    if ext in (".yaml", ".yml"):
        return _flatten_structured(yaml.safe_load(content.decode("utf-8", "replace"))), []
    if ext == ".docx":
        return _parse_docx(content), []
    raise DocumentError("unsupported-file-type", f"Unsupported file type '{ext}'.")


def _parse_pdf(content: bytes) -> tuple[str, list[str]]:
    try:
        reader = PdfReader(io.BytesIO(content))
    except PdfReadError:
        raise DocumentError("unsupported-file-type", "The PDF could not be read.")

    if reader.is_encrypted:
        # We don't have the password, so it stays unreadable.
        raise DocumentError(
            "password-protected",
            "This PDF is password protected and cannot be read. "
            "Please remove the password protection before uploading.",
        )

    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    if not text.strip():
        # No text layer → almost certainly a scan. OCR is out of scope for Phase 1.
        raise DocumentError(
            "scanned-pdf",
            "This PDF appears to be scanned or image-based. Text could not be extracted. "
            "Please use a PDF with a text layer or convert it using OCR first.",
        )
    return text, []


def _parse_text(content: bytes) -> tuple[str, list[str]]:
    # Detect the encoding rather than assuming UTF-8; fall back to replacement.
    guess = chardet.detect(content)
    encoding = guess.get("encoding") or "utf-8"
    warnings: list[str] = []
    try:
        text = content.decode(encoding)
        if encoding.lower() not in ("utf-8", "ascii"):
            warnings.append(f"Non-UTF-8 encoding detected ({encoding}).")
    except (UnicodeDecodeError, LookupError):
        text = content.decode("utf-8", errors="replace")
        warnings.append("Encoding could not be determined; some characters may be replaced.")
    return text, warnings


def _parse_csv(content: bytes) -> tuple[str, list[str]]:
    text = content.decode("utf-8", errors="replace")
    sample = text[:2048]
    warnings: list[str] = []

    try:
        has_header = csv.Sniffer().has_header(sample)
    except csv.Error:
        has_header = True  # ambiguous → assume the first row is a header

    rows = [r for r in csv.reader(io.StringIO(text)) if any(cell.strip() for cell in r)]
    if not rows:
        return "", warnings

    if has_header:
        header, data_rows = rows[0], rows[1:]
    else:
        header = [f"Col_{i + 1}" for i in range(len(rows[0]))]
        data_rows = rows
        warnings.append("No column headers detected. Generic column names (Col_1, Col_2...) were assigned.")

    lines = [
        " | ".join(f"{header[i]}: {cell}" for i, cell in enumerate(row) if i < len(header))
        for row in data_rows
    ]
    return "\n".join(lines), warnings


def _parse_excel(content: bytes) -> tuple[str, list[str]]:
    try:
        book = pd.ExcelFile(io.BytesIO(content))
    except Exception:
        raise DocumentError("unsupported-file-type", "The spreadsheet could not be read.")
    lines: list[str] = []
    for sheet in book.sheet_names:
        frame = book.parse(sheet).fillna("")
        for _, row in frame.iterrows():
            cells = " | ".join(f"{col}: {val}" for col, val in row.items())
            lines.append(f"[{sheet}] {cells}")  # sheet name kept for traceability
    warnings = ["Charts, images, and complex formatting were ignored. Only cell values were indexed."]
    return "\n".join(lines), warnings


def _parse_docx(content: bytes) -> str:
    try:
        document = DocxDocument(io.BytesIO(content))
    except Exception:
        raise DocumentError("unsupported-file-type", "The Word document could not be read.")
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _flatten_structured(obj, prefix: str = "") -> str:
    """Flatten nested JSON/YAML into readable `key: value` lines for retrieval."""
    lines: list[str] = []

    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")
        else:
            lines.append(f"{path}: {node}")

    walk(obj, prefix)
    return "\n".join(lines)
