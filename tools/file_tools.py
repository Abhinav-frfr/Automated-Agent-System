"""Filesystem tools: search_files, read_file.

search_files now scans *filename AND content* and returns parsed invoice metadata
for every matching file, so the agent can identify "the latest" without reading
every candidate.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from pypdf import PdfReader

INVOICE_DIR = Path(__file__).resolve().parent.parent / "data" / "invoices"

_STOPWORDS = {
    "invoice", "invoices", "the", "a", "an", "from", "find", "latest", "most",
    "recent", "for", "of", "get", "retrieve", "look", "up", "please", "process",
    "and", "to", "my", "list", "all", "show", "me", "give", "month", "months",
    "check", "whether", "was", "were", "enter", "entered", "record", "recorded",
    "submit", "submitted", "into", "in", "system", "every", "each", "any",
    "how", "many", "total", "count", "which", "what", "when", "who", "then",
}

_FIELD_RES = {
    "vendor":         re.compile(r"Vendor:\s*(.+)", re.I),
    "invoice_number": re.compile(r"Invoice\s+Number:\s*(.+)", re.I),
    "invoice_date":   re.compile(r"Invoice\s+Date:\s*(.+)", re.I),
    "amount":         re.compile(r"Amount\s+Due:\s*\$?\s*([\d,]+(?:\.\d+)?)", re.I),
    "due_date":       re.compile(r"Due\s+Date:\s*(.+)", re.I),
}

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _keywords(query: str) -> list[str]:
    tokens = re.split(r"\W+", query.lower())
    return [t for t in tokens if t and t not in _STOPWORDS]


def _extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    return path.read_text(encoding="utf-8")


def _parse_invoice(text: str) -> dict:
    out: dict = {}
    for key, rx in _FIELD_RES.items():
        m = rx.search(text)
        if m:
            out[key] = m.group(1).strip()
    # Normalize amount to a float if we got one.
    if "amount" in out:
        try:
            out["amount"] = float(str(out["amount"]).replace(",", ""))
        except ValueError:
            pass
    # Add a sortable ISO date if we can parse it.
    if "invoice_date" in out:
        iso = _to_iso_date(out["invoice_date"])
        if iso:
            out["invoice_date_iso"] = iso
    if "due_date" in out:
        iso = _to_iso_date(out["due_date"])
        if iso:
            out["due_date_iso"] = iso
    return out


def _to_iso_date(raw: str) -> str | None:
    """Best-effort parse of 'September 1, 2026' -> '2026-09-01'."""
    raw = raw.strip()
    m = re.match(r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})", raw)
    if m:
        mon = _MONTHS.get(m.group(1).lower())
        if mon:
            return f"{int(m.group(3)):04d}-{mon:02d}-{int(m.group(2)):02d}"
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def search_files(query: str) -> dict:
    if not INVOICE_DIR.exists():
        return {"error": f"Invoice directory not found: {INVOICE_DIR}"}

    all_files = sorted(
        n for n in os.listdir(INVOICE_DIR) if n.lower().endswith((".pdf", ".txt"))
    )
    kws = _keywords(query)
    matches: list[dict] = []

    for name in all_files:
        lname = name.lower()
        matched_by: list[str] = []

        path = INVOICE_DIR / name
        try:
            text = _extract_text(path)
            meta = _parse_invoice(text)
        except Exception as e:
            text = ""
            meta = {"_read_error": f"{type(e).__name__}: {e}"}

        if kws:
            if any(k in lname for k in kws):
                matched_by.append("filename")
            blob = (text + " " + name).lower()
            if any(k in blob for k in kws) and "content" not in matched_by:
                matched_by.append("content")

        if not kws or matched_by:
            matches.append({"filename": name, "matched_by": matched_by or ["all"], **meta})

    return {"query": query, "count": len(matches), "files": matches}


def read_file(filename: str) -> dict:
    path = INVOICE_DIR / filename
    if not path.exists():
        return {"error": f"File not found: {filename}"}
    try:
        text = _extract_text(path)
    except Exception as e:
        return {"error": f"Failed to read {filename}: {e}"}
    return {
        "filename": filename,
        "content": text.strip(),
        "parsed": _parse_invoice(text),
    }