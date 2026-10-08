"""Mock internal accounting system (JSON-backed).

Business invariants enforced here (not by the LLM):
- submit_invoice() rejects duplicates by (vendor, amount, due_date) or invoice_number.
- verify_invoice() performs explicit field comparison and returns {verified, mismatches}.
"""

from __future__ import annotations

import json
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "internal_db.json"


def _load() -> dict:
    if DB_PATH.exists():
        try:
            return json.loads(DB_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"invoices": [], "next_id": 9843}


def _save(db: dict) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    DB_PATH.write_text(json.dumps(db, indent=2), encoding="utf-8")


def search_invoice(
    vendor: str | None = None,
    month: int | None = None,
    year: int | None = None,
    invoice_number: str | None = None,
) -> dict:
    db = _load()
    rows = db["invoices"]

    def keep(r: dict) -> bool:
        if vendor and vendor.lower() not in str(r.get("vendor", "")).lower():
            return False
        if invoice_number and str(r.get("invoice_number", "")).lower() != str(invoice_number).lower():
            return False
        if month or year:
            dd = str(r.get("due_date", ""))  # YYYY-MM-DD
            parts = dd.split("-")
            if len(parts) != 3:
                return False
            try:
                y, m = int(parts[0]), int(parts[1])
            except ValueError:
                return False
            if year and y != int(year):
                return False
            if month and m != int(month):
                return False
        return True

    rows = [r for r in rows if keep(r)]
    return {"count": len(rows), "invoices": rows}


def _find_duplicate(db: dict, vendor: str, amount: float, due_date: str, invoice_number: str | None) -> dict | None:
    for inv in db["invoices"]:
        if invoice_number and inv.get("invoice_number") and inv["invoice_number"] == invoice_number:
            return inv
        if (
            str(inv.get("vendor", "")).lower() == str(vendor).lower()
            and float(inv.get("amount", 0)) == float(amount)
            and inv.get("due_date") == due_date
        ):
            return inv
    return None


def submit_invoice(
    vendor: str,
    amount: float,
    due_date: str,
    invoice_number: str | None = None,
) -> dict:
    db = _load()

    dup = _find_duplicate(db, vendor, amount, due_date, invoice_number)
    if dup:
        return {
            "status": "REJECTED_DUPLICATE",
            "reason": (
                "An invoice with the same vendor + amount + due_date "
                "(or the same invoice_number) already exists."
            ),
            "existing_invoice_id": dup["invoice_id"],
            "existing": dup,
        }

    invoice_id = f"INV-{db['next_id']}"
    db["next_id"] += 1

    record = {
        "invoice_id": invoice_id,
        "vendor": vendor,
        "amount": float(amount),
        "due_date": due_date,
        "status": "SUCCESS",
    }
    if invoice_number:
        record["invoice_number"] = invoice_number
    db["invoices"].append(record)
    _save(db)
    return {"status": "SUCCESS", "invoice_id": invoice_id}


def verify_invoice(
    invoice_id: str,
    expected_vendor: str | None = None,
    expected_amount: float | None = None,
    expected_due_date: str | None = None,
) -> dict:
    db = _load()
    for inv in db["invoices"]:
        if inv["invoice_id"] != invoice_id:
            continue

        mismatches: list[dict] = []
        if expected_vendor is not None and str(inv.get("vendor", "")).lower() != str(expected_vendor).lower():
            mismatches.append({"field": "vendor", "expected": expected_vendor, "stored": inv.get("vendor")})
        if expected_amount is not None:
            try:
                if float(inv.get("amount", 0)) != float(expected_amount):
                    mismatches.append({"field": "amount", "expected": float(expected_amount), "stored": inv.get("amount")})
            except (TypeError, ValueError):
                mismatches.append({"field": "amount", "expected": expected_amount, "stored": inv.get("amount")})
        if expected_due_date is not None and inv.get("due_date") != expected_due_date:
            mismatches.append({"field": "due_date", "expected": expected_due_date, "stored": inv.get("due_date")})

        return {
            "verified": not mismatches,
            "invoice_id": invoice_id,
            "mismatches": mismatches,
            "stored": inv,
        }

    return {"verified": False, "error": f"Invoice {invoice_id} not found"}