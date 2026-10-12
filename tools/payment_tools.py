"""Invoice payment operations. Writes to data/paid_invoices.json.

This database is SEPARATE from internal_db.json (the invoice *processing*
record). Paying an invoice is a distinct action from recording/submitting it,
and the two must not be conflated:

    internal_db.json   ← submit_invoice   (WRITE goals)
    paid_invoices.json ← pay_invoice      (PAY goals)

Business invariants enforced here:
- pay_invoice() requires a valid approval_id from approvals_log.json
  with approved=true.
- pay_invoice() rejects duplicates by invoice_number (or, if no invoice_number
  is given, by vendor + amount + due_date).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "paid_invoices.json"
APPROVALS_PATH = Path(__file__).resolve().parent.parent / "data" / "approvals_log.json"


def _load() -> dict:
    if DB_PATH.exists():
        try:
            return json.loads(DB_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"payments": [], "next_id": 1}


def _save(db: dict) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    DB_PATH.write_text(json.dumps(db, indent=2, default=str), encoding="utf-8")


def _approval_was_approved(approval_id: str) -> bool:
    if not approval_id or not APPROVALS_PATH.exists():
        return False
    try:
        log = json.loads(APPROVALS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    for a in log.get("approvals", []):
        if a.get("approval_id") == approval_id and a.get("approved") is True:
            return True
    return False


def search_payment(
    vendor: str | None = None,
    invoice_number: str | None = None,
    month: int | None = None,
    year: int | None = None,
) -> dict:
    """Search recorded payments in paid_invoices.json.

    Use this BEFORE paying to detect duplicates. A non-empty result for a
    given invoice_number means the invoice was already paid.
    """
    db = _load()
    rows = db["payments"]

    def keep(r: dict) -> bool:
        if vendor and vendor.lower() not in str(r.get("vendor", "")).lower():
            return False
        if invoice_number and str(r.get("invoice_number", "")).lower() != str(invoice_number).lower():
            return False
        if month or year:
            pd = str(r.get("paid_at", ""))  # ISO timestamp
            parts = pd.split("-")
            if len(parts) >= 2:
                try:
                    y, m = int(parts[0]), int(parts[1])
                    if year and y != int(year):
                        return False
                    if month and m != int(month):
                        return False
                except ValueError:
                    return False
            else:
                return False
        return True

    rows = [r for r in rows if keep(r)]
    total = sum(float(r.get("amount", 0)) for r in rows)
    return {
        "count": len(rows),
        "total_paid": round(total, 2),
        "payments": rows,
    }


def pay_invoice(
    vendor: str,
    amount: float,
    due_date: str,
    invoice_number: str | None = None,
    approval_id: str | None = None,
    currency: str = "USD",
    notes: str | None = None,
) -> dict:
    """Record an invoice payment. Rejects duplicates by invoice_number."""
    if not vendor:
        return {"status": "ERROR", "reason": "vendor is required"}
    try:
        amt = float(amount)
    except (TypeError, ValueError):
        return {"status": "ERROR", "reason": "amount must be a number"}
    if amt <= 0:
        return {"status": "ERROR", "reason": "amount must be > 0"}
    if not due_date:
        return {"status": "ERROR", "reason": "due_date is required"}

    if not approval_id:
        return {
            "status": "REQUIRES_APPROVAL",
            "reason": (
                "pay_invoice requires prior human approval. "
                "Call request_human_approval first, then pass its approval_id."
            ),
        }
    if not _approval_was_approved(approval_id):
        return {
            "status": "APPROVAL_NOT_VALID",
            "reason": f"approval_id {approval_id!r} was not approved or not found in the log",
        }

    db = _load()

    # Primary duplicate check: invoice_number.
    if invoice_number:
        for existing in db["payments"]:
            if str(existing.get("invoice_number", "")).lower() == str(invoice_number).lower():
                return {
                    "status": "REJECTED_DUPLICATE",
                    "reason": (
                        f"Invoice {invoice_number} has already been paid "
                        f"(payment_id {existing.get('payment_id')}). "
                        f"Duplicate payments are not allowed."
                    ),
                    "existing_payment_id": existing.get("payment_id"),
                    "existing": existing,
                }
    else:
        # Secondary duplicate check: vendor + amount + due_date.
        for existing in db["payments"]:
            if (
                str(existing.get("vendor", "")).lower() == vendor.lower()
                and float(existing.get("amount", 0)) == amt
                and existing.get("due_date") == due_date
            ):
                return {
                    "status": "REJECTED_DUPLICATE",
                    "reason": (
                        "A matching payment (same vendor + amount + due_date) "
                        "already exists. Duplicate payments are not allowed."
                    ),
                    "existing_payment_id": existing.get("payment_id"),
                    "existing": existing,
                }

    payment_id = f"PMT-{db['next_id']:04d}"
    db["next_id"] += 1

    record = {
        "payment_id": payment_id,
        "vendor": vendor,
        "amount": amt,
        "currency": currency,
        "due_date": due_date,
        "invoice_number": invoice_number or "",
        "approval_id": approval_id,
        "status": "SUCCESS",
        "paid_at": datetime.now().isoformat(timespec="seconds"),
        "notes": notes or "",
    }
    db["payments"].append(record)
    _save(db)

    return {
        "status": "SUCCESS",
        "payment_id": payment_id,
        "vendor": vendor,
        "amount": amt,
        "currency": currency,
        "due_date": due_date,
        "invoice_number": invoice_number or "",
        "approval_id": approval_id,
        "paid_at": record["paid_at"],
    }


def verify_payment(
    payment_id: str,
    expected_vendor: str | None = None,
    expected_amount: float | None = None,
    expected_due_date: str | None = None,
) -> dict:
    """Fetch a stored payment and compare it to expected values.

    Returns {verified: bool, mismatches: [...], stored: {...}}.
    """
    db = _load()
    for p in db["payments"]:
        if p["payment_id"] != payment_id:
            continue

        mismatches: list[dict] = []
        if expected_vendor is not None and str(p.get("vendor", "")).lower() != str(expected_vendor).lower():
            mismatches.append({"field": "vendor", "expected": expected_vendor, "stored": p.get("vendor")})
        if expected_amount is not None:
            try:
                if float(p.get("amount", 0)) != float(expected_amount):
                    mismatches.append({
                        "field": "amount",
                        "expected": float(expected_amount),
                        "stored": p.get("amount"),
                    })
            except (TypeError, ValueError):
                mismatches.append({"field": "amount", "expected": expected_amount, "stored": p.get("amount")})
        if expected_due_date is not None and p.get("due_date") != expected_due_date:
            mismatches.append({"field": "due_date", "expected": expected_due_date, "stored": p.get("due_date")})

        return {
            "verified": not mismatches,
            "payment_id": payment_id,
            "mismatches": mismatches,
            "stored": p,
        }

    return {"verified": False, "error": f"Payment {payment_id} not found"}