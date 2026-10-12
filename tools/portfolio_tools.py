"""Stock trading (buy / list). Writes to data/purchased_stock.json.

buy_stock() enforces a business invariant at the tool level: the caller must
supply an approval_id that exists in data/approvals_log.json with
approved=true. The agent's policy guard also enforces this — but we enforce
it here too, so the tool cannot be bypassed.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "purchased_stock.json"
APPROVALS_PATH = Path(__file__).resolve().parent.parent / "data" / "approvals_log.json"


def _load() -> dict:
    if DB_PATH.exists():
        try:
            return json.loads(DB_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"purchases": [], "next_id": 1}


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


def buy_stock(
    symbol: str,
    quantity: float,
    price: float,
    currency: str = "USD",
    approval_id: str | None = None,
    notes: str | None = None,
) -> dict:
    """Record a stock purchase.

    symbol: e.g. "AAPL", "AAPL:NASDAQ", "RELIANCE:NSE".
    quantity: number of shares.
    price: per-share price at execution.
    approval_id: the APV-xxxx returned by request_human_approval.
    """
    if not symbol:
        return {"status": "ERROR", "reason": "symbol is required"}
    try:
        q = float(quantity)
        p = float(price)
    except (TypeError, ValueError):
        return {"status": "ERROR", "reason": "quantity and price must be numbers"}
    if q <= 0:
        return {"status": "ERROR", "reason": "quantity must be > 0"}
    if p <= 0:
        return {"status": "ERROR", "reason": "price must be > 0"}

    if not approval_id:
        return {
            "status": "REQUIRES_APPROVAL",
            "reason": (
                "buy_stock requires prior human approval. "
                "Call request_human_approval first, then pass its approval_id."
            ),
        }

    if not _approval_was_approved(approval_id):
        return {
            "status": "APPROVAL_NOT_VALID",
            "reason": f"approval_id {approval_id!r} was not approved or not found in the log",
        }

    db = _load()
    total = round(q * p, 2)
    purchase_id = f"PUR-{db['next_id']:04d}"
    db["next_id"] += 1

    record = {
        "purchase_id": purchase_id,
        "symbol": symbol,
        "quantity": q,
        "price": p,
        "currency": currency,
        "total": total,
        "approval_id": approval_id,
        "status": "SUCCESS",
        "executed_at": datetime.now().isoformat(timespec="seconds"),
        "notes": notes or "",
    }
    db["purchases"].append(record)
    _save(db)

    return {
        "status": "SUCCESS",
        "purchase_id": purchase_id,
        "symbol": symbol,
        "quantity": q,
        "price": p,
        "currency": currency,
        "total": total,
        "approval_id": approval_id,
        "executed_at": record["executed_at"],
    }


def list_purchases(symbol: str | None = None) -> dict:
    """Return recorded purchases (optionally filtered by symbol)."""
    db = _load()
    rows = db["purchases"]
    if symbol:
        s = symbol.upper()
        rows = [r for r in rows if s in str(r.get("symbol", "")).upper()]
    total_value = sum(float(r.get("total", 0)) for r in rows)
    return {
        "count": len(rows),
        "total_value": round(total_value, 2),
        "purchases": rows,
    }