"""Human-in-the-loop approval gate.

request_human_approval() blocks on stdin, prints a clear prompt, records the
decision in data/approvals_log.json, and returns a structured response.

This is synchronous and blocking: the tool call does not return until the
human answers y/n. That is the essence of a HITL gate — the agent cannot
proceed past it without a decision.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "approvals_log.json"


def _load_log() -> dict:
    if LOG_PATH.exists():
        try:
            return json.loads(LOG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"approvals": [], "next_id": 1}


def _save_log(db: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(db, indent=2, default=str), encoding="utf-8")


def request_human_approval(action: str, summary: str, details: dict | None = None) -> dict:
    """Block until a human operator approves or rejects the proposed action.

    Args:
        action:  short machine-readable action label, e.g. "BUY_STOCK",
                 "PAY_INVOICE", "SUBMIT_INVOICE".
        summary: one-sentence description of what the agent intends to do.
        details: optional structured details (symbol, quantity, amount, etc.).

    Returns:
        {
          "approved": bool,
          "approval_id": "APV-0001",
          "action": ..., "summary": ..., "details": {...},
          "decided_at": "2026-10-09T12:34:56",
          "note": "..." (if rejected)
        }
    """
    details = details or {}

    # Build a clear, scannable prompt for the human.
    print()
    print("+" + "-" * 68 + "+")
    print(f"|  HUMAN APPROVAL REQUIRED - action: {action}")
    print("+" + "-" * 68 + "+")
    print(f"|  {summary}")
    if details:
        for k, v in details.items():
            print(f"|    {k}: {v}")
    print("+" + "-" * 68 + "+")

    try:
        raw = input("Approve? (y = yes, n = no): ")
    except (EOFError, KeyboardInterrupt):
        raw = ""

    answer = (raw or "").strip().lower()
    approved = answer in {"y", "yes", "approve", "approved", "ok", "okay", "go"}

    db = _load_log()
    approval_id = f"APV-{db['next_id']:04d}"
    db["next_id"] += 1

    record = {
        "approval_id": approval_id,
        "action": action,
        "summary": summary,
        "details": details,
        "approved": approved,
        "raw_answer": answer,
        "decided_at": datetime.now().isoformat(timespec="seconds"),
    }
    db["approvals"].append(record)
    _save_log(db)

    note = "" if approved else "Human rejected the proposed action."
    return {
        "approved": approved,
        "approval_id": approval_id,
        "action": action,
        "summary": summary,
        "details": details,
        "decided_at": record["decided_at"],
        "note": note,
    }