"""Verification helper. Compares a stored invoice record to expected values."""


def verify_record(stored: dict, expected: dict) -> dict:
    mismatches = []
    for key in ("amount", "due_date"):
        if key in expected:
            if str(stored.get(key)) != str(expected[key]):
                mismatches.append(key)
    return {
        "ok": not mismatches,
        "mismatches": mismatches,
        "stored": stored,
        "expected": expected,
    }