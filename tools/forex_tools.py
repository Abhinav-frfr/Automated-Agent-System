"""Exchange rate lookup via a free public API.

Public tools:
    get_exchange_rate(from_currency, to_currency, amount=1.0)
    list_exchange_rates(limit=20)

Storage:
    Every successful (and every failed) lookup is appended to
    data/exchange_rates.json, alongside a monotonically increasing id.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "exchange_rates.json"

# Common ISO 4217 codes. Used only as a sanity hint — the tool itself
# forwards whatever the caller passes to the API.
_KNOWN_CODES = {
    "USD", "EUR", "GBP", "JPY", "INR", "AUD", "CAD", "CHF", "CNY", "HKD",
    "SGD", "NZD", "SEK", "NOK", "DKK", "PLN", "MXN", "BRL", "ZAR", "TRY",
    "RUB", "KRW", "THB", "MYR", "IDR", "PHP", "VND", "AED", "SAR", "ILS",
    "CZK", "HUF", "RON", "BGN", "HRK", "ISK",
}

_PRIMARY_TIMEOUT_S = 10


# ---------------------------------------------------------------- #
# Persistence
# ---------------------------------------------------------------- #
def _load_db() -> dict:
    if DB_PATH.exists():
        try:
            return json.loads(DB_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"queries": [], "next_id": 1}


def _save_db(db: dict) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    DB_PATH.write_text(json.dumps(db, indent=2, default=str), encoding="utf-8")


def _record(entry: dict) -> str:
    db = _load_db()
    entry_id = f"FX-{db['next_id']:05d}"
    db["next_id"] += 1
    entry["id"] = entry_id
    entry.setdefault("fetched_at", datetime.now().isoformat(timespec="seconds"))
    db["queries"].append(entry)
    _save_db(db)
    return entry_id


# ---------------------------------------------------------------- #
# API calls
# ---------------------------------------------------------------- #
def _fetch_frankfurter(frm: str, to: str) -> dict:
    """https://www.frankfurter.app/ — ECB reference rates, no key required."""
    url = f"https://api.frankfurter.app/latest?from={urllib.parse.quote(frm)}&to={urllib.parse.quote(to)}"
    req = urllib.request.Request(url, headers={"User-Agent": "AgenticWorker/1.0"})
    with urllib.request.urlopen(req, timeout=_PRIMARY_TIMEOUT_S) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    rates = payload.get("rates") or {}
    if to not in rates:
        raise ValueError(f"frankfurter returned no rate for {to}")
    return {
        "rate": float(rates[to]),
        "base": payload.get("base", frm),
        "date": payload.get("date"),
        "source": "frankfurter.app",
    }


def _fetch_open_er(frm: str, to: str) -> dict:
    """https://open.er-api.com/ — free tier, no key required."""
    url = f"https://open.er-api.com/v6/latest/{urllib.parse.quote(frm)}"
    req = urllib.request.Request(url, headers={"User-Agent": "AgenticWorker/1.0"})
    with urllib.request.urlopen(req, timeout=_PRIMARY_TIMEOUT_S) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if payload.get("result") != "success":
        raise ValueError(f"open.er-api.com returned result={payload.get('result')!r}")
    rates = payload.get("rates") or {}
    if to not in rates:
        raise ValueError(f"open.er-api.com returned no rate for {to}")
    return {
        "rate": float(rates[to]),
        "base": payload.get("base_code", frm),
        "date": payload.get("time_last_update_utc", ""),
        "source": "open.er-api.com",
    }


# ---------------------------------------------------------------- #
# Public tools
# ---------------------------------------------------------------- #
def get_exchange_rate(
    from_currency: str,
    to_currency: str,
    amount: float = 1.0,
) -> dict:
    """Fetch the current exchange rate between two currencies.

    Args:
        from_currency: ISO 4217 code, e.g. "USD", "INR", "EUR".
        to_currency:   ISO 4217 code.
        amount:        Amount to convert (default 1).

    Returns a dict with the rate, converted amount, and source.
    Every call (success or failure) is stored in data/exchange_rates.json.
    """
    frm = (from_currency or "").strip().upper()
    to = (to_currency or "").strip().upper()

    if not frm or not to:
        result = {
            "status": "ERROR",
            "error": "from_currency and to_currency are required",
            "from": frm,
            "to": to,
        }
        _record(result)
        return result

    try:
        amt = float(amount)
    except (TypeError, ValueError):
        result = {
            "status": "ERROR",
            "error": f"amount must be a number, got {amount!r}",
            "from": frm,
            "to": to,
        }
        _record(result)
        return result

    if amt <= 0:
        result = {
            "status": "ERROR",
            "error": "amount must be > 0",
            "from": frm,
            "to": to,
        }
        _record(result)
        return result

    # Same currency — trivial.
    if frm == to:
        result = {
            "status": "SUCCESS",
            "from": frm,
            "to": to,
            "rate": 1.0,
            "amount": amt,
            "converted": round(amt, 6),
            "source": "identity",
            "date": datetime.now().date().isoformat(),
        }
        _record(result)
        return result

    # Try the primary then fall back to the secondary API.
    errors: list[str] = []
    payload: dict | None = None
    for fetcher in (_fetch_frankfurter, _fetch_open_er):
        try:
            payload = fetcher(frm, to)
            break
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError, TimeoutError) as e:
            errors.append(f"{fetcher.__name__}: {type(e).__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{fetcher.__name__}: {type(e).__name__}: {e}")

    if payload is None:
        result = {
            "status": "ERROR",
            "error": "All exchange-rate providers failed.",
            "details": errors,
            "from": frm,
            "to": to,
        }
        _record(result)
        return result

    rate = payload["rate"]
    converted = round(amt * rate, 6)

    result = {
        "status": "SUCCESS",
        "from": frm,
        "to": to,
        "rate": rate,
        "amount": amt,
        "converted": converted,
        "source": payload["source"],
        "date": payload.get("date") or "",
    }
    result["id"] = _record(result)
    return result


def list_exchange_rates(limit: int = 20) -> dict:
    """Return the most recent stored exchange-rate lookups."""
    db = _load_db()
    rows = db.get("queries", [])
    try:
        n = max(1, int(limit))
    except (TypeError, ValueError):
        n = 20
    return {
        "count": len(rows),
        "showing": min(n, len(rows)),
        "queries": rows[-n:],
    }