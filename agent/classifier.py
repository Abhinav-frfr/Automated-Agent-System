"""Goal classifier — pattern-first, LLM fallback.

Adds PAY detection (invoice payment — distinct from WRITE). Existing invoice,
BROWSE, and TRADE behavior is unchanged.
"""

from __future__ import annotations

import json
import re

from agent.planner import CLASSIFIER_SYSTEM_PROMPT

_VALID = {"READ", "AGGREGATE", "WRITE", "VERIFY", "BROWSE", "TRADE", "PAY"}

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12, "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

_KNOWN_VENDORS = [
    ("abc corp", "ABC Corp"),
    ("amazon", "Amazon"),
    ("openai", "OpenAI"),
]

_KNOWN_COMPANY_ALIASES = [
    ("reliance industries", "Reliance Industries"),
    ("reliance", "Reliance Industries"),
    ("tata consultancy services", "TCS"),
    ("tata consultancy", "TCS"),
    ("tcs", "TCS"),
    ("infosys", "Infosys"),
    ("hdfc bank", "HDFC Bank"),
    ("hdfc", "HDFC Bank"),
    ("icici bank", "ICICI Bank"),
    ("icici", "ICICI Bank"),
    ("wipro", "Wipro"),
    ("state bank of india", "State Bank of India"),
    ("sbi", "State Bank of India"),
    ("bharti airtel", "Bharti Airtel"),
    ("airtel", "Bharti Airtel"),
    ("itc", "ITC"),
    ("larsen & toubro", "Larsen & Toubro"),
    ("larsen and toubro", "Larsen & Toubro"),
    ("l&t", "Larsen & Toubro"),
    ("asian paints", "Asian Paints"),
    ("kotak mahindra", "Kotak Mahindra Bank"),
    ("kotak", "Kotak Mahindra Bank"),
    ("axis bank", "Axis Bank"),
    ("maruti suzuki", "Maruti Suzuki"),
    ("maruti", "Maruti Suzuki"),
    ("bajaj finance", "Bajaj Finance"),
    ("sun pharma", "Sun Pharma"),
    ("titan", "Titan Company"),
    ("apple", "Apple"),
    ("microsoft", "Microsoft"),
    ("alphabet", "Alphabet"),
    ("google", "Alphabet"),
    ("amazon", "Amazon"),
    ("tesla", "Tesla"),
    ("nvidia", "NVIDIA"),
    ("meta", "Meta"),
    ("facebook", "Meta"),
    ("netflix", "Netflix"),
    ("zomato", "Zomato"),
]

_KNOWN_EXCHANGES = {"NSE", "BSE", "NASDAQ", "NYSE", "AMEX"}

_BROWSE_MARKERS = (
    "stock", "share price", "ticker", "market price", "quote",
    "website", "web page", "browse", "navigate to", "open the site",
    "open the website", "open the page", "google finance",
)

_TRADE_MARKERS = ("buy ", "purchase ", "sell ", "invest in ")

# Approval wording for WRITE goals. PAY goals always require approval,
# independent of these markers.
_APPROVAL_MARKERS = (
    "with approval", "get approval", "request approval",
    "confirm before", "with confirmation", "get confirmation",
    "ask before", "need approval",
)

# Verb "pay" as a whole word (not "repay" / "payment gateway" style tokens).
_PAY_RE = re.compile(r"\b(pay|paying|settle|settling)\b", re.IGNORECASE)

_THINK_RE = re.compile(r" thinking.*?", re.DOTALL | re.IGNORECASE)


def _sniff_vendor(text: str) -> str | None:
    t = text.lower()
    for needle, canonical in _KNOWN_VENDORS:
        if needle in t:
            return canonical
    stop = {
        "find", "show", "give", "process", "check", "enter", "submit",
        "list", "all", "every", "what", "which", "how", "many", "is", "was",
        "were", "invoice", "invoices", "latest", "most", "recent", "newest",
        "amount", "due", "date", "please", "me", "get", "the",
    }
    for m in re.finditer(r"\b[A-Z][A-Za-z&]+(?:\s+[A-Z][A-Za-z&]+){0,2}\b", text):
        cand = m.group(0).strip()
        head = cand.split()[0].lower()
        if head not in stop and len(cand) >= 3:
            return cand
    return None


def _sniff_month(text: str) -> int | None:
    t = text.lower()
    for name, num in _MONTHS.items():
        if re.search(rf"\b{name}\b", t):
            return num
    return None


def _sniff_year(text: str) -> int | None:
    m = re.search(r"\b(20\d{2})\b", text)
    return int(m.group(1)) if m else None


def _sniff_invoice_number(text: str) -> str | None:
    m = re.search(r"\b([A-Z]{2,5}-\d{4}-\d{3,4})\b", text)
    return m.group(1) if m else None


def _sniff_companies(text: str) -> list[str]:
    t = text.lower()
    found: list[str] = []
    occupied: list[tuple[int, int]] = []
    for alias, canonical in _KNOWN_COMPANY_ALIASES:
        for m in re.finditer(rf"\b{re.escape(alias)}\b", t):
            span = m.span()
            if any(s <= span[0] < e or s < span[1] <= e for s, e in occupied):
                continue
            occupied.append(span)
            if canonical not in found:
                found.append(canonical)
    return found


def _sniff_exchange(text: str) -> str | None:
    for m in re.finditer(r"\b([A-Za-z]{2,6})\b", text):
        ex = m.group(1).upper()
        if ex in _KNOWN_EXCHANGES:
            return ex
    return None


def _sniff_quantity(text: str) -> float | None:
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:shares?|units?|stocks?)\b", text, re.I)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _sniff_side(text: str) -> str | None:
    t = text.lower()
    if "sell " in t or "sell shares" in t:
        return "SELL"
    if any(m in t for m in ("buy ", "purchase ", "invest in ")):
        return "BUY"
    return None


def _sniff_requires_approval(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _APPROVAL_MARKERS)


def _pattern_task_type(text: str) -> str | None:
    t = text.lower()
    has_invoice = "invoice" in t or "invoices" in t

    # TRADE before BROWSE: "buy Apple stock" must not become BROWSE.
    if not has_invoice and any(m in t for m in _TRADE_MARKERS):
        return "TRADE"

    if not has_invoice:
        if _sniff_companies(text) or any(m in t for m in _BROWSE_MARKERS):
            return "BROWSE"
        return None

    # ----- Invoice tasks: VERIFY → PAY → WRITE → AGGREGATE → READ -----
    verify_markers = (
        "check whether", "check if", "was ", "were ", "is there",
        "has been", "have been", "already recorded", "already entered",
        "already in", "recorded in", "in the system", "in our system",
    )
    if any(m in t for m in verify_markers):
        return "VERIFY"

    # PAY is distinct from WRITE. Word-boundary "pay" / "settle".
    if _PAY_RE.search(t):
        return "PAY"

    write_markers = (
        "process ", "submit ", "enter ", "record ", "add to",
        "into the system", "into our system", "into internal",
    )
    if any(m in t for m in write_markers):
        return "WRITE"

    aggregate_markers = (
        "all ", "every ", "list ", "show me all", "give me all",
        "how many", "count ", "total ", "every invoice",
    )
    if any(m in t for m in aggregate_markers):
        return "AGGREGATE"

    read_markers = (
        "find ", "show me", "show ", "what is", "what's", "when is",
        "how much", "extract", "tell me", "look up", "get me",
    )
    if any(m in t for m in read_markers):
        return "READ"

    return None


def _pattern_goal(task: str) -> dict:
    ttype = _pattern_task_type(task)
    is_browse = ttype == "BROWSE"
    is_trade = ttype == "TRADE"
    is_pay = ttype == "PAY"

    requires_approval = False
    if is_trade:
        # Buying/selling stock always requires HITL.
        requires_approval = True
    elif is_pay:
        # Paying an invoice always requires HITL.
        requires_approval = True
    elif ttype == "WRITE":
        # WRITE only requires approval if the wording asks for it.
        requires_approval = _sniff_requires_approval(task)

    return {
        "task_type": ttype or "READ",
        "vendor": None if (is_browse or is_trade) else _sniff_vendor(task),
        "companies": (_sniff_companies(task) if (is_browse or is_trade) else None),
        "exchange": _sniff_exchange(task) if (is_browse or is_trade) else None,
        "quantity": _sniff_quantity(task) if is_trade else None,
        "side": _sniff_side(task) if is_trade else None,
        "requires_approval": requires_approval,
        "month": _sniff_month(task),
        "year": _sniff_year(task),
        "latest": bool(re.search(r"\b(latest|most recent|newest|recent)\b", task, re.I)),
        "invoice_number": None if (is_browse or is_trade) else _sniff_invoice_number(task),
        "desired_outcome": task,
    }


def _llm_fallback(llm_chat, task: str) -> dict | None:
    messages = [
        {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    try:
        msg = llm_chat(messages, tools=None, force_json=True)
    except Exception:
        return None
    content = _THINK_RE.sub("", msg.get("content") or "").strip()
    content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.MULTILINE).strip()
    try:
        goal = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(goal, dict):
        return None
    if goal.get("task_type") not in _VALID:
        return None
    for k in ("vendor", "companies", "exchange", "quantity", "side",
              "requires_approval", "month", "year", "latest",
              "invoice_number", "desired_outcome"):
        goal.setdefault(k, None)
    return goal


def classify(llm_chat, task: str) -> dict:
    pattern_goal = _pattern_goal(task)

    needs_llm = (
        _pattern_task_type(task) is None
        or (
            pattern_goal["task_type"] in {"READ", "WRITE", "VERIFY", "PAY"}
            and not pattern_goal["vendor"]
        )
        or (
            pattern_goal["task_type"] in {"BROWSE", "TRADE"}
            and not pattern_goal["companies"]
        )
    )
    if not needs_llm:
        return pattern_goal

    llm_goal = _llm_fallback(llm_chat, task)
    if not llm_goal:
        return pattern_goal

    for k, v in pattern_goal.items():
        if v in (None, False, [], ""):
            continue
        if llm_goal.get(k) in (None, False, [], ""):
            llm_goal[k] = v
    if llm_goal.get("desired_outcome") in (None, ""):
        llm_goal["desired_outcome"] = task
    return llm_goal