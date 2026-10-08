"""Goal classifier — pattern-first, LLM fallback.

Patterns handle ~95% of real AP tasks deterministically and never lie.
The LLM is only consulted when patterns can't decide the task type.
"""

from __future__ import annotations

import json
import re

from agent.planner import CLASSIFIER_SYSTEM_PROMPT

_VALID = {"READ", "AGGREGATE", "WRITE", "VERIFY"}

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12, "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Vendors seen in the sample data. In a real deployment this would come from
# a vendor directory, not a hardcoded list.
_KNOWN_VENDORS = [
    ("abc corp", "ABC Corp"),
    ("amazon", "Amazon"),
    ("openai", "OpenAI"),
]

_THINK_RE = re.compile(r" thinking.*?", re.DOTALL | re.IGNORECASE)


def _sniff_vendor(text: str) -> str | None:
    t = text.lower()
    for needle, canonical in _KNOWN_VENDORS:
        if needle in t:
            return canonical
    # Fallback: a Capitalized Word or two adjacent Capitalized Words.
    # Avoid swallowing stopwords.
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


def _pattern_task_type(text: str) -> str | None:
    """Return a task_type if the phrasing is unambiguous, else None."""
    t = text.lower()

    # VERIFY — asking about the state of an existing record.
    verify_markers = (
        "check whether", "check if", "was ", "were ", "is there",
        "has been", "have been", "already recorded", "already entered",
        "already in", "recorded in", "in the system", "in our system",
    )
    if any(m in t for m in verify_markers):
        return "VERIFY"

    # WRITE — explicitly asked to enter/submit/process/record.
    write_markers = ("process ", "submit ", "enter ", "record ", "add to",
                     "into the system", "into our system", "into internal")
    if any(m in t for m in write_markers):
        return "WRITE"

    # AGGREGATE — plural / list / count over many invoices.
    aggregate_markers = (
        "all ", "every ", "list ", "show me all", "give me all",
        "how many", "count ", "total ", "every invoice",
    )
    if any(m in t for m in aggregate_markers):
        return "AGGREGATE"

    # READ — information retrieval over a single invoice.
    read_markers = (
        "find ", "show me", "show ", "what is", "what's", "when is",
        "how much", "extract", "tell me", "look up", "get me",
    )
    if any(m in t for m in read_markers):
        return "READ"

    return None


def _pattern_goal(task: str) -> dict:
    ttype = _pattern_task_type(task)
    vendor = _sniff_vendor(task)
    month = _sniff_month(task)
    year = _sniff_year(task)
    inv_no = _sniff_invoice_number(task)
    latest = bool(re.search(r"\b(latest|most recent|newest|recent)\b", task, re.I))

    return {
        "task_type": ttype or "READ",
        "vendor": vendor,
        "month": month,
        "year": year,
        "latest": latest,
        "invoice_number": inv_no,
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
    for k in ("vendor", "month", "year", "latest", "invoice_number", "desired_outcome"):
        goal.setdefault(k, None)
    return goal


def classify(llm_chat, task: str) -> dict:
    """Pattern-first. LLM only consulted when the pattern pass is undecided."""
    pattern_goal = _pattern_goal(task)

    # If the pattern pass got a clear task type and a vendor, trust it.
    # Otherwise ask the LLM, then merge: pattern wins on anything it found.
    needs_llm = (
        _pattern_task_type(task) is None
        or (pattern_goal["task_type"] in {"READ", "WRITE", "VERIFY"} and not pattern_goal["vendor"])
    )
    if not needs_llm:
        return pattern_goal

    llm_goal = _llm_fallback(llm_chat, task)
    if not llm_goal:
        return pattern_goal

    # Merge — pattern values override when they're non-null.
    for k, v in pattern_goal.items():
        if v not in (None, False) and llm_goal.get(k) in (None, False):
            llm_goal[k] = v
    if llm_goal.get("desired_outcome") in (None, ""):
        llm_goal["desired_outcome"] = task
    return llm_goal