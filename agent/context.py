"""Reference resolution across turns.

Given the current session history and a new user task, rewrite the task
into a fully self-contained request — replacing "it", "that", "the same",
"its", etc. with the specific entities they refer to.

This runs BEFORE classification so the classifier and the tool loop always
see a self-contained task, and don't need to know anything about history.
"""

from __future__ import annotations

import re

from agent.planner import REFERENCE_RESOLVER_SYSTEM_PROMPT

_THINK_RE = re.compile(r" thinking.*?", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^```(?:.*?)?\s*|\s*```$", re.DOTALL)


def _format_history(session, k: int = 5) -> str:
    if not session.turns:
        return "(no prior conversation)"
    lines: list[str] = []
    for t in session.turns[-k:]:
        lines.append(f"User: {t.original_task}")
        if t.resolved_task and t.resolved_task != t.original_task:
            lines.append(f"(interpreted as: {t.resolved_task})")
        if t.summary:
            lines.append(f"Assistant: {t.summary[:600]}")
    return "\n".join(lines)


def _clean(text: str) -> str:
    text = _THINK_RE.sub("", text or "").strip()
    text = _FENCE_RE.sub("", text).strip()
    # Strip common prefixes the model might prepend.
    text = re.sub(
        r"^(rewritten(\s+task)?|resolved(\s+task)?|self-contained(\s+task)?)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    # Strip wrapping quotes.
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    return text


def resolve_references(llm_chat, session, task: str) -> str:
    """Return a self-contained rewrite of `task` using session history.

    If the session is empty, the task is returned unchanged.
    If the resolver fails or returns something unreasonable, the original task
    is returned unchanged (safe fallback).
    """
    if not session.turns:
        return task

    history = _format_history(session)
    user_block = (
        f"Conversation history:\n{history}\n\n"
        f"New user message: {task}\n\n"
        f"Rewritten (self-contained):"
    )
    messages = [
        {"role": "system", "content": REFERENCE_RESOLVER_SYSTEM_PROMPT},
        {"role": "user", "content": user_block},
    ]
    try:
        msg = llm_chat(messages, tools=None, force_json=False)
    except Exception:
        return task

    rewritten = _clean(msg.get("content") or "")
    if not rewritten:
        return task
    # Sanity check: a reasonable rewrite should be a short task, not an essay.
    if len(rewritten) > 500:
        return task
    # Guard against the model returning a full answer instead of a rewrite.
    if rewritten.lower().startswith(("the answer", "here is", "sure,", "certainly,")):
        return task
    return rewritten