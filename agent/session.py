"""Conversation session with persistent turn history.

A session stores every completed turn:
    (original_task, resolved_task, goal, status, summary, evidence)

Sessions persist to data/sessions/<session_id>.json so conversations
survive restarts. Multiple named sessions are supported.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

SESSIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "sessions"


@dataclass
class Turn:
    turn: int
    original_task: str
    resolved_task: str
    goal: dict
    status: str
    summary: str
    evidence: dict | None
    timestamp: str


@dataclass
class Session:
    session_id: str = "default"
    created_at: str = ""
    updated_at: str = ""
    turns: list[Turn] = field(default_factory=list)

    # ---------- persistence ---------- #
    @classmethod
    def load(cls, session_id: str = "default") -> "Session":
        path = SESSIONS_DIR / f"{session_id}.json"
        now = datetime.now().isoformat(timespec="seconds")
        if not path.exists():
            return cls(session_id=session_id, created_at=now, updated_at=now)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls(session_id=session_id, created_at=now, updated_at=now)
        return cls(
            session_id=data.get("session_id", session_id),
            created_at=data.get("created_at", now),
            updated_at=data.get("updated_at", now),
            turns=[Turn(**t) for t in data.get("turns", [])],
        )

    def save(self) -> None:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now().isoformat(timespec="seconds")
        path = SESSIONS_DIR / f"{self.session_id}.json"
        payload = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turns": [asdict(t) for t in self.turns],
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # ---------- access ---------- #
    def next_turn_number(self) -> int:
        return len(self.turns) + 1

    def recent_for_llm(self, k: int = 3) -> list[dict]:
        """Return the last k turns as plain (user, assistant) message dicts.

        These get injected into the tool-loop memory so the LLM sees what
        happened before the current task. We deliberately do NOT include any
        tool-call chains from past turns — only the user task and the final
        summary, so the context stays small.
        """
        out: list[dict] = []
        for t in self.turns[-k:]:
            user_text = t.original_task
            if t.resolved_task and t.resolved_task != t.original_task:
                user_text = (
                    f"{t.original_task}\n"
                    f"(interpreted as: {t.resolved_task})"
                )
            out.append({"role": "user", "content": user_text})
            if t.summary:
                out.append({"role": "assistant", "content": t.summary[:800]})
        return out

    def append_turn(
        self,
        *,
        original_task: str,
        resolved_task: str,
        goal: dict,
        status: str,
        summary: str,
        evidence: dict | None,
    ) -> Turn:
        t = Turn(
            turn=self.next_turn_number(),
            original_task=original_task,
            resolved_task=resolved_task,
            goal=goal or {},
            status=status,
            summary=summary or "",
            evidence=evidence,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )
        self.turns.append(t)
        self.save()
        return t

    def clear(self) -> None:
        self.turns = []
        self.save()

    def summary_line(self) -> str:
        n = len(self.turns)
        if n == 0:
            return f"session '{self.session_id}' (empty)"
        return f"session '{self.session_id}' ({n} prior turn{'s' if n != 1 else ''})"