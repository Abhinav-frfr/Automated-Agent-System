"""CLI entry point. Run with:  python -m backend.main [--session NAME]"""

from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

load_dotenv()

from agent.agent import Agent          # noqa: E402
from agent.session import Session      # noqa: E402


def _print_result(result: dict) -> None:
    print("\n=== RESULT ===")
    print(f"Status:  {result.get('status')}")
    goal = result.get("goal") or {}
    if goal:
        print(f"Goal:    {goal.get('task_type')} — {goal.get('desired_outcome')}")
    if result.get("summary"):
        print(f"\n{result['summary']}")
    if result.get("evidence"):
        print("\nEvidence:")
        print(json.dumps(result["evidence"], indent=2))


def _print_history(session: Session, k: int = 10) -> None:
    if not session.turns:
        print("(no history)")
        return
    for t in session.turns[-k:]:
        print(f"  {t.turn}. [{t.status}] {t.original_task}")
        if t.resolved_task and t.resolved_task != t.original_task:
            print(f"     ↳ {t.resolved_task}")
        if t.summary:
            one = t.summary.replace("\n", " ")[:180]
            print(f"     → {one}{'...' if len(t.summary) > 180 else ''}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="default",
                        help="Session name (persists to data/sessions/<name>.json)")
    parser.add_argument("--new", action="store_true",
                        help="Clear the session on startup")
    args = parser.parse_args()

    session = Session.load(args.session)
    if args.new:
        session.clear()
        print(f"Session '{args.session}' cleared.")

    agent = Agent(session=session)
    print(f"Autonomous AI Worker — Day 5.1 (Groq · {agent.model})")
    print(f"Chat:    {session.summary_line()}")
    print()
    print("Commands:  /new        start a fresh session")
    print("           /history    show recent turns")
    print("           /exit       quit")
    print()
    print("Try:")
    print("  'Process the latest Amazon invoice.'    ← records into internal_db.json")
    print("  'Pay the latest Amazon invoice.'        ← HITL → paid_invoices.json")
    print("  'What is the stock symbol of Apple?'")
    print("  'Buy 10 shares of Apple.'               ← HITL → purchased_stock.json")
    print()

    while True:
        try:
            task = input("Task> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not task:
            continue

        cmd = task.lower()
        if cmd in {"exit", "quit", "/exit", "/quit"}:
            break
        if cmd == "/new":
            session.clear()
            print(f"Session '{session.session_id}' cleared.\n")
            continue
        if cmd == "/history":
            _print_history(session)
            print()
            continue

        try:
            result = agent.run(task)
        except Exception as e:
            print(f"\n[error] {e}\n")
            continue

        _print_result(result)


if __name__ == "__main__":
    main()