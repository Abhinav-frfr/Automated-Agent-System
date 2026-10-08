"""CLI entry point. Run with:  python -m backend.main"""

from __future__ import annotations

import json

from dotenv import load_dotenv

load_dotenv()

from agent.agent import Agent  # noqa: E402


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


def main() -> None:
    agent = Agent()
    print(f"Autonomous AI Worker — Day 2 (Groq · {agent.model})")
    print("Try: 'Process the latest Amazon invoice.'")
    print("     'Give me all July invoices.'")
    print("     'Check whether the September Amazon invoice was entered.'")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            task = input("Task> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not task:
            continue
        if task.lower() in {"exit", "quit"}:
            break

        try:
            result = agent.run(task)
        except Exception as e:
            print(f"\n[error] {e}\n")
            continue

        _print_result(result)


if __name__ == "__main__":
    main()