"""Evaluation harness — expanded in later days.

For Day 1 this simply checks that the agent's tool registry is populated.
"""

from agent.agent import TOOL_REGISTRY


def main() -> None:
    print("Registered tools:")
    for name in TOOL_REGISTRY:
        print(f"  - {name}")


if __name__ == "__main__":
    main()