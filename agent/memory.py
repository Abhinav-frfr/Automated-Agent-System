"""Very small message memory for the agent loop."""


class Memory:
    def __init__(self):
        self.history: list[dict] = []

    def add(self, role: str, content, **extra):
        msg = {"role": role, "content": content}
        msg.update(extra)
        self.history.append(msg)

    def add_raw(self, message: dict):
        self.history.append(message)

    def get(self) -> list[dict]:
        return list(self.history)

    def reset(self):
        self.history = []