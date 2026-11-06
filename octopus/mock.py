"""Mock engine so the interface can be developed and demoed without a GPU or model download."""

from __future__ import annotations

import time

CHAT = """Hi, I'm **Octopus** 🐙, running in mock mode, so no model is loaded yet.

When you start Octopus normally I run on *your* GPU, and I can:

- chat and answer questions
- write, explain and debug code, then run it with one click
- learn from your own examples in the **Train** section

| Mode | What it does |
|---|---|
| Chat | Everyday assistant |
| Code | Programming help with a Run button |
"""

CODE = """Here's an iterative Fibonacci that runs in O(n) time:

```python
def fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

print([fib(i) for i in range(12)])
```

Press **Run** to try it on your machine. The loop keeps only the last two numbers, so it uses constant memory and avoids the exponential blow-up of the naive recursive version."""


class MockEngine:
    status = "ready"
    base_id = "mock"
    adapter = None
    error = None

    def info(self):
        return {"status": "ready", "error": None, "base": "mock", "adapter": self.adapter, "device": "Mock mode (no GPU used)"}

    def ensure_loaded(self, base_id=None, adapter=None):
        self.base_id, self.adapter = base_id or "mock", adapter

    def unload(self):
        pass

    def stream(self, messages, mode="chat", stop=None, **kw):
        text = CODE if mode == "code" else CHAT
        for i in range(0, len(text), 5):
            if stop and stop.event.is_set():
                return
            time.sleep(0.012)
            yield text[i:i + 5]
