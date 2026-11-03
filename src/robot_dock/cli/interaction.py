"""Console input without blocking native viewer event processing."""

import logging
from collections.abc import Callable
from queue import Empty, Queue
from threading import Thread

logger = logging.getLogger(__name__)


def next_instruction(tick: Callable[[], bool], *, allow_next: bool = True) -> str | None:
    answers: Queue[str | None] = Queue()

    def read() -> None:
        try:
            prompt = (
                "Next instruction (Enter: default; /stack: place red on green; /quit: exit): "
                if allow_next
                else "Session stopped; /quit: close viewer: "
            )
            answers.put(input(prompt).strip())
        except EOFError:
            answers.put(None)

    if not tick():
        return None
    Thread(target=read, name="robot-dock-console", daemon=True).start()
    try:
        while tick():
            try:
                answer = answers.get(timeout=0.05)
            except Empty:
                continue
            if answer is None or answer.lower() in ("/quit", "quit", "exit"):
                return None
            if not allow_next:
                logger.warning("Execution needs reconciliation; only /quit is available.")
                Thread(target=read, name="robot-dock-console", daemon=True).start()
                continue
            return answer
    except KeyboardInterrupt:
        pass
    return None
