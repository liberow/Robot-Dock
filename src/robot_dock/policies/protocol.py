from typing import Protocol

from robot_dock.protocol import ActionChunk, ExecutionReceipt, Observation


class Policy(Protocol):
    def open(self) -> None: ...

    def predict(
        self, observation: Observation, instruction: str, feedback: ExecutionReceipt | None
    ) -> ActionChunk: ...

    def close(self, feedback: ExecutionReceipt | None) -> None: ...
