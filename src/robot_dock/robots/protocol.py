from threading import Event
from typing import Callable, Mapping, Protocol

from robot_dock.protocol import ActionChunk, ActionSpec, ExecutionReceipt, Observation

StepRecorder = Callable[[Observation, ActionChunk, int, int], None]


class RobotBackend(Protocol):
    @property
    def action_specs(self) -> Mapping[str, ActionSpec]: ...

    def open(self) -> None: ...

    def observe(self) -> Observation: ...

    def execute(
        self, actions: ActionChunk, cancel: Event, *, record_step: StepRecorder | None = None
    ) -> ExecutionReceipt: ...

    def stop(self) -> None: ...

    def finish_recording(self) -> None: ...

    def close(self) -> None: ...
