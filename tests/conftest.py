import time

import numpy as np
import pytest

from robot_dock.protocol import (
    ActionChunk,
    ActionSpec,
    ExecutionReceipt,
    ExecutionStatus,
    Observation,
)


@pytest.fixture
def action_specs():
    return {"gripper": ActionSpec("gripper_position", 1, unit="normalized_opening")}


class FakePolicy:
    def __init__(self, specs, steps=4):
        self.specs = specs
        self.steps = steps
        self.feedback = []
        self.closed_feedback = None
        self.calls = 0
        self.closed = False

    def open(self):
        pass

    def predict(self, observation, instruction, feedback):
        self.calls += 1
        self.feedback.append(feedback)
        return ActionChunk(
            {"gripper": np.ones((self.steps, 1))}, self.specs, 1 / 30, observation.frame_id
        )

    def close(self, feedback):
        self.closed = True
        self.closed_feedback = feedback


class FakeBackend:
    def __init__(self, specs):
        self.action_specs = specs
        self.count = 0
        self.executions = 0
        self.stopped = False
        self.closed = False

    def open(self):
        pass

    def observe(self):
        return Observation(self.executions, time.monotonic_ns(), {"count": np.array([self.count])})

    def execute(self, chunk, cancel, *, record_step=None):
        self.executions += 1
        for index in range(chunk.steps):
            before = self.observe()
            self.count += 1
            if record_step is not None:
                record_step(before, chunk, index, time.monotonic_ns())
        return ExecutionReceipt(chunk.chunk_id, chunk.steps, chunk.steps, ExecutionStatus.COMPLETED)

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True

    def finish_recording(self):
        pass


@pytest.fixture
def pair(action_specs):
    return FakePolicy(action_specs), FakeBackend(action_specs)
