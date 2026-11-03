"""Reference rig command delay and sampled encoder, independent of task objects."""

import math
from collections import deque
from typing import Mapping


class GripperResponse:
    def __init__(self, config: Mapping[str, float], now: float) -> None:
        self.config = config
        self.queue: deque[tuple[float, float]] = deque()
        self.raw = self.goal = config["open_raw"]
        self.last = self.command_deadline = self.sensor_deadline = now
        self.measured = 1.0

    def command(self, now: float, opening: float) -> None:
        c = self.config
        raw = c["closed_raw"] - min(1.0, max(0.0, opening)) * (c["closed_raw"] - c["open_raw"])
        self.queue.append((now + c["delay_s"], raw))

    def advance(self, now: float, measured_opening: float) -> float:
        c = self.config
        dt = now - self.last
        if dt < 0:
            raise ValueError("Actuator time must be monotonic")
        self.last = now
        if now + 1e-9 >= self.command_deadline:
            while self.queue and self.queue[0][0] <= now + 1e-9:
                self.goal = self.queue.popleft()[1]
            period = c["command_period_s"]
            self.command_deadline += (
                math.floor((now - self.command_deadline) / period) + 1
            ) * period
        delta = -math.expm1(-dt / c["tau_s"]) * (self.goal - self.raw)
        self.raw += min(c["max_close_raw_per_s"] * dt, max(-c["max_open_raw_per_s"] * dt, delta))
        span = c["closed_raw"] - c["open_raw"]
        if now + 1e-9 >= self.sensor_deadline:
            raw = c["closed_raw"] - min(1.0, max(0.0, measured_opening)) * span
            quantum = c["sensor_quantum_raw"]
            raw = round(raw / quantum) * quantum
            self.measured = min(1.0, max(0.0, (c["closed_raw"] - raw) / span))
            period = c["sensor_period_s"]
            self.sensor_deadline += (math.floor((now - self.sensor_deadline) / period) + 1) * period
        return (c["closed_raw"] - self.raw) / span
