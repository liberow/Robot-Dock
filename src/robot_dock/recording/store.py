from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from robot_dock.protocol import ActionChunk, Observation

from .dataset import EpisodeWriter


def json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, (Path, Enum)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


class RunRecorder:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=False)
        self._sequence = 0
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="robot-dock-records")
        self._pending: list[Future[None]] = []
        self._dataset = EpisodeWriter(root / "dataset.h5")

    def event(self, kind: str, data: Any) -> None:
        payload = {"sequence": self._sequence, "kind": kind, "data": json_value(data)}
        line = json.dumps(payload, allow_nan=False) + "\n"
        self._pending.append(self._worker.submit(self._write_event, line))
        self._sequence += 1

    def _write_event(self, line: str) -> None:
        event = json.loads(line)
        self._dataset.event(event["kind"], event["data"])
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(line)

    def observation(self, observation: Observation) -> None:
        self._pending.append(self._worker.submit(self._dataset.observation, observation))
        images = {}
        for name, frame in observation.images.items():
            path = f"dataset.h5::/observations/{observation.frame_id}/images/{name}"
            images[name] = {
                "path": path,
                "projection": frame.projection,
                "frame": frame.frame,
                "captured_at_ns": frame.captured_at_ns,
            }
        self.event(
            "observation",
            {
                "frame_id": observation.frame_id,
                "captured_at_ns": observation.captured_at_ns,
                "state": observation.state,
                "images": images,
                "simulation_time_s": observation.simulation_time_s,
                "metadata": observation.metadata,
            },
        )

    def step(
        self, observation: Observation, chunk: ActionChunk, index: int, dispatched_at_ns: int
    ) -> None:
        self._pending.append(
            self._worker.submit(self._dataset.step, observation, chunk, index, dispatched_at_ns)
        )

    def flush(self) -> None:
        pending, self._pending = self._pending, []
        errors = []
        for future in pending:
            try:
                future.result()
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise OSError("; ".join(errors))

    def summary(self, value: Any) -> None:
        # Finalize the episode on the same worker that owns all HDF5 writes.
        self._pending.append(self._worker.submit(self._dataset.close, json_value(value)))
        self._worker.shutdown(wait=True)
        self.flush()
        temporary = self.root / "summary.json.tmp"
        temporary.write_text(
            json.dumps(json_value(value), indent=2, allow_nan=False), encoding="utf-8"
        )
        temporary.replace(self.root / "summary.json")
