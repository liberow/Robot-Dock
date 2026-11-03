"""One portable HDF5 episode, with decision observations and synchronized execution samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np

from robot_dock.protocol import ActionChunk, ActionSpec, Observation


class EpisodeWriter:
    """Called only by the recorder's ordered I/O worker."""

    def __init__(self, path: Path):
        self.path = path
        self.file: Any = None

    def open(self) -> Any:
        if self.file is None:
            self.file = h5py.File(self.path, "x")
            self.file.attrs.update(format="robot-dock-episode", version=1, complete=False)
            for name in ("observations", "decisions", "steps"):
                self.file.create_group(name, track_order=True)
        return self.file

    def observation(self, observation: Observation) -> None:
        group = self.open()["observations"].require_group(str(observation.frame_id))
        if "state" in group:
            return
        group.attrs.update(
            captured_at_ns=observation.captured_at_ns,
            simulation_time_s=observation.simulation_time_s
            if observation.simulation_time_s is not None
            else np.nan,
        )
        for name, values in observation.state.items():
            group.create_dataset("state/" + name, data=values)
        for name, frame in observation.images.items():
            image = group.create_dataset("images/" + name, data=frame.pixels, compression="lzf")
            image.attrs.update(
                captured_at_ns=frame.captured_at_ns, projection=frame.projection, frame=frame.frame
            )

    def event(self, kind: str, data: dict) -> None:
        file = self.open()
        if kind == "run_started":
            file.attrs["instruction"] = data["config"]["instruction"]
            file.attrs["run_id"] = data["run_id"]
            file.attrs["configuration"] = json.dumps(data)
        elif kind == "observation":
            file["observations"].require_group(str(data["frame_id"])).attrs["metadata"] = (
                json.dumps(data["metadata"])
            )
        elif kind in ("policy_action", "execution_planned", "execution_receipt"):
            group = file["decisions"].require_group(data["chunk_id"])
            group.create_dataset(kind, data=json.dumps(data), dtype=h5py.string_dtype())
        elif kind == "run_finished":
            file.attrs["outcome"] = json.dumps(data)

    def step(
        self, observation: Observation, chunk: ActionChunk, index: int, dispatched_at_ns: int
    ) -> None:
        group = self.open()["steps"]
        row = int(group.attrs.get("length", 0))
        values: dict[str, Any] = {
            "captured_at_ns": observation.captured_at_ns,
            "dispatched_at_ns": dispatched_at_ns,
            "simulation_time_s": observation.simulation_time_s
            if observation.simulation_time_s is not None
            else np.nan,
            "chunk_id": chunk.chunk_id,
            "step_index": index,
            "source_frame_id": chunk.source_frame_id,
        }
        values.update({"state/" + name: value for name, value in observation.state.items()})
        values.update({"action/" + name: value[index] for name, value in chunk.channels.items()})
        values.update(
            {"images/" + name: frame.pixels for name, frame in observation.images.items()}
        )
        values.update(
            {
                "image_timestamps_ns/" + name: frame.captured_at_ns
                for name, frame in observation.images.items()
            }
        )
        schema = sorted(values)
        if row and json.loads(group.attrs["columns"]) != schema:
            raise ValueError("Execution sample schema changed within an episode")
        for name, value in values.items():
            array = np.asarray(value)
            if not row:
                kwargs = {"dtype": h5py.string_dtype()} if isinstance(value, str) else {}
                group.create_dataset(
                    name,
                    shape=(0, *array.shape),
                    maxshape=(None, *array.shape),
                    chunks=(1, *array.shape) if name.startswith("images/") else True,
                    compression="lzf",
                    **kwargs,
                    **({"dtype": array.dtype} if not kwargs else {}),
                )
            dataset = group[name]
            if dataset.shape[1:] != array.shape:
                raise ValueError(f"Execution sample shape changed: {name}")
            dataset.resize(row + 1, axis=0)
            dataset[row] = value
        group.attrs["columns"] = json.dumps(schema)
        group.attrs["length"] = row + 1

    def close(self, outcome: dict) -> None:
        file = self.open()
        file.attrs["outcome"] = json.dumps(outcome)
        samples = int(file["steps"].attrs.get("length", 0))
        file.attrs["samples"] = samples
        file.attrs["training_ready"] = bool(
            samples
            and samples == outcome.get("dispatched_steps", 0)
            and not any("recording" in error for error in outcome.get("cleanup_errors", []))
        )
        file.attrs["complete"] = True
        file.close()
        self.file = None


class EpisodeDataset:
    """Lazy NumPy samples; usable directly in a training loop or PyTorch DataLoader.

    No open HDF5 handle is retained, so instances can be passed to worker processes.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        with h5py.File(self.path, "r") as file:
            if file.attrs.get("format") != "robot-dock-episode" or file.attrs.get("version") != 1:
                raise ValueError("Unsupported episode dataset")
            if not file.attrs.get("complete", False):
                raise ValueError(
                    "Episode is incomplete; do not use it as a completed training episode"
                )
            self.instruction = str(file.attrs.get("instruction", ""))
            self.outcome = json.loads(file.attrs["outcome"])
            self.training_ready = bool(file.attrs["training_ready"])
            self._length = int(file.attrs["samples"])

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not -len(self) <= index < len(self):
            raise IndexError(index)
        with h5py.File(self.path, "r") as file:
            group = file["steps"]
            sample = {name: group[name][index] for name in json.loads(group.attrs["columns"])}
        for name, value in sample.items():
            if isinstance(value, bytes):
                sample[name] = value.decode()
        sample["instruction"] = self.instruction
        return sample

    def action_chunks(self, *, dispatched_only: bool = True) -> Iterator[ActionChunk]:
        """Return commands in decision order; by default omit every undispatched row."""
        with h5py.File(self.path, "r") as file:
            for group in file["decisions"].values():
                if "execution_planned" not in group:
                    continue
                data = json.loads(group["execution_planned"].asstr()[()])
                count = len(next(iter(data["channels"].values())))
                if dispatched_only:
                    if "execution_receipt" not in group:
                        continue
                    receipt = json.loads(group["execution_receipt"].asstr()[()])
                    if receipt["dispatched_steps"] is None:
                        raise ValueError("Cannot replay a chunk with unknown execution")
                    count = receipt["dispatched_steps"]
                    if not count:
                        continue
                yield ActionChunk(
                    channels={
                        name: np.asarray(value)[:count] for name, value in data["channels"].items()
                    },
                    specs={name: ActionSpec(**spec) for name, spec in data["specs"].items()},
                    period_s=data["period_s"],
                    source_frame_id=data["source_frame_id"],
                    chunk_id=data["chunk_id"],
                    metadata=data["metadata"],
                )

    def observation(self, frame_id: int) -> dict[str, Any]:
        def arrays(group: Any) -> dict[str, Any]:
            result = {}
            group.visititems(
                lambda name, node: (
                    result.update({name: node[:]}) if isinstance(node, h5py.Dataset) else None
                )
            )
            return result

        with h5py.File(self.path, "r") as file:
            group = file[f"observations/{frame_id}"]
            return {
                "state": arrays(group["state"]),
                "images": arrays(group["images"]) if "images" in group else {},
                "captured_at_ns": int(group.attrs["captured_at_ns"]),
                "metadata": json.loads(group.attrs.get("metadata", "{}")),
            }
