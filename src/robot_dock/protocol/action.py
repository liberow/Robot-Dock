from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ActionSpec:
    space: Literal["joint_position", "end_effector_pose", "gripper_position"]
    dimension: int
    mode: Literal["absolute", "delta"] = "absolute"
    frame: str | None = None
    control_point: str | None = None
    unit: str = ""

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise ValueError("Action dimensions must be positive")
        if self.space == "end_effector_pose" and (
            self.dimension != 7
            or not self.frame
            or not self.control_point
            or self.mode != "absolute"
            or self.unit != "m_xyzw"
        ):
            raise ValueError(
                "Cartesian poses currently require absolute meters/xyzw, frame and control point"
            )


@dataclass(frozen=True)
class ActionChunk:
    channels: Mapping[str, NDArray[np.float64]]
    specs: Mapping[str, ActionSpec]
    period_s: float
    source_frame_id: int
    chunk_id: str = field(default_factory=lambda: uuid4().hex)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.period_s) or self.period_s <= 0:
            raise ValueError("Action period must be finite and positive")
        if not self.channels or set(self.channels) != set(self.specs):
            raise ValueError("Every action channel needs one matching spec")
        channels = {}
        lengths = set()
        for name, values in self.channels.items():
            array = np.array(values, dtype=np.float64, copy=True)
            spec = self.specs[name]
            if (
                array.ndim != 2
                or array.shape[1] != spec.dimension
                or not len(array)
                or not np.isfinite(array).all()
            ):
                raise ValueError(f"Invalid action channel: {name}")
            if spec.space == "end_effector_pose" and not np.allclose(
                np.linalg.norm(array[:, 3:7], axis=1), 1, atol=1e-3
            ):
                raise ValueError("Cartesian quaternion must be normalized xyzw")
            array.flags.writeable = False
            channels[name] = array
            lengths.add(len(array))
        if len(lengths) != 1:
            raise ValueError("Action channel lengths must match")
        object.__setattr__(self, "channels", channels)

    @property
    def steps(self) -> int:
        return len(next(iter(self.channels.values())))
