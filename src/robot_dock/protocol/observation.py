from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ImageFrame:
    pixels: NDArray[np.uint8]
    projection: str
    captured_at_ns: int
    frame: str

    def __post_init__(self) -> None:
        pixels = np.array(self.pixels, copy=True)
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError("Images must be HxWx3 uint8 RGB")
        pixels.flags.writeable = False
        object.__setattr__(self, "pixels", pixels)


@dataclass(frozen=True)
class Observation:
    frame_id: int
    captured_at_ns: int
    state: Mapping[str, NDArray[np.float64]]
    images: Mapping[str, ImageFrame] = field(default_factory=dict)
    simulation_time_s: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        state = {}
        for name, values in self.state.items():
            array = np.array(values, dtype=np.float64, copy=True)
            if array.ndim != 1 or not np.isfinite(array).all():
                raise ValueError(f"Invalid state channel: {name}")
            array.flags.writeable = False
            state[name] = array
        object.__setattr__(self, "state", state)
