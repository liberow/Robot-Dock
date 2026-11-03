"""Explicit reference CLI configuration, not a general workflow language."""

import json
import math
from pathlib import Path


def load_reference_config(path: Path) -> dict:
    path = path.resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    allowed = {
        "policy",
        "robot",
        "task",
        "instruction",
        "max_chunks",
        "max_wall_time_s",
        "output_root",
    }
    if set(config) - allowed:
        raise ValueError(f"Unknown configuration fields: {set(config) - allowed}")
    config.setdefault("task", "pick_place")
    if config["task"] not in ("pick_place", "stack_red_on_green", "lift_red"):
        raise ValueError("Unknown reference task")
    if config["policy"] not in ("pokebot", "openai_cartesian"):
        raise ValueError("Unknown policy; expected pokebot or openai_cartesian")
    robot = config["robot"]
    if (
        not {"type", "assets_dir"} <= set(robot)
        or set(robot) - {"type", "assets_dir", "inter_chunk_pause_s", "support_plane_z_m"}
        or robot["type"] != "mujoco_ur5e"
    ):
        raise ValueError("Expected explicit mujoco_ur5e robot configuration")
    pause = robot.get("inter_chunk_pause_s", 0.0)
    if (
        isinstance(pause, bool)
        or not isinstance(pause, (int, float))
        or not math.isfinite(pause)
        or pause < 0
    ):
        raise ValueError("inter_chunk_pause_s must be finite and non-negative")
    plane = robot.get("support_plane_z_m")
    if plane is not None and (
        isinstance(plane, bool) or not isinstance(plane, (int, float)) or not math.isfinite(plane)
    ):
        raise ValueError("support_plane_z_m must be a finite calibrated height")
    for owner, key in [(robot, "assets_dir"), (config, "output_root")]:
        value = Path(owner[key])
        owner[key] = str(
            (path.parent / value).resolve() if not value.is_absolute() else value.resolve()
        )
    return config
