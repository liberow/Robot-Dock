"""Reference scene's ground-truth evaluator. Never used as a policy observation."""

import numpy as np

from robot_dock.protocol import Observation


def lift_geometry_valid(observation: Observation) -> bool:
    """A cube held clear of the table by both pads, without another support."""
    objects = observation.metadata["objects"]
    return bool(
        objects["red_lift_m"] >= 0.05
        and objects["red_speed_m_s"] < 0.03
        and not objects["red_green_contact"]
        and all(objects["red_pad_contacts"][side] > 0 for side in ("left", "right"))
        and all(objects["red_pad_normal_force_n"][side] > 0 for side in ("left", "right"))
    )


def evaluate_lift(observation: Observation) -> bool:
    """Require a physical bilateral hold for one second, not a peak or a toss."""
    return lift_geometry_valid(observation) and observation.metadata.get("lift_stable_s", 0) >= 1


def evaluate_pick_place(observation: Observation) -> bool:
    objects = observation.metadata["objects"]
    return bool(
        observation.metadata["peak_red_lift_m"] > 0.04
        and objects["red_displacement_m"] > 0.06
        and 0.045 < objects["block_distance_xy_m"] < 0.09
        and objects["red_xyz"][2] < 0.04
        and observation.state["gripper.opening"][0] > 0.75
    )


def stack_geometry_valid(observation: Observation) -> bool:
    """Both cubes must be aligned, touching, released and nearly at rest."""
    objects = observation.metadata["objects"]
    height = objects["red_xyz"][2] - objects["green_xyz"][2]
    pose = observation.state["arm.ee_pose"]
    # Reference gripper's fingertip center is 158 mm along local flange +Z.
    q, offset = pose[3:6], np.array([0, 0, 0.158])
    fingertip = pose[:3] + offset + 2 * np.cross(q, np.cross(q, offset) + pose[6] * offset)
    return bool(
        objects["block_distance_xy_m"] < 0.012
        and abs(height - 0.044) < 0.006
        and abs(objects["green_xyz"][2] - 0.022) < 0.006
        and objects["red_green_contact"]
        and objects["red_speed_m_s"] < 0.02
        and objects["green_speed_m_s"] < 0.02
        and observation.state["gripper.opening"][0] > 0.75
        and np.linalg.norm(fingertip - objects["red_xyz"]) > 0.085
    )


def evaluate_stack(observation: Observation) -> bool:
    """The released stack must persist for at least one second of physical time."""
    return (
        stack_geometry_valid(observation) and observation.metadata.get("stack_stable_s", 0) >= 1.0
    )
