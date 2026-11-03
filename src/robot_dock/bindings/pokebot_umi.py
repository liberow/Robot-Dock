"""Reference UMI-like simulation binding. Approximate camera calibration is explicit."""

from robot_dock.bindings import PolicyBinding
from robot_dock.protocol import Observation


def prepare_observation(observation: Observation) -> Observation:
    if (
        observation.metadata.get("pose_frame") != "task"
        or observation.metadata.get("control_point") != "flange"
    ):
        raise ValueError("Expected task-frame flange pose; configure an explicit transform")
    if observation.images["wrist"].projection != "fisheye":
        raise ValueError("Expected fisheye projection")
    # Ground-truth object positions are evaluator-only, not policy input.
    return Observation(
        observation.frame_id,
        observation.captured_at_ns,
        {name: observation.state[name] for name in ("arm.ee_pose", "gripper.opening")},
        {"wrist": observation.images["wrist"]},
        observation.simulation_time_s,
    )


def pokebot_umi_binding() -> PolicyBinding:
    return PolicyBinding(
        name="pokebot_umi",
        prepare_observation=prepare_observation,
    )
