"""Expose only measured robot state, RGB and static reference camera calibration."""

from robot_dock.bindings import PolicyBinding
from robot_dock.bindings.pokebot_umi import prepare_observation as prepare_umi
from robot_dock.protocol import Observation

CALIBRATION_FIELDS = (
    "camera_position_flange_m",
    "opencv_camera_orientation_flange_xyzw",
    "reference_lens",
    "calibration_scope",
)


def prepare_observation(observation: Observation) -> Observation:
    clean = prepare_umi(observation)
    hardware = observation.metadata.get("hardware", {})
    calibration = {name: hardware[name] for name in CALIBRATION_FIELDS if name in hardware}
    state = dict(clean.state)
    for name in ("gripper.center_pose", "gripper.width_m"):
        if name in observation.state:
            state[name] = observation.state[name]
    workspace = observation.metadata.get("workspace_calibration", {})
    return Observation(
        clean.frame_id,
        clean.captured_at_ns,
        state,
        clean.images,
        clean.simulation_time_s,
        {
            "camera_calibration": calibration,
            "robot_geometry": {"flange_to_pinch_m": hardware.get("flange_to_pinch_m")},
            "workspace_calibration": {
                name: workspace[name] for name in ("support_plane_task_z_m",) if name in workspace
            },
        },
    )


def openai_cartesian_binding() -> PolicyBinding:
    return PolicyBinding(name="openai_cartesian", prepare_observation=prepare_observation)
