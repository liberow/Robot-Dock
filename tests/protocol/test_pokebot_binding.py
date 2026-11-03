import numpy as np
import pytest

from robot_dock.bindings.pokebot_umi import pokebot_umi_binding
from robot_dock.protocol import ActionChunk, ActionSpec, ImageFrame, Observation


def test_grasp_ground_truth_and_extra_state_are_not_sent_to_policy():
    observation = Observation(
        1,
        100,
        {
            "arm.ee_pose": np.array([0, 0, 0.4, 0, 1, 0, 0]),
            "gripper.opening": np.array([1.0]),
            "gripper.width_m": np.array([0.085]),
        },
        {"wrist": ImageFrame(np.zeros((8, 8, 3), dtype=np.uint8), "fisheye", 100, "wrist")},
        metadata={
            "pose_frame": "task",
            "control_point": "flange",
            "objects": {"red_pad_contacts": {"left": 2, "right": 2}},
        },
    )
    prepared = pokebot_umi_binding().prepare_observation(observation)
    assert set(prepared.state) == {"arm.ee_pose", "gripper.opening"}
    assert prepared.metadata == {}
    assert prepared.images["wrist"] is observation.images["wrist"]


@pytest.mark.parametrize("z", [0.14986114, 0.14443737268447876, 0.13372685, 0.9])
def test_cartesian_targets_pass_through_without_artificial_bounds_or_clipping(z):
    original = ActionChunk(
        {"arm": np.array([[-0.13, -0.65, z, 0, 1, 0, 0]]), "gripper": np.array([[0.5]])},
        {
            "arm": ActionSpec(
                "end_effector_pose", 7, frame="task", control_point="flange", unit="m_xyzw"
            ),
            "gripper": ActionSpec("gripper_position", 1, unit="normalized_opening"),
        },
        1 / 30,
        7,
    )
    applied = pokebot_umi_binding().prepare_actions(original)
    assert applied is original
    assert applied.channels["arm"][0, 2] == z
