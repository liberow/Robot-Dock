import time
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from robot_dock.protocol import Observation
from robot_dock.robots.mujoco.tabletop import evaluate_lift, evaluate_stack


def stack_observation():
    return Observation(
        1,
        time.monotonic_ns(),
        {
            "arm.ee_pose": np.array([0.02, -0.65, 0.4, 0, 1, 0, 0]),
            "gripper.opening": np.array([1.0]),
        },
        metadata={
            "stack_stable_s": 1.0,
            "objects": {
                "red_xyz": [0.02, -0.65, 0.066],
                "green_xyz": [0.02, -0.65, 0.022],
                "block_distance_xy_m": 0,
                "red_green_contact": True,
                "red_speed_m_s": 0,
                "green_speed_m_s": 0,
            },
        },
    )


def test_released_supported_stack_with_gripper_withdrawn_succeeds():
    assert evaluate_stack(stack_observation())


@pytest.mark.parametrize(
    "changes",
    [
        {"block_distance_xy_m": 0.06, "red_xyz": [0.08, -0.65, 0.022]},
        {"red_xyz": [0.02, -0.65, 0.12]},
        {"red_green_contact": False},
        {"red_speed_m_s": 0.1},
        {"green_xyz": [0.02, -0.65, 0.08]},
    ],
)
def test_adjacent_floating_moving_or_lifted_blocks_are_not_success(changes):
    obs = stack_observation()
    metadata = deepcopy(dict(obs.metadata))
    metadata["objects"].update(changes)
    assert not evaluate_stack(replace(obs, metadata=metadata))


def test_gripper_must_release_and_withdraw():
    obs = stack_observation()
    state = dict(obs.state)
    state["gripper.opening"] = np.array([0.0])
    assert not evaluate_stack(replace(obs, state=state))
    state["gripper.opening"] = np.array([1.0])
    state["arm.ee_pose"] = np.array([0.02, -0.65, 0.224, 0, 1, 0, 0])
    assert not evaluate_stack(replace(obs, state=state))


def test_brief_stack_does_not_count_as_stable_success():
    obs = stack_observation()
    assert not evaluate_stack(replace(obs, metadata={**obs.metadata, "stack_stable_s": 0.5}))


@pytest.mark.parametrize(
    "failure", [None, "brief", "low", "moving", "one_pad", "no_force", "supported"]
)
def test_lift_requires_a_stable_bilateral_unsupported_grasp(failure):
    obs = stack_observation()
    metadata = deepcopy(dict(obs.metadata))
    metadata["lift_stable_s"] = 1.2
    objects = metadata["objects"]
    objects.update(
        red_lift_m=0.08,
        red_green_contact=False,
        red_pad_contacts={"left": 2, "right": 2},
        red_pad_normal_force_n={"left": 1.0, "right": 1.0},
    )
    if failure == "brief":
        metadata["lift_stable_s"] = 0.5
    if failure == "low":
        objects["red_lift_m"] = 0.04
    if failure == "moving":
        objects["red_speed_m_s"] = 0.2
    if failure == "one_pad":
        objects["red_pad_contacts"]["left"] = 0
    if failure == "no_force":
        objects["red_pad_normal_force_n"]["right"] = 0
    if failure == "supported":
        objects["red_green_contact"] = True
    assert bool(evaluate_lift(replace(obs, metadata=metadata))) == (failure is None)
