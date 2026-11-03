import numpy as np
import pytest

from robot_dock.protocol import (
    ActionChunk,
    ActionSpec,
    ExecutionReceipt,
    ExecutionStatus,
    ImageFrame,
    Observation,
)


@pytest.mark.parametrize("values", [[[float("nan")]], [[float("inf")]]])
def test_nonfinite_commands_are_rejected(values, action_specs):
    with pytest.raises(ValueError):
        ActionChunk({"gripper": np.array(values)}, action_specs, 1 / 30, 0)


@pytest.mark.parametrize("value", [-0.0012, -0.1, 1.1])
def test_raw_gripper_commands_are_preserved_for_the_device(value, action_specs):
    chunk = ActionChunk({"gripper": np.array([[value]])}, action_specs, 1 / 30, 0)
    assert chunk.channels["gripper"][0, 0] == value


def test_cartesian_contract_is_explicit():
    with pytest.raises(ValueError):
        ActionSpec("end_effector_pose", 7)


def test_incomplete_execution_cannot_claim_completion():
    with pytest.raises(ValueError):
        ExecutionReceipt("id", 40, 17, ExecutionStatus.COMPLETED)


def test_observations_copy_sensor_memory():
    values = np.array([1.0])
    observation = Observation(1, 1, {"joint": values})
    values[0] = 2
    assert observation.state["joint"][0] == 1
    assert not observation.state["joint"].flags.writeable


def test_rgb_frame_does_not_alias_capture_buffer():
    pixels = np.zeros((3, 4, 3), dtype=np.uint8)
    frame = ImageFrame(pixels, "fisheye", 1, "wrist")
    pixels[:] = 255
    assert frame.pixels.max() == 0
