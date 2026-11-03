"""Device response and optical geometry tests; no remote inference validation."""

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("scipy")

from robot_dock.robots.mujoco.actuator import GripperResponse
from robot_dock.robots.mujoco.simulation import fisheye_rays


def test_ftheta_off_axis_ray_uses_pixel_radius_polynomial():
    lens = dict(opticalCenter=[320, 180], k0=0, k1=0.005, k2=0, k3=1e-8, k4=0)
    rays = fisheye_rays(lens)
    # A pixel 200 px from centre leaves at 1.08 radians, not the nominal FOV angle.
    np.testing.assert_allclose(rays[180, 520], [np.sin(1.08), 0, -np.cos(1.08)])
    np.testing.assert_allclose(rays[180, 320], [0, 0, -1])
    np.testing.assert_allclose(np.linalg.norm(rays, axis=-1), 1, atol=1e-12)


def test_gripper_delays_commands_limits_speed_and_samples_measured_travel():
    response = GripperResponse(
        dict(
            open_raw=3,
            closed_raw=229,
            delay_s=0.1,
            tau_s=0.063,
            max_close_raw_per_s=339.5,
            max_open_raw_per_s=312,
            command_period_s=0.05,
            sensor_period_s=0.15,
            sensor_quantum_raw=1,
        ),
        0,
    )
    response.command(0, 0)
    for i in range(50):
        assert response.advance(i * 0.002, 1) == 1
    before = response.raw
    response.advance(0.1, 0.5)
    assert 0 < response.raw - before <= 339.5 * 0.002 + 1e-9
    assert response.measured == 1  # Not a reflection of the closed command.
    response.advance(0.15, 0.5)
    assert response.measured == pytest.approx(0.5)
    response.advance(0.2, 0.1)
    assert response.measured == pytest.approx(0.5)
    with pytest.raises(ValueError, match="monotonic"):
        response.advance(0.19, 0.1)
