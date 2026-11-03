import numpy as np
import pytest

from robot_dock.policies.openai.geometry import pixel_to_plane, rotation_xyzw


def calibration():
    return {
        "reference_lens": dict(
            opticalCenter=[320, 180],
            nominalWidth=640,
            nominalHeight=360,
            k0=0,
            k1=np.pi / 400,
            k2=0,
            k3=0,
            k4=0,
        ),
        "camera_position_flange_m": [0, 0, 0.1],
        "opencv_camera_orientation_flange_xyzw": [0, 0, 0, 1],
    }


def test_pixel_plane_intersection_uses_extrinsics_and_task_flange_orientation():
    # Camera at (1,2,.9), optical axis down, image +X points task -X.
    # A radius of 100 px is a 45-degree ray: a .9 m drop moves task X by -.9 m.
    pose = np.array([1, 2, 1, 0, 1, 0, 0])
    np.testing.assert_allclose(
        pixel_to_plane([320, 180], 0, pose, calibration(), (360, 640, 3)), [1, 2, 0]
    )
    np.testing.assert_allclose(
        pixel_to_plane([420, 180], 0, pose, calibration(), (360, 640, 3)), [0.1, 2, 0]
    )
    # A different assumed object height changes XY via parallax, not a depth sensor.
    np.testing.assert_allclose(
        pixel_to_plane([420, 180], 0.4, pose, calibration(), (360, 640, 3)), [0.5, 2, 0.4]
    )


@pytest.mark.parametrize(
    "uv,z,shape",
    [
        ([-1, 180], 0, (360, 640, 3)),
        ([640, 180], 0, (360, 640, 3)),
        ([320, float("nan")], 0, (360, 640, 3)),
        ([320, 180], float("inf"), (360, 640, 3)),
        ([320, 180], 2, (360, 640, 3)),
        ([320, 180], 0, (180, 320, 3)),
        ([520, 180], 0, (360, 640, 3)),
    ],
)
def test_invalid_or_backward_pixel_plane_queries_are_rejected(uv, z, shape):
    with pytest.raises(ValueError):
        pixel_to_plane(uv, z, np.array([0, 0, 1, 0, 1, 0, 0]), calibration(), shape)


def test_quaternion_rotation_does_not_accept_invalid_calibration():
    with pytest.raises(ValueError):
        rotation_xyzw(np.array([0, 0, 0, 0]))
