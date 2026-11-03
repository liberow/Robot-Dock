"""Read-only calibrated RGB geometry. No simulator or object-state access."""

import math

import numpy as np
from numpy.typing import NDArray


def rotation_xyzw(quaternion: NDArray[np.float64]) -> NDArray[np.float64]:
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q), 1):
        raise ValueError("Expected a unit XYZW quaternion")
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def pixel_to_plane(
    pixel_uv: list[float],
    plane_z_m: float,
    flange_pose: NDArray[np.float64],
    calibration: dict,
    image_shape: tuple[int, ...],
) -> list[float]:
    """Intersect a calibrated pixel ray with a model-selected horizontal plane.

    This is conditional geometry, not measured object depth: plane_z_m is an
    explicit assumption made by the model or supplied as a workspace calibration.
    """
    uv = np.asarray(pixel_uv, dtype=float)
    if uv.shape != (2,) or not np.isfinite(uv).all() or not math.isfinite(plane_z_m):
        raise ValueError("Expected finite pixel coordinates and plane height")
    height, width = image_shape[:2]
    if not (0 <= uv[0] < width and 0 <= uv[1] < height):
        raise ValueError("Pixel lies outside the observed image")
    lens = calibration["reference_lens"]
    if (width, height) != (lens["nominalWidth"], lens["nominalHeight"]):
        raise ValueError("Image dimensions differ from native lens calibration")
    offset = uv - lens["opticalCenter"]
    radius = float(np.linalg.norm(offset))
    angle = sum(lens[f"k{i}"] * radius**i for i in range(5))
    ray = (
        np.array([0.0, 0.0, 1.0])
        if radius == 0
        else np.r_[offset / radius * math.sin(angle), math.cos(angle)]
    )
    flange_rotation = rotation_xyzw(flange_pose[3:7])
    direction = (
        flange_rotation
        @ rotation_xyzw(np.asarray(calibration["opencv_camera_orientation_flange_xyzw"]))
        @ ray
    )
    origin = flange_pose[:3] + flange_rotation @ np.asarray(calibration["camera_position_flange_m"])
    if abs(direction[2]) < 1e-8:
        raise ValueError("Pixel ray is parallel to the requested plane")
    distance = (plane_z_m - origin[2]) / direction[2]
    if distance <= 0:
        raise ValueError("Requested plane is behind the camera ray")
    return (origin + distance * direction).tolist()
