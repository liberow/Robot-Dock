from itertools import product
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("scipy")

import mujoco
from scipy.spatial.transform import Rotation

from robot_dock.robots.mujoco.simulation import HOME, WRIST_FOCAL_PX, Simulation, fisheye_rays


def test_controller_error_is_reported_without_rejecting_or_clipping_policy_command():
    sim = object.__new__(Simulation)
    sim.lock = RLock()
    commands = []
    sim.data = SimpleNamespace(ctrl=np.zeros(7), time=0.0)
    sim.gripper_response = SimpleNamespace(command=lambda t, g: commands.append((t, g)))
    sim.model = SimpleNamespace(actuator_ctrlrange=np.array([[0, 255]] * 7))
    sim.elapsed_actions = sim.gripper_saturated_steps = 0
    # A poor numerical solution is not proof of physical unreachability.
    sim.ik = lambda row: (np.zeros(6), 0.2)
    command = np.array([0, -0.6, 0.1444, 0, 1, 0, 0, -0.1])
    assert sim.step(command) == 0.2
    assert sim.elapsed_actions == 1
    assert commands == [(0.0, -0.1)]
    assert command[2] == 0.1444 and command[7] == -0.1
    assert sim.gripper_saturated_steps == 1


def test_physical_wrist_assembly_and_parallel_lock_with_local_assets(tmp_path):
    assets = Path(__file__).resolve().parents[2] / "assets"
    if not (assets / "pokebot/ace_pro_2/manifest.json").is_file():
        pytest.skip("Optional downloaded hardware assets are not present")
    sim = Simulation(tmp_path, assets / "menagerie")
    try:
        # Scene acceptance is evaluated before any policy runs. Check all cube
        # corners analytically, then independently check the actual rendered pixels.
        sim.capture()
        pixels = sim.wrist()
        np.testing.assert_allclose(
            sim.hardware_metadata["flange_to_pinch_m"], [0, 0, 0.1558], atol=1e-10
        )
        np.testing.assert_allclose(
            sim.state_channels(sim.renderdata)["gripper.center_pose"][:3],
            sim.renderdata.site_xpos[sim.pinch],
        )
        camera = sim.model.camera("front").id
        rotation = sim.data.cam_xmat[camera].reshape(3, 3)
        offsets = np.array(list(product((-0.022, 0.022), repeat=3)))
        for name, channel in (("red", 0), ("green", 1)):
            body = sim.model.body(name).id
            corners = offsets @ sim.data.xmat[body].reshape(3, 3).T + sim.data.xpos[body]
            local = (corners - sim.data.cam_xpos[camera]) @ rotation
            radius = np.linalg.norm(local[:, :2], axis=1)
            angle = np.arctan2(radius, -local[:, 2])
            u = 319.5 + WRIST_FOCAL_PX * angle * local[:, 0] / radius
            v = 179.5 - WRIST_FOCAL_PX * angle * local[:, 1] / radius
            assert np.all((u > 0) & (u < 639) & (v > 0) & (v < 359))
            colors = pixels.astype(float)
            mask = colors[..., channel] > 60
            for other in set(range(3)) - {channel}:
                mask &= colors[..., channel] > 2 * colors[..., other]
            ys, xs = np.where(mask)
            assert len(xs) > 100
            assert xs.min() > 0 and xs.max() < 639 and ys.min() > 0 and ys.max() < 359
        assert sim.hardware_metadata["camera_brackets_lens_mass_kg"] > 0.1772
        ee_rotation = sim.data.site_xmat[sim.flange].reshape(3, 3)
        camera_rotation = sim.data.cam_xmat[sim.model.camera("front").id].reshape(3, 3)
        import json

        transform = np.array(
            json.loads((assets / "pokebot/ace_pro_2/reference_rig/camera.json").read_text())[
                "T_tcp_camera_optical"
            ]
        )
        np.testing.assert_allclose(
            ee_rotation.T @ camera_rotation @ np.diag([1, -1, -1]),
            transform[:3, :3],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            sim.hardware_metadata["camera_position_flange_m"],
            transform[:3, 3],
            atol=1e-6,
        )
        assert sim.hardware_metadata["camera_position_flange_m"][1] < 0
        assert (ee_rotation.T @ -camera_rotation[:, 2])[2] > 0
        # Lock the imported optical transform, including its nonzero tilt and
        # slight calibrated roll/yaw, independently of the nominal CAD mount.
        optical_axis = ee_rotation.T @ -camera_rotation[:, 2]
        np.testing.assert_allclose(optical_axis, transform[:3, 2], atol=1e-6)
        assert 20 < sim.hardware_metadata["camera_tilt_from_flange_deg"] < 40
        assert pixels.dtype == np.uint8 and pixels.shape == (360, 640, 3)
        # No black fisheye mask: the original rectangular RGB frame is retained.
        assert np.all(pixels[[0, 0, -1, -1], [0, -1, 0, -1]].max(axis=1) > 10)
        attachment = sim.model.site("attachment_site").id
        np.testing.assert_allclose(
            sim.data.site_xpos[sim.flange], sim.data.site_xpos[attachment], atol=1e-9
        )
        np.testing.assert_allclose(
            sim.data.site_xmat[sim.flange], sim.data.site_xmat[attachment], atol=1e-9
        )
        for name in ("hw_upper-0", "hw_lower-0", "hw_camera-0", "wide_lens_barrel"):
            geom = sim.model.geom(name)
            assert geom.contype[0] and geom.conaffinity[0]
        assert sum(sim.model.eq_type == 0) == 2  # Retain the original four-bar connects.
        for control in (0, 255):
            sim.gripper_response.command(float(sim.data.time), 1 - control / 255)
            for _ in range(600):
                sim.advance()
            for side in ("left", "right"):
                difference = (
                    sim.data.joint(f"g_{side}_spring_link_joint").qpos[0]
                    - sim.data.joint(f"g_{side}_driver_joint").qpos[0]
                )
                assert abs(difference) < 0.01
        # Verify the device controller independently of task objects or policies.
        # A 5 cm/s Cartesian ramp should not inherit the upstream overdamped lag.
        for _ in range(60):
            sim.step(HOME)
            for _ in range(17):
                sim.advance()
        errors = []
        for index in range(30):
            target = HOME.copy()
            target[0] += 0.05 * (index + 1) / 30
            sim.step(target)
            for _ in range(17):
                sim.advance()
            errors.append(np.linalg.norm(sim.state(sim.data)[:3] - target[:3]))
        assert max(errors) < 0.005
        sim.capture()
        pixels = sim.wrist()
        assert pixels.shape == (360, 640, 3) and pixels.std() > 10
        sim.gripper_response.command(float(sim.data.time), 1 - 128 / 255)
        for _ in range(600):
            sim.advance()
        channels = sim.state_channels(sim.data)
        raw = channels["gripper.raw_position"][0]
        assert abs(raw - 128) < 3
        assert channels["gripper.opening"][0] == pytest.approx(1 - raw / 255, abs=0.01)
        assert 0.04 < channels["gripper.width_m"][0] < 0.047
        # Independent physical direction checks, analogous to the vendor's trials.
        for axis, displacement in ((0, 0.05), (1, -0.05), (2, 0.05)):
            target = HOME.copy()
            target[axis] += displacement
            for _ in range(30):
                sim.step(target)
                for _ in range(17):
                    sim.advance()
            sim.capture()
            assert np.linalg.norm(sim.state()[:3] - target[:3]) < 0.001
            for _ in range(30):
                sim.step(HOME)
                for _ in range(17):
                    sim.advance()
        target = HOME.copy()
        target[3:7] = (
            Rotation.from_euler("x", -30, degrees=True) * Rotation.from_quat(HOME[3:7])
        ).as_quat()
        for _ in range(60):
            sim.step(target)
            for _ in range(17):
                sim.advance()
        sim.capture()
        orientation_error = (
            Rotation.from_quat(sim.state()[3:7]).inv() * Rotation.from_quat(target[3:7])
        ).magnitude()
        assert orientation_error < 0.01
        # Isolate the contact measurement from task planning: place a cube between
        # open pads with gravity disabled, then verify a bilateral static grip.
        sim.model.opt.gravity[:] = 0
        sim.gripper_response.command(float(sim.data.time), 1.0)
        for _ in range(600):
            sim.advance()
        pad_ids = [
            sim.model.geom(f"g_{side}_pad{n}").id for side in ("left", "right") for n in (1, 2)
        ]
        joint = sim.model.body("red").jntadr[0]
        address = sim.model.jnt_qposadr[joint]
        sim.data.qpos[address : address + 3] = np.mean(sim.data.geom_xpos[pad_ids], axis=0)
        rotation = Rotation.from_matrix(sim.data.site_xmat[sim.flange].reshape(3, 3)).as_quat()
        sim.data.qpos[address + 3 : address + 7] = np.roll(rotation, 1)
        sim.data.qvel[sim.model.jnt_dofadr[joint] : sim.model.jnt_dofadr[joint] + 6] = 0
        sim.gripper_response.command(float(sim.data.time), 0.0)
        for _ in range(600):
            sim.advance()
        measured = sim.metrics(sim.data)
        assert all(count > 0 for count in measured["red_pad_contacts"].values())
        assert all(force > 0 for force in measured["red_pad_normal_force_n"].values())
        assert abs(measured["red_offset_from_pad_center_flange_m"][0]) < 0.001
    finally:
        sim.close()


def test_fisheye_raster_matches_independent_3d_landmarks_and_rigid_camera_motion():
    # This tests rendered pixels, including side cameras and seams, rather than
    # merely asserting a transform matrix agrees with its own construction.
    points = [
        ([0, 0, -2], [1, 0, 0]),
        ([0.6, 0, -2], [0, 1, 0]),
        ([0, 0.5, -2], [0, 0, 1]),
        ([-2, 0, -1], [1, 1, 0]),
        ([0.9, 0.5, -0.9], [0, 1, 1]),
        ([0, -1, -1], [1, 0, 1]),
    ]
    cameras = [
        ("front", "0 0 0"),
        ("left", "0 1.57079632679 0"),
        ("right", "0 -1.57079632679 0"),
        ("up", "1.57079632679 0 0"),
        ("down", "-1.57079632679 0 0"),
    ]
    xml = '<mujoco><compiler angle="radian"/><visual><global offwidth="512" offheight="512"/><headlight ambient="1 1 1" diffuse="0 0 0" specular="0 0 0"/></visual><worldbody><body name="rig">'
    xml += "".join(
        f'<camera name="{name}" euler="{angles}" fovy="90"/>' for name, angles in cameras
    )
    xml += "".join(
        f'<geom type="sphere" pos="{" ".join(map(str, p))}" size=".03" rgba="{" ".join(map(str, color))} 1"/>'
        for p, color in points
    )
    xml += "</body></worldbody></mujoco>"
    sim = object.__new__(Simulation)
    sim.model = mujoco.MjModel.from_xml_string(xml)
    sim.renderdata = mujoco.MjData(sim.model)
    sim.renderer = mujoco.Renderer(sim.model, height=512, width=512)
    sim.rays = fisheye_rays()
    sim._fisheye_maps = []
    try:
        for quaternion in ([1, 0, 0, 0], [0.9238795325, 0, 0, 0.3826834324]):
            sim.model.body_quat[sim.model.body("rig").id] = quaternion
            mujoco.mj_forward(sim.model, sim.renderdata)
            pixels = sim.wrist()
            for point, color in points:
                x, y, z = point
                radius = np.hypot(x, y)
                angle = np.arctan2(radius, -z)
                u = 319.5 + WRIST_FOCAL_PX * angle * x / radius if radius else 319.5
                v = 179.5 - WRIST_FOCAL_PX * angle * y / radius if radius else 179.5
                patch = pixels[int(v) - 4 : int(v) + 5, int(u) - 4 : int(u) + 5]
                selected = np.array(color, dtype=bool)
                matches = (patch[..., selected] > 100).all(axis=-1)
                matches &= (patch[..., ~selected] < 40).all(axis=-1)
                assert matches.any(), (point, (u, v))
    finally:
        sim.renderer.close()
