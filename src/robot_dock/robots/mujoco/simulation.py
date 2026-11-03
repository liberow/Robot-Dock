"""UR5e + upstream Robotiq 2F-85 linkage, contact physics and wrist RGB."""

import copy
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from threading import RLock
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from .actuator import GripperResponse
from .hardware import add_wrist_hardware

HOME = np.array([-0.13, -0.46, 0.49, 0.0, 1.0, 0.0, 0.0, 1.0])
WRIST_FOCAL_PX = math.hypot(319.5, 179.5) / math.radians(189 / 2)


def fisheye_rays(lens: dict | None = None) -> NDArray[np.float64]:
    yy, xx = np.mgrid[:360, :640]
    if lens is not None:
        cx, cy = lens["opticalCenter"]
        dx, dy = xx - cx, yy - cy
        radius = np.hypot(dx, dy)
        angle = sum(lens[f"k{i}"] * radius**i for i in range(5))
        scale = np.sin(angle) / np.maximum(radius, 1e-12)
        return np.stack((dx * scale, -dy * scale, -np.cos(angle)), axis=-1)
    xx, yy = (xx - 319.5) / WRIST_FOCAL_PX, (yy - 179.5) / WRIST_FOCAL_PX
    angle = np.hypot(xx, yy)
    scale = np.sin(angle) / np.maximum(angle, 1e-12)
    return np.stack((xx * scale, -yy * scale, -np.cos(angle)), axis=-1)


def required_element(parent: ET.Element, path: str) -> ET.Element:
    element = parent.find(path)
    if element is None:
        raise ValueError(f"Robot asset is missing XML element: {path}")
    return element


def build_scene(destination: Path, assets_root: Path) -> None:
    paths = Path(assets_root)
    arm_path = paths / "universal_robots_ur5e" / "ur5e.xml"
    grip_path = paths / "robotiq_2f85" / "2f85.xml"
    arm = ET.parse(arm_path).getroot()
    grip = ET.parse(grip_path).getroot()
    # Mesh paths remain valid after composition; avoid overlapping default classes/materials.
    for tree, folder, prefix in [(arm, arm_path.parent, ""), (grip, grip_path.parent, "g_")]:
        for node in tree.iter():
            if node.tag == "mesh" and "file" in node.attrib:
                node.set("name", node.get("name", Path(node.attrib["file"]).stem))
                node.set("file", (folder / "assets" / node.attrib["file"]).as_posix())
            if prefix:
                for attr in [
                    "name",
                    "class",
                    "childclass",
                    "material",
                    "mesh",
                    "joint",
                    "joint1",
                    "joint2",
                    "body1",
                    "body2",
                    "tendon",
                ]:
                    if attr in node.attrib:
                        node.set(attr, prefix + node.attrib[attr])
    arm.remove(required_element(arm, "keyframe"))
    required_element(arm, "compiler").attrib.pop("meshdir", None)
    opt = required_element(arm, "option")
    opt.set("timestep", "0.002")
    opt.set("cone", "elliptic")
    opt.set("impratio", "10")
    opt.set("iterations", "80")
    ET.SubElement(arm, "size", nuserdata="1")
    ET.SubElement(arm, "visual")
    vis = required_element(arm, "visual")
    ET.SubElement(vis, "global", offwidth="1280", offheight="720")
    # Ambient only: view-dependent headlight shading creates seams between the
    # five cubemap faces used to synthesize the fisheye image.
    ET.SubElement(vis, "headlight", ambient=".8 .8 .8", diffuse="0 0 0", specular="0 0 0")
    # Illuminate the full fisheye frame, including rays above the horizon.
    # A black clear colour otherwise looks like a lens mask at the frame corners.
    ET.SubElement(
        required_element(arm, "asset"),
        "texture",
        name="room_background",
        type="skybox",
        builtin="gradient",
        rgb1=".65 .68 .72",
        rgb2=".35 .38 .42",
        width="512",
        height="3072",
    )
    for tag in ["asset", "default", "actuator"]:
        for child in list(required_element(grip, tag)):
            required_element(arm, tag).append(copy.deepcopy(child))
    for tag in ["contact", "tendon", "equality"]:
        arm.append(copy.deepcopy(required_element(grip, tag)))
    wrist = required_element(arm, ".//body[@name='wrist_3_link']")
    flange = ET.SubElement(wrist, "body", name="flange", pos="0 .1 0", quat="-1 1 0 0")
    ET.SubElement(flange, "site", name="flange_site", size=".003")
    flange.append(copy.deepcopy(required_element(grip, "worldbody/body")))
    cameras = add_wrist_hardware(arm, flange, paths.parent / "pokebot" / "ace_pro_2")
    for name, angles in [
        ("front", "0 0 0"),
        ("left", "0 1.5707963 0"),
        ("right", "0 -1.5707963 0"),
        ("up", "1.5707963 0 0"),
        ("down", "-1.5707963 0 0"),
    ]:
        ET.SubElement(cameras, "camera", name=name, euler=angles, fovy="90")
    # Avoid robot self-contact at the attachment while retaining object collisions.
    contact = required_element(arm, "contact")
    for b in ["g_base_mount", "g_base"]:
        ET.SubElement(contact, "exclude", body1="wrist_3_link", body2=b)
    world = required_element(arm, "worldbody")
    for light in list(world.findall("light")):
        world.remove(light)
    ET.SubElement(
        world,
        "light",
        pos="0 -.6 2",
        dir=".3 .3 -1",
        diffuse=".5 .5 .5",
        specular="0 0 0",
        castshadow="false",
    )
    # Separate floor and worktop, with dynamic blocks. No object welding or grasp cheats.
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        pos="0 0 -.04",
        size="2 2 .02",
        rgba=".25 .28 .3 1",
    )
    ET.SubElement(
        world,
        "geom",
        name="table",
        type="box",
        pos="0 -.65 -.025",
        size=".65 .5 .025",
        rgba=".65 .46 .28 1",
        friction="1 .005 .0001",
    )
    for name, pos, color in [
        # Start within the standard wrist view at Home, as in a physical setup.
        # This fixed placement is independent of policy output or task success.
        ("red", f"-.13 {HOME[1]} .022", ".85 .025 .02 1"),
        ("green", f".02 {HOME[1]} .022", ".02 .65 .08 1"),
    ]:
        block = ET.SubElement(world, "body", name=name, pos=pos)
        ET.SubElement(block, "freejoint", name=name + "_free")
        ET.SubElement(
            block,
            "geom",
            name=name + "_geom",
            type="box",
            size=".022 .022 .022",
            mass=".04",
            rgba=color,
            friction="1.5 .01 .001",
            condim="4",
            solref=".004 1",
        )
    destination.write_text(ET.tostring(arm, encoding="unicode"), encoding="utf-8")


class Simulation:
    def __init__(self, output: Path, assets_root: Path) -> None:
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        rig = assets_root.parent / "pokebot" / "ace_pro_2" / "reference_rig"
        if not (rig / "lens.json").is_file():
            raise FileNotFoundError(
                "Reference rig missing; run uv run --group assets python scripts/fetch_assets.py --hardware-only"
            )
        self.lens = json.loads((rig / "lens.json").read_text())
        self.response_config = json.loads((rig / "actuator.json").read_text())["gripper"]
        self.reference_hashes = {
            name: hashlib.sha256((rig / name).read_bytes()).hexdigest()
            for name in ("camera.json", "actuator.json", "lens.json", "bundle_manifest.json")
        }
        build_scene(self.output / "scene.xml", assets_root)
        self.model = m = mujoco.MjModel.from_xml_path(str(self.output / "scene.xml"))
        self.data = d = mujoco.MjData(m)
        self.ikdata = mujoco.MjData(m)
        self.renderdata = mujoco.MjData(m)
        self.lock = RLock()
        self.flange = m.site("flange_site").id
        self.pinch = m.site("g_pinch").id
        self.red = m.body("red").id
        self.green = m.body("green").id
        self.arm_q = np.array([m.jnt_qposadr[i] for i in range(6)])
        self.arm_dof = np.array([m.jnt_dofadr[i] for i in range(6)])
        self.joint_lower = m.jnt_range[:6, 0].copy()
        self.joint_upper = m.jnt_range[:6, 1].copy()
        self.last_ik_position_error_m = 0.0
        self.last_ik_orientation_error_rad = 0.0
        self.gripper_saturated_steps = 0
        q = np.array([-1.57, -1.57, 1.57, -1.57, -1.57, 0.0])
        q, error = self.ik(HOME, q, iterations=250)
        if error > 0.002:
            raise RuntimeError(f"Home IK failed: {error}")
        d.qpos[self.arm_q] = q
        d.ctrl[:6] = q
        d.ctrl[6] = 0
        for _ in range(400):
            d.qfrc_applied[self.arm_dof] = d.qfrc_bias[self.arm_dof]
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        # Menagerie's fixed derivative gains heavily overdamp the lightweight
        # wrists. Use model inertia to set critical joint damping, retaining
        # upstream stiffness and force limits. Targets remain untouched.
        inertia = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, inertia)
        damping = 2 * np.sqrt(m.actuator_gainprm[:6, 0] * np.diag(inertia)[self.arm_dof])
        m.actuator_biasprm[:6, 2] = -damping
        self.gripper_response = GripperResponse(self.response_config, float(d.time))
        d.userdata[0] = 1.0
        assembly = json.loads(
            (assets_root.parent / "pokebot" / "ace_pro_2" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        flange_rotation = d.site_xmat[self.flange].reshape(3, 3)
        camera_id = m.camera("front").id
        self.hardware_metadata = {
            "flange_to_pinch_m": (
                (d.site_xpos[self.pinch] - d.site_xpos[self.flange]) @ flange_rotation
            ).tolist(),
            "source_hashes": {name: item["sha256"] for name, item in assembly["sources"].items()},
            "camera_position_flange_m": (
                (d.cam_xpos[camera_id] - d.site_xpos[self.flange]) @ flange_rotation
            ).tolist(),
            "camera_orientation_flange_xyzw": Rotation.from_matrix(
                flange_rotation.T @ d.cam_xmat[camera_id].reshape(3, 3)
            )
            .as_quat()
            .tolist(),
            "camera_axes": "mujoco_x_right_y_up_z_backward",
            "opencv_camera_orientation_flange_xyzw": Rotation.from_matrix(
                flange_rotation.T @ d.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1, -1, -1])
            )
            .as_quat()
            .tolist(),
            "camera_tilt_from_flange_deg": float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            np.dot(
                                flange_rotation[:, 2], -d.cam_xmat[camera_id].reshape(3, 3)[:, 2]
                            ),
                            -1,
                            1,
                        )
                    )
                )
            ),
            "projection": "reference_rig_v1_ftheta_polynomial",
            "reference_lens": self.lens,
            "reference_rig_hashes": self.reference_hashes,
            "reference_gripper_response": self.response_config,
            "arm_servo_damping": damping.tolist(),
            "arm_servo_profile": "home_inertia_critical_damping_upstream_stiffness_force_limits",
            "principal_point_px": self.lens["opticalCenter"],
            "camera_brackets_lens_mass_kg": float(m.body_subtreemass[m.body("wrist_hardware").id]),
            "camera_body_mass_kg": assembly["camera_mass_kg"],
            "accessory_masses_estimated": True,
            "optics_calibrated": True,
            "calibration_scope": "reference_rig_v1_not_this_physical_UR5e",
            "mesh_collisions": "convex_hull_proxies",
            "gripper_feedback": "reference_rig_v1_sampled_calibrated_travel_proxy",
        }
        self.initial_red = d.xpos[self.red].copy()
        self.renderer = mujoco.Renderer(m, height=512, width=512)
        self.overview_renderer = mujoco.Renderer(m, height=480, width=640)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [-0.05, -0.45, 0.25]
        self.camera.distance = 1.55
        self.camera.azimuth = 135
        self.camera.elevation = -30
        # Native reference-rig pixel-radius to ray-angle polynomial.
        self.rays = fisheye_rays(self.lens)
        self._fisheye_maps: list[tuple[Any, Any, Any]] = []
        self.elapsed_actions = 0
        self.capture()

    def capture(self) -> None:
        with self.lock:
            mujoco.mj_copyData(self.renderdata, self.model, self.data)
        mujoco.mj_forward(self.model, self.renderdata)

    def advance(self) -> None:
        """One physical tick; arm servos include gravity/Coriolis compensation."""
        with self.lock:
            opening = 1 - self.gripper_raw_position(self.data) / 255
            target = self.gripper_response.advance(float(self.data.time), opening)
            self.data.userdata[0] = self.gripper_response.measured
            self.data.ctrl[6] = round((1 - target) * 255)
            self.data.qfrc_applied[self.arm_dof] = self.data.qfrc_bias[self.arm_dof]
            mujoco.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all():
                raise RuntimeError("Physics diverged")

    def width(self, data: Any = None) -> float:
        d = self.data if data is None else data
        a, b = (self.model.geom(name).id for name in ("g_left_pad1", "g_right_pad1"))
        delta = d.geom_xpos[a] - d.geom_xpos[b]
        distance = float(np.linalg.norm(delta))
        axis = delta / max(distance, 1e-12)
        extent = sum(
            float(np.sum(np.abs(d.geom_xmat[g].reshape(3, 3).T @ axis) * self.model.geom_size[g]))
            for g in (a, b)
        )
        return max(0.0, distance - extent)

    def gripper_raw_position(self, data: Any) -> float:
        # Invert the actual position actuator, rather than treating jaw width as
        # motor position. This native actuator proxy is rescaled into calibrated
        # travel before the reference rig's sampled encoder is applied.
        gain = self.model.actuator_gainprm[6, 0]
        bias = self.model.actuator_biasprm[6]
        raw = (-bias[1] * data.actuator_length[6] - bias[0]) / gain
        lower, upper = self.model.actuator_ctrlrange[6]
        return float(np.rint(np.clip(raw, lower, upper)))

    def state(self, data: Any = None) -> NDArray[np.float64]:
        d = self.renderdata if data is None else data
        q = Rotation.from_matrix(d.site_xmat[self.flange].reshape(3, 3)).as_quat()
        opening = float(d.userdata[0])
        return np.r_[d.site_xpos[self.flange], q, opening]

    def state_channels(self, data: Any = None) -> dict[str, NDArray[np.float64]]:
        d = self.renderdata if data is None else data
        state = self.state(d)
        return {
            "arm.ee_pose": state[:7],
            "gripper.opening": state[7:],
            "gripper.center_pose": np.r_[d.site_xpos[self.pinch], state[3:7]],
            "arm.joint_position": d.qpos[self.arm_q].copy(),
            "arm.joint_velocity": d.qvel[self.arm_dof].copy(),
            "gripper.width_m": np.array([self.width(d)]),
            "gripper.raw_position": np.array([self.gripper_raw_position(d)]),
        }

    def ik(
        self,
        target: NDArray[np.float64],
        seed: NDArray[np.float64] | None = None,
        iterations: int = 35,
    ) -> tuple[NDArray[np.float64], float]:
        m, d = self.model, self.ikdata
        with self.lock:
            d.qpos[:] = self.data.qpos
        q = d.qpos[self.arm_q].copy() if seed is None else seed.copy()
        goalrot = Rotation.from_quat(target[3:7]).as_matrix()
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iterations):
            d.qpos[self.arm_q] = q
            mujoco.mj_forward(m, d)
            poserr = target[:3] - d.site_xpos[self.flange]
            roterr = Rotation.from_matrix(
                goalrot @ d.site_xmat[self.flange].reshape(3, 3).T
            ).as_rotvec()
            if np.linalg.norm(poserr) < 0.00005 and np.linalg.norm(roterr) < 0.001:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self.flange)
            jac = np.vstack((jacp[:, self.arm_dof], 0.25 * jacr[:, self.arm_dof]))
            err = np.r_[poserr, 0.25 * roterr]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.00003 * np.eye(6), err)
            q += np.clip(dq, -0.2, 0.2)
            q = np.clip(q, self.joint_lower, self.joint_upper)
        # Evaluate the returned solution, not the previous iteration's pose.
        d.qpos[self.arm_q] = q
        mujoco.mj_forward(m, d)
        self.last_ik_position_error_m = float(np.linalg.norm(target[:3] - d.site_xpos[self.flange]))
        self.last_ik_orientation_error_rad = float(
            np.linalg.norm(
                Rotation.from_matrix(goalrot @ d.site_xmat[self.flange].reshape(3, 3).T).as_rotvec()
            )
        )
        return q, self.last_ik_position_error_m

    def step(self, action: NDArray[np.float64]) -> float:
        row = np.asarray(action, dtype=float).copy()
        if row.shape != (8,) or not np.isfinite(row).all() or np.linalg.norm(row[3:7]) < 1e-8:
            raise ValueError("Bad action")
        row[3:7] /= np.linalg.norm(row[3:7])
        q, residual = self.ik(row)
        with self.lock:
            self.data.ctrl[:6] = q
            # Preserve raw model values in the trace; the delayed device response
            # saturates calibrated travel and maps it to the native actuator.
            self.gripper_response.command(float(self.data.time), float(row[7]))
            self.gripper_saturated_steps += int(not 0 <= row[7] <= 1)
            self.elapsed_actions += 1
        return residual

    def wrist(self, data: Any = None, renderer: Any = None) -> NDArray[np.uint8]:
        m, d = self.model, self.renderdata if data is None else data
        renderer = self.renderer if renderer is None else renderer
        faces, rotations = [], []
        for name in ["front", "left", "right", "up", "down"]:
            option = mujoco.MjvOption()
            option.geomgroup[5] = 0
            renderer.update_scene(d, camera=name, scene_option=option)
            faces.append(renderer.render().copy())
            if not self._fisheye_maps:
                rotations.append(d.cam_xmat[m.camera(name).id].reshape(3, 3).copy())
        if not self._fisheye_maps:
            # Camera mounts are rigid: cache the projection, not images or sensor state.
            world = self.rays @ rotations[0].T
            best = np.full((360, 640), -np.inf)
            for rotation in rotations:
                local = world @ rotation
                depth = -local[..., 2]
                uv = local[..., :2] / np.maximum(depth[..., None], 1e-9)
                valid = (
                    (depth > best) & (depth > 0) & (abs(uv[..., 0]) <= 1) & (abs(uv[..., 1]) <= 1)
                )
                ix = np.clip(((uv[..., 0] + 1) * 255.5).astype(int), 0, 511)
                iy = np.clip(((-uv[..., 1] + 1) * 255.5).astype(int), 0, 511)
                self._fisheye_maps.append((valid, iy[valid], ix[valid]))
                best[valid] = depth[valid]
        image = np.zeros((360, 640, 3), dtype=np.uint8)
        for pixels, (valid, iy, ix) in zip(faces, self._fisheye_maps):
            image[valid] = pixels[iy, ix]
        return image

    def overview(self) -> NDArray[np.uint8]:
        self.overview_renderer.update_scene(self.renderdata, camera=self.camera)
        return self.overview_renderer.render().copy()

    def metrics(self, data: Any = None) -> dict[str, Any]:
        data = self.renderdata if data is None else data
        red = data.xpos[self.red].copy()
        green = data.xpos[self.green].copy()
        pads = {
            side: {self.model.geom(f"g_{side}_pad{number}").id for number in (1, 2)}
            for side in ("left", "right")
        }
        pad_contacts = {side: 0 for side in pads}
        pad_forces = {side: 0.0 for side in pads}
        force = np.zeros(6)
        for index, contact in enumerate(data.contact[: data.ncon]):
            geoms = {int(contact.geom1), int(contact.geom2)}
            if not any(int(self.model.geom_bodyid[g]) == self.red for g in geoms):
                continue
            for side, geometries in pads.items():
                if geoms & geometries:
                    pad_contacts[side] += 1
                    mujoco.mj_contactForce(self.model, data, index, force)
                    pad_forces[side] += float(force[0])
        pad_center = np.mean([data.geom_xpos[g] for ids in pads.values() for g in ids], axis=0)
        return {
            "red_xyz": red.tolist(),
            "green_xyz": green.tolist(),
            "red_displacement_m": float(np.linalg.norm(red - self.initial_red)),
            "red_lift_m": float(red[2] - self.initial_red[2]),
            "block_distance_xy_m": float(np.linalg.norm(red[:2] - green[:2])),
            "physics_time_s": float(data.time),
            "contacts": int(data.ncon),
            "red_speed_m_s": float(np.linalg.norm(data.cvel[self.red, 3:])),
            # Evaluator-only simulator diagnostics; never passed to the policy.
            "red_pad_contacts": pad_contacts,
            "red_pad_normal_force_n": pad_forces,
            "red_offset_from_pad_center_flange_m": (
                (red - pad_center) @ data.site_xmat[self.flange].reshape(3, 3)
            ).tolist(),
            "green_speed_m_s": float(np.linalg.norm(data.cvel[self.green, 3:])),
            "red_green_contact": any(
                {int(self.model.geom_bodyid[c.geom1]), int(self.model.geom_bodyid[c.geom2])}
                == {self.red, self.green}
                for c in data.contact[: data.ncon]
            ),
        }

    def close(self) -> None:
        self.renderer.close()
        self.overview_renderer.close()
