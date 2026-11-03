"""Physical wrist assembly from the official RoboAPI CAD-derived asset pack."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def numbers(values: list[float]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def add_wrist_hardware(arm: ET.Element, flange: ET.Element, directory: Path) -> ET.Element:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    asset, equality, contact = arm.find("asset"), arm.find("equality"), arm.find("contact")
    assert asset is not None and equality is not None and contact is not None
    assembly = ET.SubElement(
        flange,
        "body",
        name="wrist_hardware",
        pos=numbers(manifest["assembly_to_flange_translation_m"]),
        # The CAD presentation axes are not RoboAPI's signed EE axes. Register
        # the physical assembly so image-right is +X_EE, as the provider specifies.
        quat="0 0 0 1",
    )
    density = manifest["estimated_density_kg_m3"]
    for part in manifest["parts"]:
        name = "hw_" + part["name"]
        ET.SubElement(asset, "mesh", name=name, file=str((directory / part["mesh"]).resolve()))
        role, material = part["role"], part["material"]
        if role in ("lower", "upper"):
            mass = part["volume_m3"] * density["printed_bracket"]
        elif "fasteners" in role:
            mass = part["volume_m3"] * density["steel_fastener"]
        elif material == "ace-body":
            mass = manifest["camera_mass_kg"]
        elif material == "ace-cage":
            mass = part["volume_m3"] * density["printed_bracket"]
        else:
            mass = 0
        # Mesh collisions are convex hulls; this approximation is explicit in the asset notes.
        collide = role in ("upper", "lower") or material in ("ace-body", "ace-cage")
        ET.SubElement(
            assembly,
            "geom",
            name=name,
            type="mesh",
            mesh=name,
            rgba=numbers(part["rgba"]),
            mass=f"{mass:.9g}",
            contype="1" if collide else "0",
            conaffinity="1" if collide else "0",
            group="2",
        )
    mount = manifest["lens_mount"]
    lens = ET.SubElement(
        assembly,
        "body",
        name="wide_lens",
        pos=numbers(mount["position"]),
        euler=f"{mount['rotationX']:.9g} 0 0",
    )
    spec = manifest["wide_lens"]
    radius, depth = spec["diameter_m"] / 2, spec["depth_m"]
    ET.SubElement(
        lens,
        "geom",
        name="wide_lens_barrel",
        type="cylinder",
        pos=f"0 0 {depth / 2}",
        size=f"{radius} {depth / 2}",
        mass=str(spec["estimated_mass_kg"]),
        rgba=".08 .09 .10 1",
        group="2",
    )
    ET.SubElement(
        lens,
        "geom",
        name="wide_lens_front",
        type="cylinder",
        pos=f"0 0 {depth + 0.0003}",
        size=f"{radius * 0.89} .0003",
        mass="0",
        rgba=".02 .09 .10 1",
        group="2",
    )
    # Keep the visible CAD mount, but use the reference rig's optical centre.
    calibration = json.loads((directory / "reference_rig" / "camera.json").read_text())
    transform = np.asarray(calibration["T_tcp_camera_optical"], dtype=float)
    rotation = Rotation.from_matrix(transform[:3, :3] @ np.diag([1, -1, -1])).as_quat()
    camera_frame = ET.SubElement(
        flange,
        "body",
        name="camera_mount",
        pos=numbers(transform[:3, 3].tolist()),
        quat=numbers(rotation[[3, 0, 1, 2]].tolist()),
    )
    # The optical centre is behind the CAD glass envelope. Hide only those
    # opaque lens proxies from wrist rendering, retaining them in external views.
    for geom in lens.findall("geom"):
        geom.set("group", "5")

    # C4 kit: the pins remove adaptive folding, not the upstream four-bar linkage.
    # Only lock the additional adaptive degree of freedom; retain the native connects.
    for side, sign, source_y in (("right", 1, -0.0008), ("left", -1, 0)):
        ET.SubElement(
            equality,
            "joint",
            name=f"{side}_parallel_lock",
            joint1=f"g_{side}_spring_link_joint",
            joint2=f"g_{side}_driver_joint",
            polycoef="0 1 0 0 0",
            solref=".004 1",
        )
        driver = arm.find(f".//body[@name='g_{side}_driver']")
        assert driver is not None
        # Datums/diameter/length are taken from the official C4 assembly viewer.
        local_y = sign * (0.05362891 - 0.0306011)
        source_z = 0.02011147 + manifest["assembly_to_flange_translation_m"][2]
        local_z = source_z - (0.0108 + 0.054904)
        pin = ET.SubElement(
            driver, "body", name=f"{side}_locking_pin", pos=numbers([-source_y, local_y, local_z])
        )
        ET.SubElement(
            pin,
            "geom",
            name=f"{side}_locking_shaft",
            type="cylinder",
            size=".0025 .0115",
            quat=".707106781 0 .707106781 0",
            density=str(density["steel_fastener"]),
            rgba=".55 .58 .62 1",
            group="2",
        )
        for end in (-1, 1):
            ET.SubElement(
                pin,
                "geom",
                name=f"{side}_retaining_screw_{end}",
                type="cylinder",
                pos=f"{end * 0.01235} 0 0",
                size=".00275 .00085",
                quat=".707106781 0 .707106781 0",
                density=str(density["steel_fastener"]),
                rgba=".3 .32 .35 1",
                group="2",
            )
        # Convex mesh proxies cannot represent pin bores: omit only mating hardware
        # self-contacts; collisions with external objects remain active.
        for link in ("driver", "spring_link", "coupler", "follower"):
            ET.SubElement(contact, "exclude", body1=f"{side}_locking_pin", body2=f"g_{side}_{link}")
    return camera_frame
