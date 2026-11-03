"""Fetch pinned robot meshes and convert the official RoboAPI wrist assembly."""

import argparse
import hashlib
import io
import json
import struct
import subprocess
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

REVISION = "b0655057a9df2abe39ea7fc989914ba6f248301c"
BASE_URL = "https://developers.pokebot.com"
HARDWARE_FILES = {
    "assembly.glb": (
        "/models/assembly/ace-pro-2.glb",
        "b5840d33121868c356ec37dd893968dc1fb8225197c284de93c96022b48894d3",
    ),
    "upper.stp": (
        "/downloads/hardware/robotiq-ace-pro-2-upper.stp",
        "326701fe0bab50b35e0b48f59fed6bd45e74958dafc90c6a6fd98af830268fe5",
    ),
    "lower.stp": (
        "/downloads/hardware/robotiq-ace-pro-2-lower.stp",
        "c00f06a0332dbe9d9af456ca7f8e7d9510100067d4acdf318b6b3adad2c43158",
    ),
}

RIG_URL = "https://github.com/huashanjian/roboapi-playground/releases/download/roboapi-rig-v1.0.0/roboapi-rig-v1.0.0.tar.gz"
RIG_SHA256 = "56ab79c467a65e54c3156380bec087e4ed2bbd0361bc3b8e250909eccab2e409"


def fetch_rig(target):
    from pxr import Sdf

    target.mkdir(parents=True, exist_ok=True)
    archive = target / "rig.tar.gz"
    data = (
        archive.read_bytes()
        if archive.exists()
        else urllib.request.urlopen(RIG_URL, timeout=60).read()
    )
    if hashlib.sha256(data).hexdigest() != RIG_SHA256:
        raise ValueError("Reference rig archive checksum mismatch")
    archive.write_bytes(data)
    with tarfile.open(fileobj=io.BytesIO(data)) as bundle:
        manifest = json.load(bundle.extractfile("bundle_manifest.json"))
        for name in (
            "camera.json",
            "actuator.json",
            "assembly.usdc",
            "README.md",
            "bundle_manifest.json",
        ):
            content = bundle.extractfile(name).read()
            expected = manifest["files"].get(name)
            if expected and hashlib.sha256(content).hexdigest() != expected["sha256"]:
                raise ValueError(f"Reference rig file checksum mismatch: {name}")
            (target / name).write_bytes(content)
    # Sdf reads authored values without resolving Isaac's external references.
    layer = Sdf.Layer.FindOrOpen(str(target / "assembly.usdc"))
    camera = layer.GetPrimAtPath("/Robot/poke_wrist_camera_link/WristCamera")
    prefix = "omni:lensdistortion:ftheta:"
    lens = {}
    for attribute in camera.attributes:
        if attribute.name.startswith(prefix):
            value = attribute.default
            lens[attribute.name.removeprefix(prefix)] = (
                list(value) if attribute.name.endswith("opticalCenter") else float(value)
            )
    (target / "lens.json").write_text(json.dumps(lens, indent=2), encoding="utf-8")


def fetch_hardware(target):
    import meshopt

    target.mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in HARDWARE_FILES.items():
        path = target / name
        data = (
            path.read_bytes()
            if path.exists()
            else urllib.request.urlopen(BASE_URL + url, timeout=60).read()
        )
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f"Upstream asset differs from the reviewed revision: {url}")
        if not path.exists():
            path.write_bytes(data)
    blob = (target / "assembly.glb").read_bytes()
    length = struct.unpack_from("<I", blob, 12)[0]
    document = json.loads(blob[20 : 20 + length])
    binary = blob[28 + length :]
    decoded = {}
    types = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
    widths = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

    def accessor(index):
        entry = document["accessors"][index]
        key = entry["bufferView"]
        view = document["bufferViews"][key]
        extension = view["extensions"]["EXT_meshopt_compression"]
        if key not in decoded:
            start = extension["byteOffset"]
            decoded[key] = meshopt.decode_gltf_buffer(
                binary[start : start + extension["byteLength"]],
                extension["count"],
                extension["byteStride"],
                extension["mode"],
                extension.get("filter"),
            ).tobytes()
        dtype = np.dtype(types[entry["componentType"]])
        width = widths[entry["type"]]
        values = np.ndarray(
            (entry["count"], width),
            dtype=dtype,
            buffer=decoded[key],
            offset=entry.get("byteOffset", 0),
            strides=(view.get("byteStride", dtype.itemsize * width), dtype.itemsize),
        ).astype(float)
        if entry.get("normalized"):
            values = np.maximum(values / np.iinfo(dtype).max, -1)
        return values

    parts = []
    flange_plane = None
    camera = None

    def visit(index, parent):
        nonlocal flange_plane, camera
        node = document["nodes"][index]
        transform = np.eye(4)
        if "matrix" in node:
            transform = np.asarray(node["matrix"]).reshape(4, 4).T
        else:
            x, y, z, w = node.get("rotation", [0, 0, 0, 1])
            transform[:3, :3] = np.array(
                [
                    [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                    [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                    [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
                ]
            ) @ np.diag(node.get("scale", [1, 1, 1]))
            transform[:3, 3] = node.get("translation", [0, 0, 0])
        transform = parent @ transform
        part = node.get("extras", {}).get("part")
        if part == "camera":
            camera = node["extras"]["lensMount"]
        selected = (
            part in {"upper", "lower", "clamp-fasteners", "side-fasteners", "camera-fasteners"}
            or node.get("mesh") == 13
        )
        if "mesh" in node:
            for primitive_index, primitive in enumerate(
                document["meshes"][node["mesh"]]["primitives"]
            ):
                if not selected and part != "wrist":
                    continue
                vertices = (
                    accessor(primitive["attributes"]["POSITION"]) @ transform[:3, :3].T
                    + transform[:3, 3]
                )
                if part == "wrist":
                    flange_plane = float(vertices[:, 2].max())
                    continue
                if primitive.get("mode", 4) != 4:
                    raise ValueError("Expected triangle meshes")
                faces = accessor(primitive["indices"]).astype(int).reshape(-1, 3)
                material = document["materials"][primitive.get("material", 0)]
                color = material.get("pbrMetallicRoughness", {}).get(
                    "baseColorFactor", [0.35, 0.38, 0.4, 1]
                )
                if "COLOR_0" in primitive["attributes"]:
                    color = accessor(primitive["attributes"]["COLOR_0"]).mean(axis=0).tolist()
                # glTF factors are linear; retain their display colours in MuJoCo.
                color = [
                    12.92 * value if value <= 0.0031308 else 1.055 * value ** (1 / 2.4) - 0.055
                    for value in color[:3]
                ] + [color[3] if len(color) > 3 else 1]
                name = f"{node.get('name', 'camera')}-{primitive_index}"
                path = target / (name + ".obj")
                with path.open("w", encoding="utf-8") as output:
                    output.write(
                        "# Derived from pinned RoboAPI assembly GLB; coordinates in metres.\n"
                    )
                    for vertex in vertices:
                        output.write("v " + " ".join(f"{v:.9g}" for v in vertex) + "\n")
                    for face in faces + 1:
                        output.write("f " + " ".join(str(v) for v in face) + "\n")
                a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
                volume = abs(float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6))
                parts.append(
                    {
                        "name": name,
                        "mesh": path.name,
                        "role": part or "camera",
                        "material": material.get("name", ""),
                        "rgba": color,
                        "volume_m3": volume,
                        "lower": vertices.min(0).tolist(),
                        "upper": vertices.max(0).tolist(),
                    }
                )
        for child in node.get("children", []):
            visit(child, transform)

    for index in document["scenes"][document.get("scene", 0)]["nodes"]:
        visit(index, np.eye(4))
    if flange_plane is None or camera is None:
        raise ValueError("Missing official assembly datums")
    manifest = {
        "version": 1,
        "sources": {
            name: {"url": BASE_URL + url, "sha256": digest}
            for name, (url, digest) in HARDWARE_FILES.items()
        },
        "assembly_to_flange_translation_m": [0, 0, -flange_plane],
        "lens_mount": camera,
        "camera_mass_kg": 0.1772,
        "wide_lens": {"diameter_m": 0.046, "depth_m": 0.022, "estimated_mass_kg": 0.03},
        "estimated_density_kg_m3": {"printed_bracket": 1240, "steel_fastener": 7850},
        "notes": "Official bracket/CAD placement and reconstructed camera exterior. Accessory masses and optical entrance pupil are estimates, not measured calibration.",
        "parts": parts,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware-only", action="store_true")
    args = parser.parse_args()
    assets = Path(__file__).resolve().parents[1] / "assets"
    target = assets / "menagerie"
    if not args.hardware_only and not target.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--sparse",
                "https://github.com/google-deepmind/mujoco_menagerie.git",
                str(target),
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(target),
                "sparse-checkout",
                "set",
                "universal_robots_ur5e",
                "robotiq_2f85",
            ],
            check=True,
        )
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", REVISION], check=True)
    fetch_hardware(assets / "pokebot" / "ace_pro_2")
    fetch_rig(assets / "pokebot" / "ace_pro_2" / "reference_rig")
    print("Assets ready:", assets)


if __name__ == "__main__":
    main()
