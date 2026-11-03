import subprocess
import sys
from pathlib import Path


def test_public_import_does_not_load_device_or_vendor_integrations():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import robot_dock,sys; assert not {'mujoco','scipy','imageio','robot_dock.policies.pokebot'} & set(sys.modules)",
        ],
        check=True,
    )


def test_core_does_not_name_specific_integrations():
    root = Path(__file__).resolve().parents[2] / "src/robot_dock"
    for folder in ("runtime", "protocol", "recording"):
        for path in (root / folder).rglob("*.py"):
            source = path.read_text()
            for forbidden in (
                "robot_dock.policies.pokebot",
                "robot_dock.policies.openai",
                "robot_dock.robots.mujoco",
                "robot_dock.cli",
            ):
                assert forbidden not in source, path
