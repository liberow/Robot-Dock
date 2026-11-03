"""Check public names and reference configs; never read real credential files."""

import ast
import tomllib
from pathlib import Path

from dotenv import dotenv_values

from robot_dock.config import load_reference_config


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    assert project["project"]["name"] == "robot-dock"
    assert project["project"]["scripts"] == {"robot-dock": "robot_dock.cli.main:main"}
    module = ast.parse((root / "src/robot_dock/__init__.py").read_text())
    versions = [
        n.value.value
        for n in module.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__version__" for t in n.targets)
        and isinstance(n.value, ast.Constant)
    ]
    assert versions == [project["project"]["version"]], "Version metadata differs"
    template = dotenv_values(root / ".env.example", interpolate=False)
    assert template.get("ROBOT_DOCK_POKEBOT_API_KEY") == "", "Template must not contain a key"
    for path in (root / "configs").glob("*.json"):
        load_reference_config(path)
    print("Repository naming, version, environment template and configurations: OK")


if __name__ == "__main__":
    main()
