"""Native session ownership without opening GL or making paid requests."""

import json
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("scipy")
pytest.importorskip("imageio")

from robot_dock.cli import interaction, main
from robot_dock.policies.openai import OpenAICartesianPolicy
from robot_dock.policies.pokebot import PokebotPolicy
from robot_dock.protocol import RunOutcome, RunStatus
from robot_dock.robots.mujoco import ur5e
from robot_dock.robots.mujoco.tabletop import evaluate_lift, evaluate_stack


@pytest.mark.parametrize("followup", ["", "/stack"])
@pytest.mark.parametrize("policy_name", ["pokebot", "openai_cartesian"])
def test_native_cli_preserves_world_between_tasks_and_closes_on_quit(
    tmp_path, monkeypatch, followup, policy_name
):
    assets = tmp_path / "assets"
    hardware = assets.parent / "pokebot" / "ace_pro_2"
    hardware.mkdir(parents=True)
    (hardware / "manifest.json").write_text("{}")
    (hardware / "reference_rig").mkdir()
    (hardware / "reference_rig/lens.json").write_text("{}")
    for name in ("universal_robots_ur5e/ur5e.xml", "robotiq_2f85/2f85.xml"):
        path = assets / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "policy": policy_name,
                "robot": {"type": "mujoco_ur5e", "assets_dir": str(assets)},
                "task": "lift_red",
                "instruction": "Lift and hold the red cube.",
                "max_chunks": 8,
                "max_wall_time_s": 120,
                "output_root": str(tmp_path / "runs"),
            }
        )
    )
    events, policies, worlds = [], [], []

    class Backend:
        def __init__(self, **kwargs):
            worlds.append(self)

        def open(self):
            events.append("open")

        def stop(self):
            events.append("stop")

        def close(self):
            events.append("close")

        def idle(self):
            return True

    def execute(**kwargs):
        assert isinstance(
            kwargs["policy"],
            OpenAICartesianPolicy if policy_name == "openai_cartesian" else PokebotPolicy,
        )
        assert kwargs["backend"] is worlds[0]
        assert kwargs["manage_backend"] is False
        assert "close" not in events and events.count("open") == 1
        expected = evaluate_stack if policies and followup == "/stack" else evaluate_lift
        assert kwargs["evaluate"] is expected
        policies.append(kwargs["policy"])
        root = kwargs["config"].output_dir
        root.mkdir(parents=True)
        events.append("task")
        return RunOutcome("test", RunStatus.SUCCEEDED, True, 8, 620, root)

    answers = iter([followup, None])

    def prompt(*args, **kwargs):
        assert "close" not in events
        events.append("prompt")
        return next(answers)

    monkeypatch.setattr(ur5e, "MujocoUR5eBackend", Backend)
    monkeypatch.setattr(
        main.PokebotEnvironment,
        "read",
        lambda: SimpleNamespace(
            api_key="test-only", model="test-model", base_url="https://test.invalid"
        ),
    )
    monkeypatch.setattr(main, "run", execute)
    monkeypatch.setattr(
        main.OpenAIEnvironment,
        "read",
        lambda: SimpleNamespace(
            api_key="test-only", model="test-model", base_url="https://test.invalid"
        ),
    )
    monkeypatch.setattr(interaction, "next_instruction", prompt)
    monkeypatch.setattr(sys, "argv", ["robot-dock", "run", str(config), "--viewer"])
    assert main.main() == 0
    assert events == ["open", "task", "prompt", "task", "prompt", "stop", "close"]
    assert len(worlds) == 1 and len(policies) == 2 and policies[0] is not policies[1]
    second_config = json.loads(
        next((tmp_path / "runs").glob("*/tasks/0002/configuration.json")).read_text()
    )
    assert second_config["task"] == ("stack_red_on_green" if followup == "/stack" else "lift_red")
