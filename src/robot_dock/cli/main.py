import argparse
import getpass
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from threading import Event
from typing import Callable

from robot_dock import RunConfig, __version__, run
from robot_dock.bindings.openai_cartesian import openai_cartesian_binding
from robot_dock.bindings.pokebot_umi import pokebot_umi_binding
from robot_dock.config import load_reference_config
from robot_dock.config.environment import OpenAIEnvironment, PokebotEnvironment, load_environment
from robot_dock.policies.protocol import Policy
from robot_dock.protocol import Observation
from robot_dock.recording.store import json_value

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(prog="robot-dock")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    execute = sub.add_parser("run", help="Run the reference MuJoCo integration")
    execute.add_argument("config", type=Path)
    execute.add_argument("--viewer", action="store_true")
    execute.add_argument(
        "--once", action="store_true", help="Close the native viewer after one task"
    )
    execute.add_argument(
        "--web-viewer",
        action="store_true",
        help="Loopback browser preview; remains open after the task",
    )
    execute.add_argument("--web-port", type=int, default=8000)
    execute.add_argument("--output", type=Path)
    execute.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
    )
    execute.add_argument(
        "--env-file",
        type=Path,
        help="Explicit environment file; defaults to .env in the current directory",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    load_environment(args.env_file or Path.cwd() / ".env", required=args.env_file is not None)
    config = load_reference_config(args.config)
    if args.viewer and args.web_viewer:
        parser.error("Select either native --viewer or --web-viewer")
    if args.web_viewer and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    output = (
        args.output or Path(config["output_root"]) / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    ).resolve()
    assets = Path(config["robot"]["assets_dir"])
    if not all(
        (assets.parent / "pokebot/ace_pro_2" / name).is_file()
        for name in ("manifest.json", "reference_rig/lens.json")
    ):
        parser.error(
            "Wrist hardware assets missing. Run: uv run --group assets python scripts/fetch_assets.py"
        )
    for name in ("universal_robots_ur5e/ur5e.xml", "robotiq_2f85/2f85.xml"):
        if not (assets / name).is_file():
            parser.error(
                "Robot assets missing. Run: uv run --group assets python scripts/fetch_assets.py"
            )
    from robot_dock.robots.mujoco.tabletop import evaluate_lift, evaluate_pick_place, evaluate_stack
    from robot_dock.robots.mujoco.ur5e import MujocoUR5eBackend
    from robot_dock.visualization import PreviewState, WebViewer

    preview = PreviewState() if args.web_viewer else None
    web = WebViewer(preview, port=args.web_port) if preview is not None else None
    backend = MujocoUR5eBackend(
        assets_dir=assets,
        output_dir=output / "robot",
        viewer=args.viewer,
        preview=preview,
        inter_chunk_pause_s=config["robot"].get("inter_chunk_pause_s", 0.0),
        support_plane_z_m=config["robot"].get("support_plane_z_m"),
    )
    from robot_dock.policies.openai import OpenAICartesianPolicy
    from robot_dock.policies.pokebot import PokebotPolicy

    is_openai = config["policy"] == "openai_cartesian"
    settings = OpenAIEnvironment.read() if is_openai else PokebotEnvironment.read()
    vendor = "OpenAI-compatible Cartesian planner" if is_openai else "RoboAPI"
    logger.info("Execution source: %s remote model | model=%s", vendor, settings.model)
    key = settings.api_key or getpass.getpass(f"{vendor} key (hidden, not saved): ")

    def make_policy(task_output: Path) -> Policy:
        if is_openai:
            return OpenAICartesianPolicy(
                key,
                task_output / "policy",
                model=settings.model,
                base_url=settings.base_url,
                extra_body=settings.extra_body if isinstance(settings, OpenAIEnvironment) else {},
            )
        return PokebotPolicy(
            key, task_output / "policy", model=settings.model, base_url=settings.base_url
        )

    evaluator = {
        "stack_red_on_green": evaluate_stack,
        "pick_place": evaluate_pick_place,
        "lift_red": evaluate_lift,
    }[config["task"]]
    interactive = args.viewer and not args.once
    logger.info("Robot Dock %s | policy=%s | output=%s", __version__, config["policy"], output)
    if web is not None:
        web.start()
        logger.info("Web viewer: http://127.0.0.1:%s (SSH tunnel for remote access)", web.port)
    try:
        if interactive:
            # This session owns the world; each run owns only its policy connection.
            output.mkdir(parents=True, exist_ok=False)
            backend.open()
        instruction = config["instruction"]
        active_evaluator: Callable[[Observation], bool | None] | None = evaluator
        active_task = config["task"]
        task_number = 0
        while True:
            task_number += 1
            task_output = output / "tasks" / f"{task_number:04d}" if interactive else output
            logger.info("Task %d: %s", task_number, instruction)
            outcome = run(
                policy=make_policy(task_output),
                backend=backend,
                binding=openai_cartesian_binding() if is_openai else pokebot_umi_binding(),
                config=RunConfig(
                    output_dir=task_output,
                    instruction=instruction,
                    max_chunks=config["max_chunks"],
                    max_wall_time_s=config["max_wall_time_s"],
                ),
                evaluate=active_evaluator,
                manage_backend=not interactive,
            )
            (task_output / "configuration.json").write_text(
                json.dumps({**config, "task": active_task, "instruction": instruction}, indent=2),
                encoding="utf-8",
            )
            if not interactive:
                break
            from robot_dock.cli.interaction import next_instruction

            if outcome.cleanup_errors or outcome.status.value == "unknown":
                logger.error(
                    "Session cannot continue after uncertain execution or cleanup errors. "
                    "Viewer retained; physical simulation continues."
                )
                next_instruction(backend.idle, allow_next=False)
                break
            logger.info(
                "Task finished. Window stays open; physical simulation continues. "
                "Ctrl+C stops a running task. Custom instructions use the configured budget "
                "and have no automatic success evaluator.",
            )
            answer = next_instruction(backend.idle)
            if answer is None:
                break
            if answer == "/stack":
                instruction = "Place the red cube on top of the green cube."
                active_task = "stack_red_on_green"
                active_evaluator = evaluate_stack
            elif not answer or answer == config["instruction"]:
                instruction = config["instruction"]
                active_task = config["task"]
                active_evaluator = evaluator
            else:
                instruction = answer
                active_evaluator = None
        if web is not None and preview is not None:
            preview.update(phase="finished", outcome=json_value(outcome))
            logger.info("Task finished. Preview retained; Ctrl+C closes the web server.")
            try:
                Event().wait()
            except KeyboardInterrupt:
                pass
    finally:
        if interactive:
            try:
                backend.stop()
            finally:
                backend.close()
        if web is not None:
            web.close()
    return 0 if outcome.task_success and not outcome.cleanup_errors else 1
