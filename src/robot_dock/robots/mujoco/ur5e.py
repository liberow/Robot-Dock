from __future__ import annotations

import logging
import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event, Thread
from typing import Any

import imageio.v2 as imageio
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from robot_dock.protocol import (
    ActionChunk,
    ActionSpec,
    ExecutionReceipt,
    ExecutionStatus,
    ImageFrame,
    Observation,
)
from robot_dock.robots.protocol import StepRecorder
from robot_dock.visualization import PreviewState

from .simulation import Simulation
from .tabletop import lift_geometry_valid, stack_geometry_valid

logger = logging.getLogger(__name__)


class MujocoUR5eBackend:
    """Reference UR5e/2F-85 tabletop backend. World is defined in model task coordinates."""

    action_specs = {
        "arm": ActionSpec(
            "end_effector_pose", 7, frame="task", control_point="flange", unit="m_xyzw"
        ),
        "gripper": ActionSpec("gripper_position", 1, unit="normalized_opening"),
    }

    def __init__(
        self,
        *,
        assets_dir: Path,
        output_dir: Path,
        viewer: bool = False,
        preview: PreviewState | None = None,
        inter_chunk_pause_s: float = 0.0,
        support_plane_z_m: float | None = None,
    ):
        if not math.isfinite(inter_chunk_pause_s) or inter_chunk_pause_s < 0:
            raise ValueError("inter_chunk_pause_s must be finite and non-negative")
        self.inter_chunk_pause_s = inter_chunk_pause_s
        if support_plane_z_m is not None and not math.isfinite(support_plane_z_m):
            raise ValueError("support_plane_z_m must be a finite calibrated task-frame height")
        self.support_plane_z_m = support_plane_z_m
        self.assets_dir = assets_dir
        self.output_dir = output_dir
        self.show_viewer = viewer
        self.preview = preview
        self.sim: Simulation | None = None
        self.viewer: Any = None
        self.writer: Any = None
        self.frame_id = 0
        self.last_wrist: NDArray[np.uint8] | None = None
        self.peak_red_lift = 0.0
        self.last_execution: dict[str, float] = {}
        self._physics_stop = Event()
        self._physics_thread: Thread | None = None
        self._physics_error: Exception | None = None
        self._video_worker: ThreadPoolExecutor | None = None
        self._video_pending: list[Future[None]] = []
        self.stack_stable_s = 0.0
        self.lift_stable_s = 0.0
        self.physics_lag_s = 0.0
        self._sensor_worker: ThreadPoolExecutor | None = None
        self._sensor_renderer: Any = None
        self._sample_pending: list[Future[None]] = []

    def open(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sim = Simulation(self.output_dir, self.assets_dir)
        self.writer = imageio.get_writer(
            self.output_dir / "video.mp4", fps=10, codec="libx264", quality=8
        )
        if self.show_viewer:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.sim.model, self.sim.data)
            for key in ("lookat", "distance", "azimuth", "elevation"):
                setattr(self.viewer.cam, key, getattr(self.sim.camera, key))
            self.viewer.sync()

        # Starting the encoder/GL pipeline can block for seconds on Windows.
        # Do that before any timestamped policy observation is acquired.
        self.last_wrist = self.sim.wrist()
        self._record("Robot Dock | initializing renderer and video encoder")
        self._video_worker = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="robot-dock-video"
        )
        self._physics_stop.clear()
        self._physics_thread = Thread(target=self._advance_physics, name="robot-dock-physics")
        self._physics_thread.start()
        self._sensor_worker = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="robot-dock-sensor"
        )

    def _sample_step(
        self,
        data: Any,
        captured_at_ns: int,
        dispatched_at_ns: int,
        actions: ActionChunk,
        index: int,
        record: StepRecorder,
    ) -> None:
        import mujoco

        assert self.sim is not None
        if self._sensor_renderer is None:
            self._sensor_renderer = mujoco.Renderer(self.sim.model, height=512, width=512)
        # Rendering later from this copied physical state preserves capture-time alignment.
        mujoco.mj_forward(self.sim.model, data)
        pixels = self.sim.wrist(data, self._sensor_renderer)
        observation = Observation(
            actions.source_frame_id,
            captured_at_ns,
            self.sim.state_channels(data),
            {"wrist": ImageFrame(pixels, "fisheye", captured_at_ns, "wrist_camera")},
            simulation_time_s=float(data.time),
            metadata={
                "camera_calibration": "reference_rig_v1_ftheta",
                "hardware": self.sim.hardware_metadata,
                "workspace_calibration": (
                    {}
                    if self.support_plane_z_m is None
                    else {"support_plane_task_z_m": self.support_plane_z_m}
                ),
            },
        )
        record(observation, actions, index, dispatched_at_ns)

    def _close_sensor(self) -> None:
        if self._sensor_renderer is not None:
            self._sensor_renderer.close()
            self._sensor_renderer = None

    def _advance_physics(self) -> None:
        assert self.sim is not None
        sim = self.sim
        # On Python 3.12/Windows monotonic() can have a ~16 ms tick. Use the
        # high-resolution performance clock for pacing; observation timestamps
        # keep the public monotonic clock contract used by the runtime.
        origin, initial_time = time.perf_counter(), float(sim.data.time)
        next_sample = initial_time
        stable_since: float | None = None
        lift_since: float | None = None
        try:
            while not self._physics_stop.is_set():
                sim.advance()
                with sim.lock:
                    now = float(sim.data.time)
                    self.peak_red_lift = max(
                        self.peak_red_lift, float(sim.data.xpos[sim.red, 2] - sim.initial_red[2])
                    )
                    if now >= next_sample:
                        state = sim.state(sim.data)
                        measurement = Observation(
                            0,
                            time.monotonic_ns(),
                            {"arm.ee_pose": state[:7], "gripper.opening": state[7:]},
                            metadata={
                                "objects": sim.metrics(sim.data),
                            },
                        )
                        if stack_geometry_valid(measurement):
                            stable_since = now if stable_since is None else stable_since
                            self.stack_stable_s = now - stable_since
                        else:
                            stable_since, self.stack_stable_s = None, 0.0
                        if lift_geometry_valid(measurement):
                            lift_since = now if lift_since is None else lift_since
                            self.lift_stable_s = now - lift_since
                        else:
                            lift_since, self.lift_stable_s = None, 0.0
                        if self.viewer is not None and self.viewer.is_running():
                            self.viewer.sync()
                        next_sample = now + 0.02
                deadline = origin + now - initial_time
                self.physics_lag_s = max(0.0, time.perf_counter() - deadline)
                time.sleep(max(0, deadline - time.perf_counter()))
        except Exception as exc:
            self._physics_error = exc
            logger.exception("Physics worker failed")

    def _check_physics(self) -> None:
        if self._physics_error is not None:
            raise RuntimeError(f"Physics worker failed: {self._physics_error}")

    def _save_frame(self, canvas: Image.Image) -> None:
        self.writer.append_data(np.asarray(canvas))
        canvas.save(self.output_dir / "latest.png")

    def _record(self, label: str) -> None:
        if self.sim is None or self.writer is None or self.last_wrist is None:
            return
        canvas = Image.new("RGB", (1280, 560), "#18202a")
        if self.preview is not None:
            for name, value in (self.preview.take_camera() or {}).items():
                setattr(self.sim.camera, name, value)
        overview = self.sim.overview()
        canvas.paste(Image.fromarray(overview), (0, 0))
        canvas.paste(Image.fromarray(self.last_wrist), (640, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((15, 490), label, fill="white")
        draw.text(
            (15, 515),
            f"Simulation {self.sim.renderdata.time:.2f}s | peak red lift {self.peak_red_lift * 1000:.1f}mm",
            fill="white",
        )
        draw.text(
            (650, 395),
            "Latest wrist observation | physical contact, no object attachment",
            fill="white",
        )
        if self._video_worker is None:
            self._save_frame(canvas)
        else:
            self._video_pending.append(self._video_worker.submit(self._save_frame, canvas))
        if self.preview is not None:
            self.preview.publish(
                overview,
                self.last_wrist,
                label=label,
                simulation_time_s=float(self.sim.renderdata.time),
                observation_frame=self.frame_id,
                peak_red_lift_m=self.peak_red_lift,
                camera={
                    name: float(getattr(self.sim.camera, name))
                    for name in ("azimuth", "elevation", "distance")
                },
            )

    def observe(self) -> Observation:
        assert self.sim is not None
        self._check_physics()
        self.sim.capture()
        timestamp = time.monotonic_ns()
        state = self.sim.state_channels()
        self.last_wrist = self.sim.wrist()
        self.frame_id += 1
        metrics = self.sim.metrics()
        self._record(f"Robot Dock | observation {self.frame_id}")
        if self.preview is not None:
            self.preview.update(phase="waiting_for_policy")
        return Observation(
            self.frame_id,
            timestamp,
            state,
            {"wrist": ImageFrame(self.last_wrist, "fisheye", timestamp, "wrist_camera")},
            simulation_time_s=float(self.sim.renderdata.time),
            metadata={
                "pose_frame": "task",
                "control_point": "flange",
                "objects": metrics,
                "peak_red_lift_m": self.peak_red_lift,
                "timing_mode": "continuous_physics",
                "pacing_clock": "perf_counter",
                "inter_chunk_pause_s": self.inter_chunk_pause_s,
                "physics_lag_s": self.physics_lag_s,
                "stack_stable_s": self.stack_stable_s,
                "lift_stable_s": self.lift_stable_s,
                "controller": {
                    "ik_position_error_m": self.sim.last_ik_position_error_m,
                    "ik_orientation_error_rad": self.sim.last_ik_orientation_error_rad,
                    "gripper_saturated_steps": self.sim.gripper_saturated_steps,
                    **self.last_execution,
                },
                "gripper_model": "robotiq_2f85_c4_parallel_locked",
                "camera_calibration": "reference_rig_v1_ftheta",
                "hardware_profile": "roboapi_reference_rig_v4",
                "scene": "tabletop_home_visible_v2",
                "accessory_inertias": "estimated_not_measured",
                "hardware": self.sim.hardware_metadata,
            },
        )

    def execute(
        self, actions: ActionChunk, cancel: Event, *, record_step: StepRecorder | None = None
    ) -> ExecutionReceipt:
        assert self.sim is not None
        count = 0
        if self.preview is not None:
            self.preview.update(phase="executing", chunk_steps=actions.steps)
        if dict(actions.specs) != self.action_specs or not np.isclose(actions.period_s, 1 / 30):
            return ExecutionReceipt(
                actions.chunk_id,
                actions.steps,
                0,
                ExecutionStatus.FAILED,
                "Unsupported action contract",
            )
        if actions.source_frame_id != self.frame_id:
            return ExecutionReceipt(
                actions.chunk_id,
                actions.steps,
                0,
                ExecutionStatus.FAILED,
                "Stale observation frame",
            )
        rows = np.c_[actions.channels["arm"], actions.channels["gripper"]]
        started = time.perf_counter()
        residuals: list[float] = []
        tracking: list[float] = []
        try:
            for row in rows:
                self._check_physics()
                if cancel.is_set() or (self.viewer is not None and not self.viewer.is_running()):
                    return ExecutionReceipt(
                        actions.chunk_id,
                        actions.steps,
                        count,
                        ExecutionStatus.INTERRUPTED,
                        "cancelled",
                    )
                # Simulation.step may throw during dispatch. Account for partial dispatch below.
                before = self.sim.elapsed_actions
                sample_data = None
                captured_at_ns = time.monotonic_ns()
                if record_step is not None:
                    import mujoco

                    sample_data = mujoco.MjData(self.sim.model)
                    with self.sim.lock:
                        mujoco.mj_copyData(sample_data, self.sim.model, self.sim.data)
                        captured_at_ns = time.monotonic_ns()
                try:
                    residuals.append(self.sim.step(row))
                except BaseException:
                    if self.sim.elapsed_actions > before:
                        count += 1
                    raise
                count += 1
                if record_step is not None and self._sensor_worker is not None:
                    self._sample_pending.append(
                        self._sensor_worker.submit(
                            self._sample_step,
                            sample_data,
                            captured_at_ns,
                            time.monotonic_ns(),
                            actions,
                            count - 1,
                            record_step,
                        )
                    )
                if self.preview is not None:
                    self.preview.update(dispatched_in_chunk=count)
                if count % 3 == 0:
                    self.sim.capture()
                    self._record(f"Robot Dock | chunk step {count}/{actions.steps}")
                # Match the reference SDK's absolute chunk deadlines. Retain
                # every command, including late rows, without accumulating jitter.
                time.sleep(max(0.0, started + count * actions.period_s - time.perf_counter()))
                with self.sim.lock:
                    tracking.append(
                        float(np.linalg.norm(self.sim.data.site_xpos[self.sim.flange] - row[:3]))
                    )
        except KeyboardInterrupt:
            return ExecutionReceipt(
                actions.chunk_id,
                actions.steps,
                count,
                ExecutionStatus.INTERRUPTED,
                "keyboard interrupt",
            )
        except Exception as exc:
            return ExecutionReceipt(
                actions.chunk_id,
                actions.steps,
                count,
                ExecutionStatus.FAILED,
                f"{type(exc).__name__}: {exc}",
            )
        finally:
            dispatch_elapsed = time.perf_counter() - started
            self.last_execution = {
                "chunk_max_ik_position_error_m": max(residuals, default=0.0),
                "chunk_max_tracking_error_m": max(tracking, default=0.0),
                "chunk_dispatch_rate_hz": count / max(dispatch_elapsed, 1e-9),
                "chunk_sampling_pending": float(sum(not f.done() for f in self._sample_pending)),
            }
        if np.any((rows[:, 7] < 0) | (rows[:, 7] > 1)):
            logger.warning("Gripper command exceeds physical actuator range; raw commands retained")
        # Match the reference device's post-chunk settling period before the next
        # observation. Physics keeps running; cancellation never replays commands.
        if self.inter_chunk_pause_s:
            cancel.wait(self.inter_chunk_pause_s)
        return ExecutionReceipt(actions.chunk_id, actions.steps, count, ExecutionStatus.COMPLETED)

    def stop(self) -> None:
        if self.sim is not None:
            with self.sim.lock:
                self.sim.data.ctrl[:6] = self.sim.data.qpos[self.sim.arm_q]

    def finish_recording(self) -> None:
        pending, self._sample_pending = self._sample_pending, []
        errors = []
        if pending:
            logger.info("Finalizing %d execution samples", len(pending))
        for sample in pending:
            try:
                sample.result()
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise OSError("Dataset sampling failed: " + "; ".join(errors))

    def idle(self) -> bool:
        """Physics continues while the terminal waits for the next instruction."""
        if self.viewer is None or not self.viewer.is_running():
            return False
        self._check_physics()
        return True

    def close(self) -> None:
        self._physics_stop.set()
        if self._physics_thread is not None:
            self._physics_thread.join()
        self.finish_recording()
        if self._sensor_worker is not None:
            self._sensor_worker.submit(self._close_sensor).result()
            self._sensor_worker.shutdown(wait=True)
        if self._video_worker is not None:
            self._video_worker.shutdown(wait=True)
        video_errors = []
        for future in self._video_pending:
            try:
                future.result()
            except Exception as exc:
                video_errors.append(str(exc))
        if self.writer is not None:
            self.writer.close()
        # Release offscreen GLFW contexts before shutting down the native viewer.
        # The viewer and renderers share GLFW's process-wide lifecycle on Windows.
        try:
            if self.sim is not None:
                self.sim.close()
        finally:
            if self.viewer is not None:
                self.viewer.close()
        if video_errors:
            raise OSError("Video recording failed: " + "; ".join(video_errors))
