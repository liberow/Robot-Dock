"""No GPU/window required; simulate first-frame encoder startup latency."""

from threading import Event, RLock
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("scipy")
pytest.importorskip("imageio")

from robot_dock.protocol import ActionChunk, ExecutionStatus
from robot_dock.robots.mujoco import ur5e


def test_dispatch_deadlines_do_not_accumulate_sleep_jitter(tmp_path, monkeypatch):
    clock = [10.0]
    dispatch_times = []
    backend = ur5e.MujocoUR5eBackend(assets_dir=tmp_path, output_dir=tmp_path)
    sim = SimpleNamespace(
        elapsed_actions=0,
        flange=0,
        lock=RLock(),
        data=SimpleNamespace(site_xpos=np.array([[0, 0, 0.4]])),
        capture=lambda: None,
    )

    def dispatch(row):
        dispatch_times.append(clock[0])
        sim.elapsed_actions += 1
        clock[0] += 0.004  # Controller work per command.
        return 0.0

    def sleep(seconds):
        clock[0] += seconds + 0.003  # Consistent OS wake-up delay.

    sim.step = dispatch
    backend.sim = sim
    monkeypatch.setattr(backend, "_record", lambda label: None)
    monkeypatch.setattr(ur5e.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(ur5e.time, "sleep", sleep)
    chunk = ActionChunk(
        {"arm": np.tile([0, 0, 0.4, 0, 1, 0, 0], (5, 1)), "gripper": np.ones((5, 1))},
        backend.action_specs,
        1 / 30,
        0,
    )
    receipt = backend.execute(chunk, Event())
    assert receipt.status == ExecutionStatus.COMPLETED and receipt.dispatched_steps == 5
    lateness = np.array(dispatch_times) - (10 + np.arange(5) / 30)
    assert lateness.max() <= 0.003001  # Jitter must remain bounded across the chunk.


def test_encoder_cold_start_does_not_age_first_observation(tmp_path, monkeypatch):
    clock = [10_000_000_000]
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)

    class FakeSimulation:
        def __init__(self, output, assets):
            self.data = SimpleNamespace(time=0.8, xpos=np.array([[0, 0, 0.022]]))
            self.renderdata = SimpleNamespace(time=0.8)
            self.lock = RLock()
            self.red = 0
            self.initial_red = np.array([0, 0, 0.022])
            self.last_ik_position_error_m = self.last_ik_orientation_error_rad = 0.0
            self.gripper_saturated_steps = 0
            self.hardware_metadata = {}
            self.elapsed_actions = 0
            self.flange = 0
            self.data.site_xpos = np.array([[0, 0, 0.4]])

        def step(self, row):
            self.elapsed_actions += 1
            return 0.0

        def advance(self):
            with self.lock:
                self.data.time += 0.002

        def capture(self):
            with self.lock:
                self.renderdata.time = self.data.time

        def wrist(self):
            return pixels

        def overview(self):
            return pixels

        def state(self, data=None):
            return np.array([0, 0, 0.4, 0, 1, 0, 0, 1.0])

        def state_channels(self, data=None):
            state = self.state(data)
            return {"arm.ee_pose": state[:7], "gripper.opening": state[7:]}

        def metrics(self, data=None):
            return {
                "red_xyz": [0, 0, 0.022],
                "green_xyz": [0.2, 0, 0.022],
                "block_distance_xy_m": 0.2,
                "red_lift_m": 0.0,
            }

        def close(self):
            pass

    class ColdEncoder:
        def __init__(self):
            self.started = False

        def append_data(self, frame):
            if not self.started:
                clock[0] += 3_000_000_000
                self.started = True

        def close(self):
            pass

    encoder = ColdEncoder()
    monkeypatch.setattr(ur5e, "Simulation", FakeSimulation)
    monkeypatch.setattr(ur5e.imageio, "get_writer", lambda *a, **k: encoder)
    monkeypatch.setattr(ur5e.time, "monotonic_ns", lambda: clock[0])
    # Pacing must not depend on Windows' coarse monotonic tick. Even a frozen
    # public observation clock must not stall the independent physics clock.
    monkeypatch.setattr(ur5e.time, "monotonic", lambda: 10.0)
    backend = ur5e.MujocoUR5eBackend(
        assets_dir=tmp_path, output_dir=tmp_path, inter_chunk_pause_s=0.12
    )
    try:
        backend.open()
        observation = backend.observe()
        assert encoder.started
        assert (clock[0] - observation.captured_at_ns) / 1e9 < 2
        assert observation.frame_id == 1
        before = backend.sim.data.time
        Event().wait(0.12)  # A blocking provider call must not freeze physical time.
        assert backend.sim.data.time - before > 0.05
        assert backend.observe().metadata["timing_mode"] == "continuous_physics"
        from time import perf_counter

        chunk = ActionChunk(
            {"arm": np.array([[0, 0, 0.4, 0, 1, 0, 0]]), "gripper": np.ones((1, 1))},
            backend.action_specs,
            1 / 30,
            backend.frame_id,
        )
        before = backend.sim.data.time
        started = perf_counter()
        receipt = backend.execute(chunk, Event())
        assert receipt.status == ExecutionStatus.COMPLETED
        assert perf_counter() - started >= 0.12
        assert backend.sim.data.time - before > 0.08
        assert backend.observe().metadata["inter_chunk_pause_s"] == 0.12
    finally:
        backend.close()


@pytest.mark.parametrize("renderer_failure", [False, True])
def test_offscreen_contexts_close_before_viewer_even_on_failure(tmp_path, renderer_failure):
    backend = ur5e.MujocoUR5eBackend(assets_dir=tmp_path, output_dir=tmp_path)
    order = []

    def close_renderers():
        order.append("renderers")
        if renderer_failure:
            raise RuntimeError("renderer cleanup failed")

    backend.sim = SimpleNamespace(close=close_renderers)
    backend.viewer = SimpleNamespace(close=lambda: order.append("viewer"))
    if renderer_failure:
        with pytest.raises(RuntimeError, match="renderer cleanup failed"):
            backend.close()
    else:
        backend.close()
    assert order == ["renderers", "viewer"]
