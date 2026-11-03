import json
import time
from dataclasses import replace
from threading import Event

import h5py
import numpy as np
import pytest

from robot_dock import RunConfig, run
from robot_dock.protocol import ExecutionReceipt, ExecutionStatus, ImageFrame
from robot_dock.recording import EpisodeDataset
from robot_dock.recording.store import RunRecorder


def test_slow_disk_writer_does_not_block_dispatch_thread(tmp_path, monkeypatch):
    entered, release = Event(), Event()
    original = RunRecorder._write_event

    def slow_write(self, line):
        entered.set()
        assert release.wait(2)
        original(self, line)

    monkeypatch.setattr(RunRecorder, "_write_event", slow_write)
    recorder = RunRecorder(tmp_path / "run")
    try:
        recorder.event("execution_planned", {"chunk_id": "test"})
        assert entered.wait(1)
        # The caller is here while the disk worker is still deliberately blocked.
        assert not release.is_set()
        recorder.event("execution_receipt", {"chunk_id": "test", "dispatched_steps": 1})
    finally:
        release.set()
        recorder.summary({"status": "done"})
    lines = (tmp_path / "run/events.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert "execution_planned" in lines[0] and "execution_receipt" in lines[1]


def test_episode_round_trip_has_measured_state_images_raw_actions_and_decision_order(
    tmp_path, pair
):
    policy, backend = pair
    observe = backend.observe

    def with_image():
        obs = observe()
        return replace(
            obs,
            images={
                "wrist": ImageFrame(
                    np.full((4, 5, 3), backend.count, dtype=np.uint8),
                    "fisheye",
                    obs.captured_at_ns,
                    "wrist",
                )
            },
        )

    backend.observe = with_image
    root = tmp_path / "run"
    result = run(policy=policy, backend=backend, config=RunConfig(root, "move", max_chunks=2))
    episode = EpisodeDataset(root / "dataset.h5")
    assert len(episode) == result.dispatched_steps == 8 and episode.training_ready
    for index in range(len(episode)):
        sample = episode[index]
        assert sample["state/count"][0] == index
        assert sample["images/wrist"].shape == (4, 5, 3)
        assert (sample["images/wrist"] == index).all()
        assert sample["action/gripper"][0] == 1
        assert sample["captured_at_ns"] <= sample["dispatched_at_ns"]
        assert sample["image_timestamps_ns/wrist"] == sample["captured_at_ns"]
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    ids = [e["data"]["chunk_id"] for e in events if e["kind"] == "execution_planned"]
    assert [c.chunk_id for c in episode.action_chunks()] == ids
    assert episode.observation(0)["images"]["wrist"].max() == 0
    assert not list(root.glob("frame-*.png"))
    assert episode[-1]["state/count"][0] == 7
    with pytest.raises(IndexError):
        episode[8]


def test_partial_episode_replays_only_actual_dispatched_prefix(tmp_path, pair):
    policy, backend = pair

    def partial(chunk, cancel, *, record_step=None):
        for index in range(2):
            before = backend.observe()
            backend.count += 1
            record_step(before, chunk, index, time.monotonic_ns())
        return ExecutionReceipt(chunk.chunk_id, chunk.steps, 2, ExecutionStatus.INTERRUPTED)

    backend.execute = partial
    run(policy=policy, backend=backend, config=RunConfig(tmp_path / "run", "move"))
    episode = EpisodeDataset(tmp_path / "run/dataset.h5")
    assert len(episode) == 2 and episode.training_ready
    assert episode.outcome["status"] == "cancelled"
    assert [c.steps for c in episode.action_chunks()] == [2]
    assert [c.steps for c in episode.action_chunks(dispatched_only=False)] == [4]


def test_unknown_execution_is_not_offered_as_known_replay(tmp_path, pair):
    policy, backend = pair
    backend.execute = lambda chunk, cancel, **kwargs: ExecutionReceipt(
        chunk.chunk_id, chunk.steps, None, ExecutionStatus.UNKNOWN
    )
    run(policy=policy, backend=backend, config=RunConfig(tmp_path / "run", "move"))
    episode = EpisodeDataset(tmp_path / "run/dataset.h5")
    assert not episode.training_ready
    with pytest.raises(ValueError, match="unknown"):
        list(episode.action_chunks())


def test_unfinalized_episode_is_not_silently_used_for_training(tmp_path):
    path = tmp_path / "incomplete.h5"
    with h5py.File(path, "w") as file:
        file.attrs.update(format="robot-dock-episode", version=1, complete=False)
    with pytest.raises(ValueError, match="incomplete"):
        EpisodeDataset(path)
