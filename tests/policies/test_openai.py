import io
import json
from urllib.error import HTTPError

import numpy as np
import pytest

from robot_dock.bindings.openai_cartesian import prepare_observation
from robot_dock.config.environment import OpenAIEnvironment
from robot_dock.policies.openai import client
from robot_dock.protocol import ExecutionReceipt, ExecutionStatus, ImageFrame, Observation


def observation():
    return Observation(
        7,
        1,
        {"arm.ee_pose": np.array([0, 0, 0.5, 0, 1, 0, 0]), "gripper.opening": np.array([1.0])},
        {"wrist": ImageFrame(np.zeros((8, 8, 3), dtype=np.uint8), "fisheye", 1, "wrist_camera")},
        metadata={
            "pose_frame": "task",
            "control_point": "flange",
            "objects": "SECRET_GT",
            "hardware": {"camera_position_flange_m": [0, 0, 0.1], "objects": "SECRET_GT"},
        },
    )


def plan():
    return {"waypoints": [{"pose": [0.06, 0, 0.5, 0, 0, 0, 1], "opening": 0.0, "duration_s": 1.0}]}


def test_interpolation_preserves_endpoint_and_uses_shortest_rotation():
    start = np.array([0, 0, 0.5, 0, 1, 0, 0, 1.0])
    p = plan()
    rows = client.interpolate_plan(p, start)
    assert rows.shape == (30, 8)
    np.testing.assert_allclose(rows[-1], p["waypoints"][0]["pose"] + [0.0])
    np.testing.assert_allclose(rows[14, :3], [0.03, 0, 0.5])
    np.testing.assert_allclose(np.linalg.norm(rows[:, 3:7], axis=1), 1)
    p["waypoints"][0]["pose"][3:] = [0, -1, 0, 0]
    rows = client.interpolate_plan(p, start)
    np.testing.assert_allclose(rows[:-1, 3:7], np.tile([0, 1, 0, 0], (29, 1)))
    np.testing.assert_array_equal(rows[-1, 3:7], [0, -1, 0, 0])


@pytest.mark.parametrize(
    "field,value",
    [
        ("opening", -1),
        ("opening", float("nan")),
        ("opening", True),
        ("duration_s", 0),
        ("duration_s", 0.05),
        ("duration_s", 7),
        ("pose", [0] * 7),
        ("pose", [0, 0, float("inf"), 0, 1, 0, 0]),
    ],
)
def test_invalid_plans_are_rejected_without_target_correction(field, value):
    p = plan()
    p["waypoints"][0][field] = value
    with pytest.raises(ValueError):
        client.interpolate_plan(p, np.array([0, 0, 0.5, 0, 1, 0, 0, 1]))


def test_plan_duration_budget_and_count():
    start = np.array([0, 0, 0.5, 0, 1, 0, 0, 1])
    for points in [[], plan()["waypoints"] * 9, plan()["waypoints"] * 7]:
        with pytest.raises(ValueError):
            client.interpolate_plan({"waypoints": points}, start)


def test_remote_wire_input_excludes_ground_truth_and_requires_real_receipt(tmp_path, monkeypatch):
    requests = []

    def respond(request, **kwargs):
        body = json.loads(request.data)
        requests.append(body)
        assert "SECRET_GT" not in json.dumps(body)
        return io.BytesIO(
            json.dumps(
                {
                    "model": "test-model",
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": json.dumps(plan())}}
                    ],
                }
            ).encode()
        )

    monkeypatch.setattr(client, "urlopen", respond)
    policy = client.OpenAICartesianPolicy("private-test-key", tmp_path, model="test-model")
    policy.open()
    obs = prepare_observation(observation())
    chunk = policy.predict(obs, "Lift the cube", None)
    assert chunk.source_frame_id == 7 and chunk.steps == 30
    assert "private-test-key" not in repr(policy)
    with pytest.raises(ValueError, match="receipt"):
        policy.predict(obs, "Lift the cube", None)
    receipt = ExecutionReceipt(chunk.chunk_id, 30, 12, ExecutionStatus.INTERRUPTED)
    next_chunk = policy.predict(obs, "Lift the cube", receipt)
    payload = json.loads(requests[1]["messages"][1]["content"][0]["text"])
    assert payload["execution_receipt"]["dispatched_steps"] == 12
    assert payload["camera_calibration"] == {"camera_position_flange_m": [0, 0, 0.1]}
    parts = requests[1]["messages"][1]["content"]
    assert len([part for part in parts if part["type"] == "image_url"]) == 2
    policy.close(ExecutionReceipt(next_chunk.chunk_id, 30, 30, ExecutionStatus.COMPLETED))
    assert all("private-test-key" not in path.read_text() for path in tmp_path.iterdir())


def test_http_failure_is_not_retried(tmp_path, monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise HTTPError("https://test.invalid", 503, "busy", {}, None)

    monkeypatch.setattr(client, "urlopen", fail)
    policy = client.OpenAICartesianPolicy("private-test-key", tmp_path, model="test-model")
    policy.open()
    with pytest.raises(RuntimeError, match="HTTP 503"):
        policy.predict(prepare_observation(observation()), "Lift", None)
    with pytest.raises(RuntimeError, match="never retried"):
        policy.predict(prepare_observation(observation()), "Lift", None)
    policy.close(None)
    assert calls == [1]


def test_openai_environment_uses_user_variables_without_exposing_key(monkeypatch):
    for name, value in {
        "OPENAI_API_KEY": "private-test-key",
        "OPENAI_API_BASE": "https://test.invalid/v1",
        "OPENAI_API_MODEL": "test-model",
    }.items():
        monkeypatch.setenv(name, value)
    settings = OpenAIEnvironment.read()
    assert settings.model == "test-model" and settings.base_url == "https://test.invalid/v1"
    assert settings.api_key == "private-test-key" and settings.api_key not in repr(settings)


def test_provider_options_and_truncated_response_stop_before_dispatch(tmp_path, monkeypatch):
    options = {"max_completion_tokens": 100, "reasoning_effort": "none"}
    policy = client.OpenAICartesianPolicy(
        "private-test-key", tmp_path, model="test-model", extra_body=options
    )

    def truncated(request, **kwargs):
        body = json.loads(request.data)
        assert "max_tokens" not in body and body["max_completion_tokens"] == 100
        return io.BytesIO(
            json.dumps(
                {"choices": [{"finish_reason": "length", "message": {"content": ""}}]}
            ).encode()
        )

    monkeypatch.setattr(client, "urlopen", truncated)
    policy.open()
    with pytest.raises(ValueError, match="Incomplete"):
        policy.predict(prepare_observation(observation()), "Lift", None)
    assert policy._pending is None
    with pytest.raises(RuntimeError, match="never retried"):
        policy.predict(prepare_observation(observation()), "Lift", None)
    policy.close(None)
    invalid = client.OpenAICartesianPolicy(
        "private-test-key", tmp_path, model="test-model", extra_body={"messages": []}
    )
    with pytest.raises(ValueError, match="Unsupported"):
        invalid.open()


def test_provider_keepalives_do_not_extend_inference_deadline(tmp_path, monkeypatch):
    ticks = iter([0.0, 0.0, 91.0])
    monkeypatch.setattr(client.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(client, "urlopen", lambda *a, **kw: io.BytesIO(b" "))
    policy = client.OpenAICartesianPolicy("private-test-key", tmp_path, model="test-model")
    policy.open()
    with pytest.raises(TimeoutError, match="deadline"):
        policy.predict(prepare_observation(observation()), "Lift", None)
    assert policy._pending is None
    policy.close(None)


def test_geometry_measurements_precede_model_actions_without_ground_truth(tmp_path, monkeypatch):
    from dataclasses import replace

    obs = prepare_observation(observation())
    cal = {
        "reference_lens": dict(
            opticalCenter=[4, 4], nominalWidth=8, nominalHeight=8, k0=0, k1=0.01, k2=0, k3=0, k4=0
        ),
        "camera_position_flange_m": [0, 0, 0],
        "opencv_camera_orientation_flange_xyzw": [0, 0, 0, 1],
    }
    obs = replace(obs, metadata={"camera_calibration": cal})
    responses = iter(
        [
            {"measurements": [{"pixel_uv": [4, 4], "plane_z_m": 0.02}]},
            plan(),
        ]
    )
    requests = []

    def respond(request, **kwargs):
        body = json.loads(request.data)
        requests.append(body)
        p = next(responses)
        return io.BytesIO(
            json.dumps(
                {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(p)}}]}
            ).encode()
        )

    monkeypatch.setattr(client, "urlopen", respond)
    policy = client.OpenAICartesianPolicy("private-test-key", tmp_path, model="test-model")
    policy.open()
    chunk = policy.predict(obs, "Lift", None)
    assert chunk.steps == 30 and len(requests) == 2
    result = json.loads(requests[1]["messages"][-1]["content"])
    np.testing.assert_allclose(result["pixel_plane_results"][0]["task_xyz_m"], [0, 0, 0.02])
    assert result["depth_is_assumed_not_measured"] is True
    assert result["source_frame_id"] == obs.frame_id
    np.testing.assert_array_equal(chunk.channels["arm"][-1], plan()["waypoints"][0]["pose"])
    policy.close(ExecutionReceipt(chunk.chunk_id, 30, 30, ExecutionStatus.COMPLETED))
