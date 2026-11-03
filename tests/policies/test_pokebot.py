import json
import time

import numpy as np
import pytest

from robot_dock.policies.pokebot import PokebotPolicy
from robot_dock.protocol import ExecutionReceipt, ExecutionStatus, ImageFrame, Observation


class StubServerPolicy(PokebotPolicy):
    requests: list

    def _request(self, method, path, body=None):
        self.requests.append((method, path, body))
        if path == "/models":
            return {"data": [{"id": self.model}]}
        if path.endswith("schema"):
            return {"output": {}}
        if path == "/sessions":
            return {"id": "test-session", "next_turn": 1}
        if path.endswith("close"):
            return {"status": "closed", "messages": body["messages"]}
        index = sum(1 for item in self.requests if item[1] == "/chat/completions")
        return {
            "metadata": {"next_turn": str(index + 1), "turn": str(index)},
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": f"call-{index}",
                                "type": "function",
                                "function": {
                                    "name": "execute_actions",
                                    "arguments": json.dumps(
                                        {"output": [[0, 0, 0.4, 0, 1, 0, 0, 1]]}
                                    ),
                                },
                            }
                        ],
                    }
                }
            ],
        }


def observation():
    now = time.monotonic_ns()
    return Observation(
        1,
        now,
        {"arm.ee_pose": np.array([0, 0, 0.4, 0, 1, 0, 0]), "gripper.opening": np.array([1.0])},
        {"wrist": ImageFrame(np.zeros((4, 4, 3), dtype=np.uint8), "fisheye", now, "wrist")},
    )


def test_wire_feedback_contains_actual_partial_count(tmp_path):
    policy = StubServerPolicy("test-only", tmp_path)
    policy.requests = []
    policy.open()
    action = policy.predict(observation(), "move", None)
    receipt = ExecutionReceipt(action.chunk_id, 1, 0, ExecutionStatus.INTERRUPTED)
    policy.close(receipt)
    body = policy.requests[-1][2]
    assert json.loads(body["messages"][-1]["content"]) == {
        "executed_steps": 0,
        "status": "interrupted",
    }
    assert "test-only" not in "".join(p.read_text() for p in tmp_path.glob("*.json"))


def test_continuation_preserves_original_assistant_and_turn(tmp_path):
    policy = StubServerPolicy("test-only", tmp_path)
    policy.requests = []
    policy.open()
    first = policy.predict(observation(), "move", None)
    second = policy.predict(
        observation(), "move", ExecutionReceipt(first.chunk_id, 1, 1, ExecutionStatus.COMPLETED)
    )
    body = policy.requests[-1][2]
    assert [m["role"] for m in body["messages"]] == ["assistant", "tool", "user"]
    assert body["metadata"]["turn"] == "2"
    policy.close(ExecutionReceipt(second.chunk_id, 1, 1, ExecutionStatus.COMPLETED))


def test_unknown_execution_does_not_send_fabricated_count(tmp_path):
    policy = StubServerPolicy("test-only", tmp_path)
    policy.requests = []
    policy.open()
    action = policy.predict(observation(), "move", None)
    count = len(policy.requests)
    with pytest.raises(RuntimeError, match="unknown"):
        policy.close(ExecutionReceipt(action.chunk_id, 1, None, ExecutionStatus.UNKNOWN))
    assert len(policy.requests) == count


def test_raw_gripper_output_is_preserved_without_an_adapter_tolerance(tmp_path):
    policy = StubServerPolicy("test-only", tmp_path)
    policy.requests = []
    original = policy._request
    value = [-0.0012]

    def request(method, path, body=None):
        reply = original(method, path, body)
        if path == "/chat/completions":
            call = reply["choices"][0]["message"]["tool_calls"][0]
            data = json.loads(call["function"]["arguments"])
            data["output"][0][7] = value[0]
            call["function"]["arguments"] = json.dumps(data)
        return reply

    policy._request = request
    policy.open()
    first = policy.predict(observation(), "move", None)
    assert first.channels["gripper"][0, 0] == -0.0012
    prior = ExecutionReceipt(first.chunk_id, 1, 1, ExecutionStatus.COMPLETED)
    value[0] = -0.1
    second = policy.predict(observation(), "move", prior)
    assert second.channels["gripper"][0, 0] == -0.1
    policy.close(ExecutionReceipt(second.chunk_id, 1, 1, ExecutionStatus.COMPLETED))
    messages = policy.requests[-1][2]["messages"]
    assert messages[-1]["tool_call_id"] == "call-2"
    assert json.loads(messages[-1]["content"])["executed_steps"] == 1


@pytest.mark.parametrize("status", [401, 403, 500])
def test_http_failure_diagnostics_and_session_safety(tmp_path, monkeypatch, status):
    import io
    from email.message import Message
    from urllib.error import HTTPError

    from robot_dock.policies.pokebot.client import RoboAPIRequestError

    policy = StubServerPolicy("test-only-secret", tmp_path)
    policy.requests = []
    policy.open()
    original = policy._request
    attempts = []

    def reject(request, timeout):
        attempts.append(request)
        headers = Message()
        headers["X-Request-ID"] = "request-test"
        body = json.dumps(
            {"error": {"code": "access_denied", "message": "Denied test-only-secret"}}
        )
        raise HTTPError(request.full_url, status, "failure", headers, io.BytesIO(body.encode()))

    monkeypatch.setattr("robot_dock.policies.pokebot.client.urlopen", reject)

    def request(method, path, body=None):
        if path == "/chat/completions":
            return PokebotPolicy._request(policy, method, path, body)
        return original(method, path, body)

    policy._request = request
    with pytest.raises(RoboAPIRequestError, match=f"HTTP {status}") as error:
        policy.predict(observation(), "move", None)
    assert "access_denied" in str(error.value)
    assert "request-test" in str(error.value)
    assert policy.api_key not in str(error.value)
    assert len(attempts) == 1
    if status in (401, 403):
        policy.close(None)
        assert policy.requests[-1][1].endswith("/close")
        assert not (tmp_path / "reconciliation-required.json").exists()
    else:
        with pytest.raises(RuntimeError, match="Ambiguous inference"):
            policy.close(None)


def test_timeout_retains_uncertain_session_without_retry(tmp_path):
    policy = StubServerPolicy("test-only", tmp_path)
    policy.requests = []
    policy.open()
    calls = []

    def timeout(method, path, body=None):
        calls.append(path)
        raise TimeoutError("request timed out")

    policy._request = timeout
    with pytest.raises(TimeoutError):
        policy.predict(observation(), "move", None)
    with pytest.raises(RuntimeError, match="Ambiguous inference"):
        policy.close(None)
    assert calls == ["/chat/completions"]
