import json
from dataclasses import replace
from threading import Event

import pytest

from robot_dock import RunConfig, run
from robot_dock.bindings import PolicyBinding
from robot_dock.protocol import ExecutionError, ExecutionReceipt, ExecutionStatus, RunStatus


def test_task_success_requires_a_model_chunk_even_if_initial_scene_satisfies_evaluator(
    tmp_path, pair
):
    policy, backend = pair
    result = run(
        policy=policy,
        backend=backend,
        config=RunConfig(tmp_path / "run", "move"),
        evaluate=lambda obs: True,
    )
    assert result.status == RunStatus.SUCCEEDED
    assert policy.calls == 1 and result.chunks == 1 and result.dispatched_steps == 4


def test_loop_observes_actual_state_and_closes_with_final_receipt(tmp_path, pair):
    policy, backend = pair
    result = run(
        policy=policy,
        backend=backend,
        config=RunConfig(tmp_path / "run", "move"),
        evaluate=lambda obs: bool(obs.state["count"][0] >= 8),
    )
    assert result.status == RunStatus.SUCCEEDED
    assert result.dispatched_steps == 8 and policy.calls == 2
    assert policy.feedback[0] is None and policy.feedback[1].dispatched_steps == 4
    assert policy.closed_feedback.dispatched_steps == 4
    assert backend.stopped and backend.closed and policy.closed
    events = [json.loads(line) for line in (tmp_path / "run/events.jsonl").read_text().splitlines()]
    assert [e["sequence"] for e in events] == list(range(len(events)))
    assert events[-1]["data"]["status"] == "succeeded"


def test_inference_success_is_not_task_success(tmp_path, pair):
    policy, backend = pair
    result = run(
        policy=policy, backend=backend, config=RunConfig(tmp_path / "run", "move", max_chunks=1)
    )
    assert result.status == RunStatus.BUDGET_EXHAUSTED and result.task_success is None


def test_partial_cancel_is_reported_without_replaying(tmp_path, pair):
    policy, backend = pair

    def execute(chunk, cancel, *, record_step=None):
        backend.executions += 1
        return ExecutionReceipt(
            chunk.chunk_id, chunk.steps, 2, ExecutionStatus.INTERRUPTED, "cancelled"
        )

    backend.execute = execute
    result = run(policy=policy, backend=backend, config=RunConfig(tmp_path / "run", "move"))
    assert result.status == RunStatus.CANCELLED and result.dispatched_steps == 2
    assert backend.executions == policy.calls == 1
    assert policy.closed_feedback.dispatched_steps == 2


def test_partial_failure_preserves_count(tmp_path, pair):
    policy, backend = pair

    def execute(chunk, cancel, *, record_step=None):
        raise ExecutionError(
            ExecutionReceipt(
                chunk.chunk_id, chunk.steps, 1, ExecutionStatus.FAILED, "controller fault"
            )
        )

    backend.execute = execute
    result = run(policy=policy, backend=backend, config=RunConfig(tmp_path / "run", "move"))
    assert result.status == RunStatus.FAILED and result.dispatched_steps == 1
    assert policy.closed_feedback.reason == "controller fault"


def test_unmeasured_dispatch_failure_is_unknown(tmp_path, pair):
    policy, backend = pair

    def execute(chunk, cancel, *, record_step=None):
        raise ConnectionError("controller disconnected after dispatch")

    backend.execute = execute
    result = run(policy=policy, backend=backend, config=RunConfig(tmp_path / "run", "move"))
    assert result.status == RunStatus.UNKNOWN
    assert policy.closed_feedback.dispatched_steps is None
    assert policy.calls == 1


def test_cancel_during_inference_never_dispatches(tmp_path, pair):
    policy, backend = pair
    cancel = Event()
    predict = policy.predict

    def cancelled_predict(*args):
        response = predict(*args)
        cancel.set()
        return response

    policy.predict = cancelled_predict
    result = run(
        policy=policy, backend=backend, cancel=cancel, config=RunConfig(tmp_path / "run", "move")
    )
    assert result.status == RunStatus.CANCELLED and backend.executions == 0
    assert policy.closed_feedback.dispatched_steps == 0


def test_rejects_stale_state_before_inference(tmp_path, pair):
    policy, backend = pair
    observe = backend.observe
    backend.observe = lambda: replace(observe(), captured_at_ns=0)
    result = run(
        policy=policy,
        backend=backend,
        config=RunConfig(tmp_path / "run", "move", max_observation_age_s=2),
    )
    assert result.status == RunStatus.FAILED and policy.calls == 0
    assert "stale" in result.error


def test_observation_age_limit_is_opt_in(tmp_path, pair):
    policy, backend = pair
    observe = backend.observe
    backend.observe = lambda: replace(observe(), captured_at_ns=0)
    result = run(
        policy=policy, backend=backend, config=RunConfig(tmp_path / "run", "move", max_chunks=1)
    )
    assert result.error is None and policy.calls == 1


def test_binding_cannot_change_action_identity(tmp_path, pair):
    policy, backend = pair
    binding = PolicyBinding(prepare_actions=lambda chunk: replace(chunk, chunk_id="wrong"))
    result = run(
        policy=policy, backend=backend, binding=binding, config=RunConfig(tmp_path / "run", "move")
    )
    assert result.status == RunStatus.FAILED and backend.executions == 0
    assert policy.closed_feedback.dispatched_steps == 0


def test_action_trace_is_ordered_before_its_execution_receipt(tmp_path, pair):
    policy, backend = pair
    original = backend.execute

    def execute(chunk, cancel, *, record_step=None):
        return original(chunk, cancel, record_step=record_step)

    backend.execute = execute
    result = run(
        policy=policy, backend=backend, config=RunConfig(tmp_path / "run", "move", max_chunks=1)
    )
    assert result.error is None
    events = [json.loads(line) for line in (tmp_path / "run/events.jsonl").read_text().splitlines()]
    kinds = [event["kind"] for event in events]
    assert kinds.index("execution_planned") < kinds.index("execution_receipt")


def test_unknown_receipt_identity_is_never_accepted(tmp_path, pair):
    policy, backend = pair
    backend.execute = lambda chunk, cancel, **kwargs: ExecutionReceipt(
        "unrelated", chunk.steps, chunk.steps, ExecutionStatus.COMPLETED
    )
    result = run(policy=policy, backend=backend, config=RunConfig(tmp_path / "run", "move"))
    assert result.status == RunStatus.UNKNOWN and result.dispatched_steps == 0


def test_open_failure_still_releases_partial_resources(tmp_path, pair):
    policy, backend = pair

    def broken_open():
        raise RuntimeError("device open failed midway")

    backend.open = broken_open
    result = run(policy=policy, backend=backend, config=RunConfig(tmp_path / "run", "move"))
    assert result.status == RunStatus.FAILED and backend.stopped and backend.closed


def test_cleanup_failure_does_not_hide_task_result(tmp_path, pair):
    policy, backend = pair

    def broken_close(feedback):
        raise ConnectionError("cloud close failed")

    policy.close = broken_close
    result = run(
        policy=policy,
        backend=backend,
        config=RunConfig(tmp_path / "run", "move"),
        evaluate=lambda obs: bool(obs.state["count"][0] >= 4),
    )
    assert result.task_success is True and backend.closed
    assert "cloud close failed" in result.cleanup_errors[0]


def test_existing_run_directory_is_never_overwritten(tmp_path, pair):
    policy, backend = pair
    config = RunConfig(tmp_path / "run", "move", max_chunks=1)
    run(policy=policy, backend=backend, config=config)
    with pytest.raises(FileExistsError):
        run(policy=policy, backend=backend, config=config)


def test_caller_owned_world_survives_two_tasks_without_reopening(tmp_path, pair):
    policy, backend = pair

    def forbidden():
        pytest.fail("A borrowed backend must not be opened or closed by run()")

    backend.open = forbidden
    backend.close = forbidden
    for index in range(2):
        result = run(
            policy=policy,
            backend=backend,
            config=RunConfig(tmp_path / str(index), "move", max_chunks=1),
            manage_backend=False,
        )
        assert result.error is None and not result.cleanup_errors
        assert backend.stopped and policy.closed
    assert backend.count == 8 and backend.executions == 2


def test_logging_reports_chunk_progress_and_failure_without_replacing_records(
    tmp_path, pair, caplog
):
    policy, backend = pair
    caplog.set_level("INFO", logger="robot_dock.runtime.runner")
    original = backend.execute

    def execute(chunk, cancel, *, record_step=None):
        if backend.executions:
            raise ExecutionError(
                ExecutionReceipt(chunk.chunk_id, chunk.steps, 0, ExecutionStatus.FAILED, "blocked")
            )
        return original(chunk, cancel, record_step=record_step)

    backend.execute = execute
    result = run(policy=policy, backend=backend, config=RunConfig(tmp_path / "run", "move"))
    assert result.status == RunStatus.FAILED
    messages = [record for record in caplog.records if record.name == "robot_dock.runtime.runner"]
    assert any(r.levelname == "INFO" and "total_steps=4" in r.message for r in messages)
    assert any(r.levelname == "ERROR" and "status=failed" in r.message for r in messages)
    assert any(r.levelname == "ERROR" and "reason=blocked" in r.message for r in messages)
    assert json.loads((tmp_path / "run/summary.json").read_text())["status"] == "failed"
