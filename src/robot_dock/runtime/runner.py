from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable
from uuid import uuid4

from robot_dock.bindings import PolicyBinding
from robot_dock.policies.protocol import Policy
from robot_dock.protocol import (
    ExecutionError,
    ExecutionReceipt,
    ExecutionStatus,
    Observation,
    RunOutcome,
    RunStatus,
)
from robot_dock.recording import RunRecorder
from robot_dock.robots.protocol import RobotBackend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunConfig:
    output_dir: Path
    instruction: str
    max_chunks: int = 30
    max_wall_time_s: float = 300
    max_observation_age_s: float | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        import math

        if not self.instruction.strip() or self.max_chunks < 1:
            raise ValueError("An instruction and positive chunk budget are required")
        limits = [self.max_wall_time_s]
        if self.max_observation_age_s is not None:
            limits.append(self.max_observation_age_s)
        if not all(math.isfinite(v) and v > 0 for v in limits):
            raise ValueError("Time limits must be finite and positive")


def run(
    *,
    policy: Policy,
    backend: RobotBackend,
    config: RunConfig,
    binding: PolicyBinding | None = None,
    evaluate: Callable[[Observation], bool | None] | None = None,
    cancel: Event | None = None,
    manage_backend: bool = True,
) -> RunOutcome:
    """One synchronous loop; no retries, task planner, or hidden provider branches.

    Wall-time budget is checked between blocking calls; adapters own call timeouts.
    A backend must raise ExecutionError with a measured partial receipt when known.
    Other dispatch exceptions are UNKNOWN and never automatically replayed.
    With manage_backend=False, the caller owns an already-open backend and must
    close it. Each task still stops motion and closes its own policy session.
    """
    binding = binding or PolicyBinding()
    cancel = cancel or Event()
    recorder = RunRecorder(config.output_dir)
    run_id = config.run_id or uuid4().hex
    recorder.event(
        "run_started",
        {
            "run_id": run_id,
            "config": config,
            "binding": binding.name,
            "policy_type": type(policy).__name__,
            "backend_type": type(backend).__name__,
        },
    )
    started = time.monotonic()
    logger.info(
        "Run started | run_id=%s | max_chunks=%d | max_wall_time_s=%g",
        run_id,
        config.max_chunks,
        config.max_wall_time_s,
    )
    status = RunStatus.BUDGET_EXHAUSTED
    task_success = None
    error = None
    cleanup_errors = []
    feedback = None
    chunks = 0
    dispatched = 0
    backend_attempted = False
    policy_attempted = False
    try:
        backend_attempted = True
        if manage_backend:
            backend.open()
        policy_attempted = True
        policy.open()
        observation = backend.observe()
        for _ in range(config.max_chunks):
            recorder.observation(observation)
            if cancel.is_set():
                status = RunStatus.CANCELLED
                break
            if chunks and evaluate is not None:
                task_success = evaluate(observation)
                if task_success is True:
                    status = RunStatus.SUCCEEDED
                    break
            if time.monotonic() - started >= config.max_wall_time_s:
                break
            age = (time.monotonic_ns() - observation.captured_at_ns) / 1e9
            if age < 0:
                raise ValueError("Observation timestamp is from a different clock")
            if config.max_observation_age_s is not None and age > config.max_observation_age_s:
                raise ValueError(
                    "Observation timestamp is stale or from a different clock "
                    f"(age={age:.3f}s, limit={config.max_observation_age_s:.3f}s)"
                )
            model_observation = binding.prepare_observation(observation)
            recorder.event("inference_started", {"frame_id": model_observation.frame_id})
            logger.debug(
                "Inference started | run_id=%s | frame_id=%s", run_id, observation.frame_id
            )
            chunk = policy.predict(model_observation, config.instruction, feedback)
            feedback = None
            recorder.event("policy_action", chunk)
            if chunk.source_frame_id != observation.frame_id:
                raise ValueError("Action was produced for a different observation")
            feedback = ExecutionReceipt(
                chunk.chunk_id, chunk.steps, 0, ExecutionStatus.INTERRUPTED, "not dispatched"
            )
            physical = binding.prepare_actions(chunk)
            if (
                physical.chunk_id != chunk.chunk_id
                or physical.steps != chunk.steps
                or physical.source_frame_id != chunk.source_frame_id
            ):
                raise ValueError("Binding must preserve chunk identity, frame, and step count")
            if dict(physical.specs) != dict(backend.action_specs):
                raise ValueError("Action specs do not match the backend contract")
            recorder.event("execution_planned", physical)
            if cancel.is_set() or time.monotonic() - started >= config.max_wall_time_s:
                recorder.event("execution_receipt", feedback)
                status = RunStatus.CANCELLED if cancel.is_set() else RunStatus.BUDGET_EXHAUSTED
                break
            try:
                feedback = backend.execute(physical, cancel, record_step=recorder.step)
            except ExecutionError as exc:
                feedback = exc.receipt
            except BaseException:
                feedback = ExecutionReceipt(
                    chunk.chunk_id,
                    chunk.steps,
                    None,
                    ExecutionStatus.UNKNOWN,
                    "dispatch raised without a measured receipt",
                )
                recorder.event("execution_receipt", feedback)
                status = RunStatus.UNKNOWN
                raise
            if feedback.chunk_id != chunk.chunk_id or feedback.requested_steps != chunk.steps:
                feedback = ExecutionReceipt(
                    chunk.chunk_id,
                    chunk.steps,
                    None,
                    ExecutionStatus.UNKNOWN,
                    "backend returned an unrelated receipt",
                )
            recorder.event("execution_receipt", feedback)
            chunks += 1
            dispatched += feedback.dispatched_steps or 0
            logger.info(
                "Chunk %d | run_id=%s | status=%s | dispatched=%s/%d | total_steps=%d",
                chunks,
                run_id,
                feedback.status,
                feedback.dispatched_steps,
                feedback.requested_steps,
                dispatched,
            )
            if feedback.status != ExecutionStatus.COMPLETED:
                status = (
                    RunStatus.UNKNOWN
                    if feedback.status == ExecutionStatus.UNKNOWN
                    else RunStatus.CANCELLED
                    if feedback.status == ExecutionStatus.INTERRUPTED
                    else RunStatus.FAILED
                )
                error = feedback.reason
                break
            observation = backend.observe()
        else:
            recorder.observation(observation)
            if evaluate is not None:
                task_success = evaluate(observation)
                if task_success is True:
                    status = RunStatus.SUCCEEDED
    except KeyboardInterrupt:
        if status != RunStatus.UNKNOWN:
            status = RunStatus.CANCELLED
        error = "keyboard interrupt"
    except Exception as exc:
        if status != RunStatus.UNKNOWN:
            status = RunStatus.FAILED
        error = f"{type(exc).__name__}: {exc}"
        recorder.event("run_error", {"error": error})
    finally:
        operations = []
        if backend_attempted:
            operations.append(("backend.stop", backend.stop))
        if policy_attempted:
            operations.append(("policy.close", lambda: policy.close(feedback)))
        if backend_attempted:
            operations.append(("backend.finish_recording", backend.finish_recording))
        if backend_attempted and manage_backend:
            operations.append(("backend.close", backend.close))
        for name, operation in operations:
            try:
                operation()
            except Exception as exc:
                cleanup_errors.append(f"{name}: {type(exc).__name__}: {exc}")
        try:
            recorder.flush()
        except Exception as exc:
            cleanup_errors.append(f"recording: {type(exc).__name__}: {exc}")
        outcome = RunOutcome(
            run_id,
            status,
            task_success,
            chunks,
            dispatched,
            config.output_dir,
            error,
            tuple(cleanup_errors),
        )
        recorder.event("run_finished", outcome)
        recorder.summary(outcome)
        level = (
            logging.ERROR
            if status in (RunStatus.FAILED, RunStatus.UNKNOWN) or cleanup_errors
            else logging.INFO
            if status == RunStatus.SUCCEEDED
            else logging.WARNING
        )
        logger.log(
            level,
            "Run finished | run_id=%s | status=%s | task_success=%s | chunks=%d | "
            "dispatched_steps=%d | output=%s",
            run_id,
            status,
            task_success,
            chunks,
            dispatched,
            config.output_dir,
        )
        if error:
            logger.log(level, "Run stopped | run_id=%s | reason=%s", run_id, error)
        for cleanup_error in cleanup_errors:
            logger.error("Cleanup failed | run_id=%s | reason=%s", run_id, cleanup_error)
    return outcome
