from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ExecutionStatus(StrEnum):
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ExecutionReceipt:
    chunk_id: str
    requested_steps: int
    dispatched_steps: int | None
    status: ExecutionStatus
    reason: str | None = None

    def __post_init__(self) -> None:
        count = self.dispatched_steps
        if self.requested_steps < 1 or (
            count is not None and not 0 <= count <= self.requested_steps
        ):
            raise ValueError("Invalid execution count")
        if self.status == ExecutionStatus.COMPLETED and count != self.requested_steps:
            raise ValueError("Completed means every requested step was dispatched")
        if self.status != ExecutionStatus.UNKNOWN and count is None:
            raise ValueError("Known execution states require an actual dispatched count")


class ExecutionError(RuntimeError):
    def __init__(self, receipt: ExecutionReceipt):
        super().__init__(receipt.reason or "Execution failed")
        self.receipt = receipt


class RunStatus(StrEnum):
    SUCCEEDED = "succeeded"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CANCELLED = "cancelled"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    status: RunStatus
    task_success: bool | None
    chunks: int
    dispatched_steps: int
    output_dir: Path
    error: str | None = None
    cleanup_errors: tuple[str, ...] = ()
