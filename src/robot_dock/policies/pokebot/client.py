"""RoboAPI session adapter. Only this module knows the vendor wire protocol."""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from robot_dock.protocol import (
    ActionChunk,
    ActionSpec,
    ExecutionReceipt,
    ExecutionStatus,
    Observation,
)


class RoboAPIRequestError(RuntimeError):
    """An explicit HTTP failure, with credential-free diagnostic context."""

    def __init__(self, status: int, detail: str) -> None:
        self.status = status
        super().__init__(detail)


@dataclass
class PokebotPolicy:
    api_key: str = field(repr=False)
    output_dir: Path
    model: str = "poke-aurora-preview"
    base_url: str = "https://api.pokebot.com/v1"
    timeout_s: float = 90
    _session: dict | None = field(default=None, init=False, repr=False)
    _pending: tuple[dict, str] | None = field(default=None, init=False, repr=False)
    _turn: str = field(default="1", init=False)
    _uncertain: bool = field(default=False, init=False)
    _acknowledged_chunk: str | None = field(default=None, init=False)

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        request = Request(
            self.base_url.rstrip("/") + path,
            data=None if body is None else json.dumps(body, allow_nan=False).encode(),
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"},
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                result = json.load(response)
                result["request_id"] = response.headers.get("X-Request-ID") or result.get(
                    "request_id"
                )
                return result
        except HTTPError as exc:
            detail = f"RoboAPI {method} {path}: HTTP {exc.code}"
            try:
                payload = json.loads(exc.read(16384))
                error = payload.get("error", {}) if isinstance(payload, dict) else {}
                if isinstance(error, dict):
                    for name in ("code", "message", "request_id"):
                        value = error.get(name)
                        if isinstance(value, str):
                            detail += f" | {name}={value[:1000]}"
                request_id = exc.headers.get("X-Request-ID") if exc.headers else None
                if request_id:
                    detail += f" | X-Request-ID={request_id}"
            except (ValueError, OSError):
                pass
            finally:
                exc.close()
            # Never serialize the request, authorization headers, or raw body.
            detail = detail.replace(self.api_key, "[redacted]")
            detail = " ".join(detail.splitlines())
            raise RoboAPIRequestError(exc.code, detail) from None

    def _save(self, name: str, value: dict) -> None:
        (self.output_dir / name).write_text(json.dumps(value, indent=2), encoding="utf-8")

    def open(self) -> None:
        if not self.api_key.strip():
            raise ValueError("RoboAPI key is empty")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        available = self._request("GET", "/models")["data"]
        if self.model not in [item["id"] for item in available]:
            raise ValueError(f"Requested model unavailable: {self.model}")
        self._save(
            "schema.json", self._request("GET", f"/models/{quote(self.model, safe='')}/schema")
        )
        self._session = self._request("POST", "/sessions", {"model": self.model})
        self._turn = str(self._session["next_turn"])
        self._save("session.json", self._session)

    def _feedback(self, receipt: ExecutionReceipt | None) -> list[dict]:
        # A predict() failure may occur after the previous receipt was accepted
        # and a new response arrived. Close the new, undispatched action with 0.
        if receipt is not None and receipt.chunk_id == self._acknowledged_chunk:
            receipt = None
        if self._pending is None:
            if receipt is not None:
                raise ValueError("Receipt supplied without a pending model action")
            return []
        message, chunk_id = self._pending
        if receipt is not None and receipt.chunk_id != chunk_id:
            raise ValueError("Execution receipt belongs to a different action chunk")
        if receipt is not None and receipt.status == ExecutionStatus.UNKNOWN:
            raise RuntimeError("Execution is unknown; retain session for manual reconciliation")
        executed = 0 if receipt is None else receipt.dispatched_steps
        completed = receipt is not None and receipt.status == ExecutionStatus.COMPLETED
        return [
            message,
            {
                "role": "tool",
                "tool_call_id": chunk_id,
                "content": json.dumps(
                    {
                        "executed_steps": executed,
                        "status": "completed" if completed else "interrupted",
                    }
                ),
            },
        ]

    def predict(
        self, observation: Observation, instruction: str, feedback: ExecutionReceipt | None
    ) -> ActionChunk:
        if self._session is None or self._uncertain:
            raise RuntimeError("Policy session is not ready")
        pose = observation.state["arm.ee_pose"]
        opening = observation.state["gripper.opening"]
        if pose.shape != (7,) or opening.shape != (1,):
            raise ValueError("RoboAPI requires a 7D flange pose and 1D opening")
        image = observation.images["wrist"]
        if image.projection != "fisheye":
            raise ValueError("RoboAPI reference binding requires a wrist fisheye image")
        buffer = io.BytesIO()
        Image.fromarray(image.pixels).save(buffer, format="PNG")
        messages = self._feedback(feedback) + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {"instruction": instruction, "state": np.r_[pose, opening].tolist()}
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + base64.b64encode(buffer.getvalue()).decode()
                        },
                    },
                ],
            }
        ]
        # Once sent, an ambiguous response must not be retried automatically.
        self._uncertain = True
        try:
            response = self._request(
                "POST",
                "/chat/completions",
                {
                    "model": self.model,
                    "metadata": {"session_id": self._session["id"], "turn": self._turn},
                    "messages": messages,
                },
            )
        except RoboAPIRequestError as exc:
            if exc.status in (401, 403):
                # Authentication/access rejection produced no new model actions.
                # Retain prior pending feedback for the final close request.
                self._uncertain = False
            raise
        self._save(f"response-{self._turn}.json", response)
        if self._pending is not None:
            self._acknowledged_chunk = self._pending[1]
        self._uncertain = False
        self._pending = None
        self._turn = str(response["metadata"]["next_turn"])
        message = response["choices"][0]["message"]
        calls = message.get("tool_calls", [])
        if (
            len(calls) != 1
            or calls[0].get("type") != "function"
            or calls[0]["function"]["name"] != "execute_actions"
        ):
            raise ValueError("Expected a single execute_actions call")
        call = calls[0]
        self._pending = (message, call["id"])
        actions = np.asarray(json.loads(call["function"]["arguments"])["output"], dtype=float)
        if actions.ndim != 2 or actions.shape[1] != 8:
            raise ValueError("Expected Tx8 actions")
        return ActionChunk(
            channels={"arm": actions[:, :7], "gripper": actions[:, 7:]},
            specs={
                "arm": ActionSpec(
                    "end_effector_pose", 7, frame="task", control_point="flange", unit="m_xyzw"
                ),
                "gripper": ActionSpec("gripper_position", 1, unit="normalized_opening"),
            },
            period_s=1 / 30,
            source_frame_id=observation.frame_id,
            chunk_id=call["id"],
            metadata={
                "request_id": response.get("request_id"),
                "model": response.get("model"),
                "turn": response["metadata"]["turn"],
            },
        )

    def close(self, feedback: ExecutionReceipt | None) -> None:
        if self._session is None:
            return
        if self._uncertain:
            self._save(
                "reconciliation-required.json",
                {
                    "session_id": self._session["id"],
                    "turn": self._turn,
                    "reason": "Inference response uncertain. Query session/turn before further actions.",
                },
            )
            raise RuntimeError("Ambiguous inference; session retained for reconciliation")
        closed = self._request(
            "POST",
            f"/sessions/{quote(self._session['id'], safe='')}/close",
            {"reason": "stopped", "messages": self._feedback(feedback)},
        )
        self._save("session-closed.json", closed)
        if closed.get("status") != "closed":
            raise RuntimeError("Server did not confirm session closure")
        self._session = None
        self._pending = None
