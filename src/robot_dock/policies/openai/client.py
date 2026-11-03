"""Remote vision planner; generic Cartesian interpolation, no task trajectories."""

from __future__ import annotations

import base64
import io
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from robot_dock.protocol import (
    ActionChunk,
    ActionSpec,
    ExecutionReceipt,
    ExecutionStatus,
    Observation,
)

from .geometry import pixel_to_plane

SYSTEM_PROMPT = """You control a UR5e with a Robotiq 2F-85 parallel gripper and wrist fisheye RGB.
Plan the next short Cartesian motion from the CURRENT observed image and measured state.
All poses are ABSOLUTE flange poses [x,y,z,qx,qy,qz,qw] in the task frame, meters,
unit quaternion XYZW. Task +Z is up. The gripper extends along flange local +Z;
The input gives the actual flange-to-pinch offset and measured pinch-center pose.
To place the pinch at P with orientation R, target flange position P - R * offset.
The current flange position is NOT the grasp point. Descend the PINCH toward the
object before closing; a fully closed empty gripper is not evidence of a grasp.
Opening is 0 closed, 1 open. Input camera extrinsics use OpenCV right/down/forward
axes. The F-theta lens polynomial maps pixel radius to ray angle in radians.
Use the image and calibration to estimate the scene; object ground truth is unavailable.
The workspace calibration, if provided, gives the measured support-plane height.
Do not perform camera/quaternion arithmetic by guessing. A read-only geometry tool
is available: instead of waypoints return JSON {"measurements":[{"pixel_uv":[u,v],
"plane_z_m":z}]}. It intersects each ORIGINAL image pixel ray with your selected
task-frame horizontal plane and returns exact coordinates. This does NOT measure
depth: explicitly choose a plane (support surface, estimated object height, etc.).
Use this tool to ground target locations before the first approach, and remeasure
after failed grasps or camera movements. Up to six pixels per call and three tool
calls per observation are available. Tool calls do not move the robot. If a clicked
point is above the assumed plane, its projection will have parallax error.
Use measured geometry to choose YOUR own Cartesian waypoints. Do not close and
lift in the same first approach chunk: re-observe the object between the fingers
before deciding to close. After closing, visually verify before lifting substantially.
The previous action-input RGB and current action-result RGB are labelled separately.
Execution completed means commands were dispatched, not that the object was grasped.
Account for tool length and orientation. Make short motions and re-observe when uncertain.
Return ONLY a JSON object {"waypoints":[{"pose":[x,y,z,qx,qy,qz,qw],
"opening":g,"duration_s":t}]}. Each waypoint specifies the endpoint after t seconds
from the previous waypoint (first starts at measured state). Position and opening
interpolate linearly; rotation uses shortest-path SLERP. For a hold, repeat the pose.
Use 1 to 8 waypoints, positive durations in multiples of 1/30 s, total <=6 s.
Choose physically modest speeds and avoid collisions. Gripper commands also interpolate,
so allocate separate stationary closure/release intervals when needed. Never assume a
previous action succeeded: use the execution receipt and new image. Continue holding or
withdraw as the instruction requests. Do not report task success; an external evaluator
decides it. There is no automatic retry, fallback, object teleportation or target correction.
"""


def interpolate_plan(plan: dict, start: NDArray[np.float64]) -> NDArray[np.float64]:
    """Validate remote waypoints and sample at 30 Hz without modifying endpoints."""
    points = plan.get("waypoints")
    if not isinstance(points, list) or not 1 <= len(points) <= 8:
        raise ValueError("Expected 1..8 Cartesian waypoints")
    previous = np.asarray(start, dtype=float)
    if previous.shape != (8,) or not np.isfinite(previous).all():
        raise ValueError("Expected finite measured 8D start state")
    rows: list[NDArray[np.float64]] = []
    for point in points:
        if not isinstance(point, dict) or set(point) != {"pose", "opening", "duration_s"}:
            raise ValueError("Invalid waypoint fields")
        pose = np.asarray(point["pose"], dtype=float)
        opening, duration = point["opening"], point["duration_s"]
        if (
            pose.shape != (7,)
            or not np.isfinite(pose).all()
            or not np.isclose(np.linalg.norm(pose[3:]), 1, atol=1e-6)
            or isinstance(opening, bool)
            or not isinstance(opening, (int, float))
            or not math.isfinite(opening)
            or not 0 <= opening <= 1
            or isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or not 1 / 30 <= duration <= 6
        ):
            raise ValueError("Invalid pose, quaternion, opening or duration")
        steps = round(duration * 30)
        if not math.isclose(steps, duration * 30, abs_tol=1e-5):
            raise ValueError("Durations must be multiples of 1/30 s")
        if len(rows) + steps > 180:
            raise ValueError("Plan exceeds six seconds")
        target = np.r_[pose, opening]
        qa, qb = previous[3:7], pose[3:].copy()
        dot = float(np.dot(qa, qb))
        if dot < 0:
            qb = -qb
            dot = -dot
        theta = math.acos(float(np.clip(dot, -1, 1)))
        for index in range(1, steps + 1):
            alpha = index / steps
            row = previous + alpha * (target - previous)
            if theta < 1e-6:
                q = (1 - alpha) * qa + alpha * qb
                row[3:7] = q / np.linalg.norm(q)
            else:
                row[3:7] = (
                    math.sin((1 - alpha) * theta) * qa + math.sin(alpha * theta) * qb
                ) / math.sin(theta)
            # Preserve the model's exact endpoint, including quaternion sign.
            rows.append(target.copy() if index == steps else row)
        previous = target
    return np.asarray(rows)


@dataclass
class OpenAICartesianPolicy:
    api_key: str = field(repr=False)
    output_dir: Path
    model: str = ""
    base_url: str = "https://api.openai.com/v1"
    timeout_s: float = 90
    extra_body: dict = field(default_factory=dict)
    _ready: bool = field(default=False, init=False)
    _turn: int = field(default=0, init=False)
    _pending: ActionChunk | None = field(default=None, init=False, repr=False)
    _last_plan: dict | None = field(default=None, init=False, repr=False)
    _last_image: dict | None = field(default=None, init=False, repr=False)

    def open(self) -> None:
        if not self.api_key.strip():
            raise ValueError("OpenAI-compatible API key is empty; set OPENAI_API_KEY")
        if not self.model.strip():
            raise ValueError("Model is empty; set OPENAI_API_MODEL")
        allowed_options = {
            "thinking",
            "reasoning_effort",
            "temperature",
            "max_tokens",
            "max_completion_tokens",
        }
        if not isinstance(self.extra_body, dict) or set(self.extra_body) - allowed_options:
            raise ValueError("Unsupported OPENAI_API_EXTRA_BODY fields")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ready = True

    def _save(self, name: str, value: dict) -> None:
        (self.output_dir / name).write_text(
            json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
        )

    def _receipt(self, feedback: ExecutionReceipt | None) -> dict | None:
        if self._pending is None:
            if feedback is not None:
                raise ValueError("Receipt supplied without a pending plan")
            return None
        if feedback is None or feedback.chunk_id != self._pending.chunk_id:
            raise ValueError("Expected receipt for the pending plan")
        if feedback.status == ExecutionStatus.UNKNOWN:
            raise RuntimeError("Unknown execution; reconcile before further inference")
        return {
            "chunk_id": feedback.chunk_id,
            "requested_steps": feedback.requested_steps,
            "dispatched_steps": feedback.dispatched_steps,
            "status": feedback.status.value,
        }

    def predict(
        self, observation: Observation, instruction: str, feedback: ExecutionReceipt | None
    ) -> ActionChunk:
        if not self._ready:
            raise RuntimeError("Planner is not ready; failed inference is never retried")
        receipt = self._receipt(feedback)
        start = np.r_[observation.state["arm.ee_pose"], observation.state["gripper.opening"]]
        image = observation.images["wrist"]
        if image.projection != "fisheye":
            raise ValueError("Expected wrist fisheye RGB")
        buffer = io.BytesIO()
        Image.fromarray(image.pixels).save(buffer, format="PNG")
        body: dict = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "max_tokens": 8192,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "instruction": instruction,
                                    "flange_pose_and_opening": start.tolist(),
                                    "camera_calibration": observation.metadata.get(
                                        "camera_calibration", {}
                                    ),
                                    "robot_geometry": observation.metadata.get(
                                        "robot_geometry", {}
                                    ),
                                    "workspace_calibration": observation.metadata.get(
                                        "workspace_calibration", {}
                                    ),
                                    "measured_pinch_pose": observation.state.get(
                                        "gripper.center_pose", np.array([])
                                    ).tolist(),
                                    "measured_jaw_width_m": observation.state.get(
                                        "gripper.width_m", np.array([])
                                    ).tolist(),
                                    "previous_plan": self._last_plan,
                                    "execution_receipt": receipt,
                                }
                            ),
                        },
                        {
                            "type": "text",
                            "text": "CURRENT ACTION RESULT RGB (or initial RGB on the first turn)",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                + base64.b64encode(buffer.getvalue()).decode()
                            },
                        },
                    ],
                },
            ],
        }
        body.update(self.extra_body)
        if "max_completion_tokens" in self.extra_body:
            body.pop("max_tokens", None)
        current_image = body["messages"][1]["content"][-1]
        if self._last_image is not None:
            body["messages"][1]["content"][1:1] = [
                {"type": "text", "text": "PREVIOUS ACTION INPUT RGB; compare with current result"},
                self._last_image,
            ]
        self._ready = False
        for measurement_round in range(4):
            result = self._complete(body)
            choice = result["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("Incomplete Cartesian plan; refusing execution")
            plan = json.loads(choice["message"]["content"])
            if "measurements" not in plan:
                break
            if measurement_round == 3 or set(plan) != {"measurements"}:
                raise ValueError("Measurement budget exhausted or ambiguous response")
            points = plan["measurements"]
            if not isinstance(points, list) or not 1 <= len(points) <= 6:
                raise ValueError("Expected 1..6 pixel-plane measurements")
            measurements = []
            for point in points:
                if not isinstance(point, dict) or set(point) != {"pixel_uv", "plane_z_m"}:
                    raise ValueError("Invalid measurement fields")
                xyz = pixel_to_plane(
                    point["pixel_uv"],
                    point["plane_z_m"],
                    start[:7],
                    observation.metadata.get("camera_calibration", {}),
                    image.pixels.shape,
                )
                measurements.append({**point, "task_xyz_m": xyz})
            body["messages"].extend(
                [
                    {"role": "assistant", "content": choice["message"]["content"]},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "pixel_plane_results": measurements,
                                "depth_is_assumed_not_measured": True,
                                "source_frame_id": observation.frame_id,
                                "remaining_measurement_calls": 2 - measurement_round,
                            }
                        ),
                    },
                ]
            )
        actions = interpolate_plan(plan, start)
        chunk = ActionChunk(
            channels={"arm": actions[:, :7], "gripper": actions[:, 7:]},
            specs={
                "arm": ActionSpec(
                    "end_effector_pose", 7, frame="task", control_point="flange", unit="m_xyzw"
                ),
                "gripper": ActionSpec("gripper_position", 1, unit="normalized_opening"),
            },
            period_s=1 / 30,
            source_frame_id=observation.frame_id,
            metadata={
                "model": result.get("model"),
                "request_id": result.get("id"),
                "waypoints": plan,
            },
        )
        self._pending, self._last_plan, self._last_image = chunk, plan, current_image
        self._ready = True
        return chunk

    def _complete(self, body: dict) -> dict:
        """One remote response, with no retry or physical action."""
        self._turn += 1
        self._save(f"request-{self._turn}.json", body)
        request = Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body, allow_nan=False).encode(),
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            started = time.monotonic()
            with urlopen(request, timeout=self.timeout_s) as response:
                # Some compatible providers send whitespace keepalives while queued.
                # json.load() can otherwise wait indefinitely despite a socket timeout.
                payload = bytearray()
                while True:
                    if time.monotonic() - started >= self.timeout_s:
                        raise TimeoutError("Cartesian inference deadline exceeded; no retry")
                    part = response.read1(65536)
                    if not part:
                        break
                    payload.extend(part)
                result = json.loads(payload)
        except HTTPError as exc:
            # Keep status only: response bodies may contain credentials or echoed headers.
            status = exc.code
            exc.close()
            raise RuntimeError(
                f"OpenAI-compatible POST /chat/completions: HTTP {status}; no retry"
            ) from None
        self._save(f"response-{self._turn}.json", result)
        return result

    def close(self, feedback: ExecutionReceipt | None) -> None:
        # Stateless API: no vendor session or close request, and no inference retry.
        if feedback is not None:
            self._save("final-receipt.json", self._receipt(feedback) or {})
        self._ready = False
