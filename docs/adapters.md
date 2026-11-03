# Writing adapters

Implement `robot_dock.policies.protocol.Policy`: `open()`,
`predict(observation, instruction, feedback) -> ActionChunk`, and `close(feedback)`.
Keep authentication, sessions and vendor response conversion in that adapter.
Send actual execution feedback on continuation and the final feedback on close.
Reject ambiguous state; never invent an executed-step count.

Implement `robot_dock.robots.protocol.RobotBackend`: expose `action_specs`, then
`open()`, `observe()`, `execute(actions, cancel)`, `stop()`, and `close()`.
`execute(..., record_step=...)` owns timing and inner-chunk cancellation and returns an
`ExecutionReceipt`. Known failures can raise `ExecutionError(receipt)`; unmeasured
exceptions are treated as unknown execution. `stop()` and `close()` must tolerate
partially opened resources. Low-level protective stops remain backend/device work.

If a step recorder is supplied, emit an actual pre-command observation with each
dispatched action's chunk, zero-based index and dispatch timestamp. Do not repeat
stale images or replace measured state with the target. `finish_recording()` drains
pending sampling work at task end, including in a persistent robot session. An
adapter without step sensors may emit no samples, but its episode must not claim
to be training-ready. See [dataset schema](datasets.md).

Use `PolicyBinding` for explicit state/image selection and coordinate/control-point
conversions. Do not infer units from array dimensions. Preserve chunk ID, source
frame and step count, and produce backend-compatible specs. A transform changing
sampling or action timing needs an explicit accounting design before inclusion.

Pass a task evaluator callable to `run()`; `True` terminates successfully,
`False` continues, and no evaluator leaves success unknown. Keep simulator
ground truth out of the policy-facing observation.

For a new vendor, test response validation, partial receipts, continuation turns,
final close and timeout behavior with a stub transport. For a device, test actual
measured execution and cancellation. Label simulated, stubbed and real validation
separately. Default CI must not need vendor keys or use paid endpoints.

## OpenAI-compatible Cartesian planner

`policies.openai.OpenAICartesianPolicy` uses Chat Completions with user text and
base64 PNG `image_url` blocks. It requires vision and `response_format: json_object`;
compatibility with one provider does not establish compatibility with every model.
Configure `OPENAI_API_KEY`, `OPENAI_API_BASE` (including `/v1` if the provider requires
it), and `OPENAI_API_MODEL`. The adapter appends `/chat/completions` exactly once.
There is no model-name default and no automatic provider or model fallback.

Optional `OPENAI_API_EXTRA_BODY` is a JSON object containing only `thinking`,
`reasoning_effort`, `temperature`, `max_tokens` or `max_completion_tokens`.
For DeepSeek Flash, `{"thinking":{"type":"disabled"}}` requests non-thinking mode.
Other providers may reject that option; remove it when switching. The default
token budget is 8192; `max_completion_tokens` replaces `max_tokens` if supplied.
Requests and responses are recorded without authorization headers or credentials.
Never put secrets in provider options, task instructions or model messages.

The remote model returns a JSON object with `waypoints`, each containing a 7D
absolute task-frame flange `pose` (meters, quaternion XYZW), `opening` in [0,1],
and segment-relative `duration_s`. One to eight waypoints and at most six seconds
are accepted. Durations must be positive multiples of 1/30 s. Position and gripper
opening interpolate linearly; orientation uses shortest-path SLERP. Each model
endpoint is preserved exactly. Invalid or truncated output stops the run without
repair, retry or physical dispatch. A subsequent call requires the actual execution
receipt for the previous chunk and uses a fresh observation, previous plan and receipt.
The immediately previous plan and its input RGB are retained in the next prompt,
alongside a clearly labelled current result RGB. A completed execution receipt means
commands were dispatched; it is not evidence of a grasp.

The model may first request `{"measurements":[{"pixel_uv":[u,v],"plane_z_m":z}]}`
instead of a motion plan. A read-only numeric tool projects each original image
pixel through the calibrated F-theta lens and intersects the ray with the requested
horizontal task-frame plane. The tool returns conditional XYZ and an explicit
`depth_is_assumed_not_measured` flag; it does not infer object depth or identity.
There are at most six pixels per call and three measurement calls per observation.
The model then chooses Cartesian waypoints from the results. These calls are distinct
read-only model decisions, not inference retries or predefined motion sequences.
No motion is dispatched until a complete valid waypoint plan arrives. Unsupported
geometry, backward rays, out-of-image pixels and exhausted budgets stop the run.

The binding exposes the same wrist RGB and measured flange/gripper state as RoboAPI,
plus a whitelist of static reference camera extrinsics and lens coefficients, measured
pinch-center pose and jaw width, and the robot's flange-to-pinch transform. It
excludes object locations, contact data and evaluator outcomes. The reference gripper
pinch offset is 0.1558 m along flange +Z, obtained from the robot kinematic model.
The backend still executes flange poses exactly as requested by the model. This is
additional explicit hardware information for a general model, so the comparison
measures complete policy integrations rather than identical model input prompts.
The camera calibration is a reference rig calibration, not a measurement of a physical UR5e.
The OpenAI example configs explicitly declare `robot.support_plane_z_m: 0.0`, the
configured reference tabletop height in task coordinates. On another setup, replace
this with a measured work-surface height or omit it. It is independent of object
positions and is never inferred from evaluator object state. Incorrect assumed object
height causes projection/parallax errors; RGB plus a plane assumption is not RGB-D.
The backend owns IK, robot control, timing and physical recording; the planner owns
only remote inference and generic interpolation. Physics continues during inference.

Use `configs/openai.mujoco.lift.json` and `configs/openai.mujoco.json` alongside the
matching `pokebot` configs. They share the scene, instructions, pause, budgets and
evaluators. Interactive `/stack` preserves the world and starts a new planning context.
Waypoint chunks and RoboAPI chunks have different horizons: compare wall-clock time,
dispatched steps and task success across repeated runs. Keep API availability, plan
validity, physical execution and task success as separate outcomes.

The DeepSeek configuration example follows the official
[vision guide](https://api-docs.deepseek.com/guides/vision/),
[JSON output guide](https://api-docs.deepseek.com/guides/json_mode/) and
[Chat Completions reference](https://api-docs.deepseek.com/api/create-chat-completion/).

Design references: [ORA](https://github.com/hao-ai-lab/ORA) supplies explicit robot
manuals, depth geometry, grasp proposals and motion tools;
[RoboHarness](https://github.com/BinceQu/RoboHarness) uses visual point selection,
tracking and depth back-projection; [ReKep](https://github.com/huangwl18/ReKep) couples
RGB-D keypoints to constraint optimization. These are architecture references, not
claims that Robot Dock reproduces their systems or published success rates. The
current RGB plane-intersection tool adds neither a depth sensor nor their grasp models.
Do not copy simulator object-mask/pose tracking into a policy observation: for example,
VoxPoser's released RLBench demo uses simulator masks, and ReKep's simulation keypoint
registry uses object transforms. Robot Dock keeps object ground truth evaluator-only.
