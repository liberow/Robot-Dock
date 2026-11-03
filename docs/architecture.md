# Architecture

Robot Dock is a policy-execution library. High-level conversation, planning and
long-lived agent task management remain external applications.

## Boundaries

`Policy` owns remote/local inference and policy-session lifecycle.
`RobotBackend` owns observation and physical execution. `run()` owns a single
bounded synchronous loop. `PolicyBinding` is explicit observation/action mapping,
implemented with ordinary functions. Recording and task evaluation are composed
into the loop; they are not independent services.

The dependency direction is `integrations -> public protocols`; the runtime uses
those protocols and has no concrete provider or device imports. The CLI composes
the reference integration and may name vendors. Python callers can supply their
own objects without editing the CLI.

## Contracts

- `Observation`: a frame ID, host monotonic capture timestamp, named measured
  state channels, RGB images, optional simulation time and evaluator metadata.
- `ActionSpec`: explicit space, dimension, absolute/delta mode, unit, reference
  frame and control point. Currently Cartesian actions require absolute m/xyzw.
- `ActionChunk`: synchronized named TxD channels, source frame ID, action period
  and identity. Different named channels allow future bimanual layouts.
- `ExecutionReceipt`: requested steps, measured dispatched steps, completion,
  interruption, failure or unknown result. Completion does not prove task success.
- `RunOutcome`: run lifecycle, optional task success, dispatch count and cleanup errors.

Bindings must preserve source frame, chunk identity and step count. The runtime
checks the resulting action contract against the backend before dispatch. Vendor
actions and transformed physical commands are both recorded. Observation images
are copied from device buffers and saved separately from structured events.

The reference `pokebot_umi` binding passes Cartesian actions through unchanged.
It does not impose a fixed scene bounding box or clip target positions. Reachability
is handled by the robot backend's controller; contact is handled by scene physics.
Observation mapping removes evaluator-only ground truth from the policy input.

## Execution

Queue a planned execution record before dispatch; observe actual device state after
execution. Report the final outstanding receipt when closing a policy session.
Partial results are never counted as a full chunk. Unknown results stop the run
and require reconciliation, not replay. Stop the device before closing network
sessions; cleanup errors remain visible without overwriting the task result.
Events and HDF5 samples use one ordered writer, drained before the final summary. They
are not guaranteed durable before dispatch or after abrupt process termination.
Recording failures are reported as cleanup errors rather than discarded.
Each task writes one HDF5 episode, with a lazy NumPy reader for analysis and
training. Model decisions and measured control-step samples are separate; their
timestamps and chunk identities are explicit. Backend sampling is drained with
`finish_recording()` before finalizing the episode, including caller-owned worlds.

By default `run()` opens and closes its backend. For a persistent world, the
caller can open it once and pass `manage_backend=False` to successive runs;
the caller must finally close it. Each run still stops the backend and opens and
closes its own policy session. The native reference CLI uses this ownership rule
to keep the world visible between terminal instructions, without adding an agent
planner or device-specific behavior to the runtime.

Cancellation is cooperative: the runtime checks before/after inference, and the
backend checks within chunks. Wall budgets are checked between blocking calls;
they do not preempt arbitrary SDK calls. Each adapter owns its transport timeout.

The current runner is deliberately synchronous. No task graph, distributed bus,
plugin discovery, database server or model-training framework is required.
Capabilities are explicit action contracts and integration-specific observation
checks; a general automatic compatibility planner is not implemented.

## Timing and evaluation

Host monotonic time controls budgets; observation age is recorded and an age limit
is opt-in. The MuJoCo backend owns a continuous 500 Hz physical worker paced to wall
time, including API waits and terminal input. Renderers consume copied snapshots;
video encoding and dataset sensor rendering run separately. Action dispatch targets 30 Hz without dropping or
compressing commands. Actual dispatch rate, physics lag, IK error and measured
tracking error distinguish controller limitations from policy behavior. Timing is
best-effort, not hard real time. IK numerical error alone does not terminate tasks.

Ground-truth object state is evaluator-only, removed from policy observations.
The reference stack condition must hold for one second of physical time. Gripper
linkage and pad friction come from the upstream asset. Mechanical constraints model
the C4 parallel-locking kit required by the provider; pad friction is not overridden.
Visible camera/brackets follow the official assembly datums, with payload and external
collisions. Accessory masses, convex collision proxies and optics remain approximate;
source hashes and hardware metadata identify the configuration in each dataset.
Arm servos apply bias-force compensation and critical joint damping derived from
Home inertia, retaining upstream stiffness and force limits. Gripper and objects
remain subject to gravity. Profile v4 imports the pinned reference rig's optical
transform, native F-theta polynomial and gripper command/encoder response. Sampled
encoder feedback travels with each copied physical state in MuJoCo userdata.
This matches reference inputs, not a validated physical UR5e configuration or
pixel-identical Isaac rendering. Raw gripper commands remain in recordings;
device saturation is accounted for by the backend. Physics continues during
inference; unlike the reference benchmark, it does not freeze between chunks.

## Extension policy

Add an adapter and contract tests for actual new requirements. Introduce shared
abstractions only after repeated integrations justify them. Remote process
transports can implement the same Python protocols when environment isolation is
needed; transport is not a requirement of every integration.
