# Episode datasets

Every new task automatically produces `dataset.h5` in its task directory. The
file stores both model decisions and execution samples. It is a versioned native
HDF5 format, readable with h5py; it is not LeRobot or RLDS. No training framework,
dataset server or automatic upload is required.

```python
from robot_dock.recording import EpisodeDataset

episode = EpisodeDataset("runs/SESSION/tasks/0001/dataset.h5")
print(episode.outcome, len(episode), episode.training_ready)
sample = episode[0]
image = sample["images/wrist"]  # uint8 H x W x 3
state = sample["state/arm.ee_pose"]  # measured pose, not the target
action = sample["action/arm"]  # dispatched Cartesian target
instruction = sample["instruction"]

# Read policy input corresponding to a decision, and actual dispatched commands.
for chunk in episode.action_chunks():
    policy_input = episode.observation(chunk.source_frame_id)
```

`EpisodeDataset` provides `__len__` and `__getitem__`, returning NumPy arrays and
scalars. It opens files lazily per sample and retains no HDF5 handles, so it can
be passed to a PyTorch DataLoader in user training code. Image arrays are HWC uint8;
training transforms and batching conventions belong to that code, not the harness.
Select successful episodes explicitly when building a positive demonstration set;
failed episodes remain useful for analysis and are not automatically deleted.

## File schema (version 1)

| Path | Meaning |
|---|---|
| file attributes | format/version, instruction, run ID/configuration, outcome, completion and sample integrity |
| `observations/<frame_id>/state/*` | measured robot state supplied to the runtime at a decision boundary |
| `observations/<frame_id>/images/*` | images from that same observation; camera metadata on each image |
| `observations/<frame_id>` attributes | capture/simulation time, evaluator metadata kept separate from model input |
| `decisions/<chunk_id>/policy_action` | full model action before binding |
| `decisions/<chunk_id>/execution_planned` | bound action with units, frame, period and control point |
| `decisions/<chunk_id>/execution_receipt` | measured dispatched count and status |
| `steps/state/*`, `steps/action/*`, `steps/images/*` | synchronized execution samples, indexed on their first axis |
| `steps/image_timestamps_ns/*` | actual camera capture times |
| `steps/captured_at_ns`, `steps/dispatched_at_ns` | host monotonic sensor and command timestamps |
| `steps/simulation_time_s` | physical simulation time, when available |
| `steps/chunk_id`, `steps/step_index`, `steps/source_frame_id` | links back to model decisions |

Commands and measurements are distinct columns. Step indices start at zero.
The reference backend records measured joints, joint velocities, driver-register
position and pad gap as well as flange pose/opening. `gripper.opening` follows the
provider's normalized driver position; `gripper.width_m` is the separate physical
gap. These should not be interchanged during training or replay.
Sampling occurs before each command is dispatched. The MuJoCo backend copies the
then-current physical state under a short lock; a dedicated sensor renderer later
renders that exact snapshot, not a cached image or a newer robot state. Numerical
projection maps are cached because cameras are rigidly mounted; RGB is never cached.

Physics and dispatch do not wait for each image to finish rendering. Pending images
are finalized at task end, before the episode is marked complete. This can add a
post-task wait and memory use, especially on a slow renderer. Capture/dispatch
timestamps, actual dispatch rate and pending-sample counts expose timing overhead;
do not assume a perfect 30 Hz grid or invent missing frames. The native reader does
not silently resample observations, convert coordinates or replay a robot.

`training_ready` means every counted dispatched command has a recorded sample and
recording reported no errors. It does not mean the task succeeded, or that camera
calibration, the controller or demonstrations are suitable for a particular model.
Unknown dispatch cannot be replayed as known execution. `action_chunks()` defaults
to recorded dispatched prefixes; `dispatched_only=False` returns complete plans
for analysis, including undispatched rows. Replaying commands is a new physical
experiment and is never performed automatically.

Datasets finalize when a task ends, including interrupted/failed tasks. An abrupt
process termination can leave an incomplete file; the normal reader rejects it.
The separate session MP4 finishes when the session closes normally. Existing older
runs remain unchanged: they have no missing control-step images reconstructed or
claimed as newly captured data. Re-run a task to collect a synchronized episode.
