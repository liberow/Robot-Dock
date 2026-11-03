# Robot assets

Run `uv run --group assets python scripts/fetch_assets.py` to fetch UR5e and Robotiq 2F-85 from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), pinned to
`b0655057a9df2abe39ea7fc989914ba6f248301c`. Large meshes are ignored by this repository.
Keep each model's upstream LICENSE and README. The composer retains the upstream
four-bar linkage and pad friction, and adds the C4 kit's mechanical parallel lock.
The upstream files themselves are not modified.

The same command downloads the RoboAPI Ace Pro 2 wrist assembly. For an existing
Menagerie installation, `--hardware-only` fetches/converts just this assembly.
The `assets` dependency group contains the meshopt decoder used during conversion;
it is not a simulator/runtime dependency. No CAD tool or JavaScript runtime is needed.
Downloaded files and converted meshes live under the ignored `assets/pokebot/`.
Their upstream terms are retained; the project's code license does not relicense them.

## Wrist assembly provenance

- [Official installation and bracket CAD](https://developers.pokebot.com/docs/roboapi/latest/hardware).
- [Official assembly GLB](https://developers.pokebot.com/models/assembly/ace-pro-2.glb).
- [Upper STEP](https://developers.pokebot.com/downloads/hardware/robotiq-ace-pro-2-upper.stp),
  [lower STEP](https://developers.pokebot.com/downloads/hardware/robotiq-ace-pro-2-lower.stp).
- [Camera dimensions/177.2 g mass](https://onlinemanual.insta360.com/acepro2/en-us/faq/specs/hardware).

`fetch_assets.py` pins SHA-256 digests and exports the official triangulated bracket,
camera exterior/frame and fastener meshes to OBJ in metres. The camera exterior is
marked as a reconstruction by the upstream GLB, not manufacturer engineering CAD.
The STEP originals are retained. The UR flange plane and lens-mount pose are read
from the assembly, rather than placing the visible camera independently of its sensor.
GLB coordinates are translated by the source wrist's flange-plane Z (about 47.3 mm)
and rotated 180 degrees around Z to register the CAD presentation axes to RoboAPI's
signed EE axes. The gripper opening axis alone cannot resolve that sign ambiguity.
The provider requires image-right to be +X_EE, not -X_EE; this is checked against
the actual camera transform. The registration rotates the physical assembly, not
the rendered image. It is not a measurement of a particular physical assembly.

The model includes visible geometry, mass/inertia and external collisions. Camera
body mass uses the manufacturer's 177.2 g specification. Bracket/frame masses use
an **estimated** 1240 kg/m³ polymer density and mesh volume; fasteners/pins use an
estimated 7850 kg/m³ steel density. The 46 mm × 22 mm lens envelope follows the
official assembly viewer and has an estimated 30 g mass. Those accessory masses,
camera centre of mass and resulting inertias need physical measurement. The camera,
brackets and lens currently add about 0.362 kg, excluding moving locking pins.

The C4 pins use the official viewer's 5 mm diameter, 23 mm length and mounting
centres. They are attached to the drivers; an equality constraint on each adaptive
joint models the real lock, while the native four-bar connects remain intact.
Only mating pin/link self-contacts are excluded because convex collision meshes
cannot represent pin bores. Environment/object collisions remain enabled.

## Reference rig v4

The fetcher installs the checksum-pinned [roboapi-playground v1.0.0 rig](https://github.com/huashanjian/roboapi-playground/releases/tag/roboapi-rig-v1.0.0).
Archive and per-input SHA-256 checks are applied. The original notices and README
remain with the assets. USD is needed only by the assets group to export authored
lens values; runtime uses JSON without resolving Isaac external references.

The physical CAD mount remains nominally 30 degrees. The optical transform uses
T_tcp_camera_optical (about 27.57 degrees tilt). Full-frame RGB rays use the native
F-theta pixel-radius-to-angle polynomial and principal point from assembly.usdc,
not nominal 189-degree optics or D_effective. Opaque lens-envelope proxies alone
are hidden from wrist rendering; external views and collisions retain them.
MuJoCo shading is not pixel-identical to Isaac RTX or a physical camera.

Gripper travel uses the rig's 3..229 register interval, 0.1 s delay, first-order
response, directional rate limits, 50 ms command period and 150 ms encoder sampling.
Travel maps to the Menagerie actuator range. Feedback samples physical actuator
travel, not commanded targets, and is retained in copied MuJoCo userdata.
Native contact dynamics remain active. This is reference-response calibration,
not identification of this physical UR5e/Robotiq. Inertias and hulls remain estimates.
Arm damping uses critical damping from Home diagonal joint inertia, retaining
upstream stiffness and force limits without changing Cartesian policy targets.

New hardware profile roboapi_reference_rig_v4 records lens/response parameters,
controller damping and source hashes. Do not pool older v2/v3 episodes with it.

Some recorded scene trials reference `assets/pokebot/tabletop/Oak_BaseColor.png`.
Keep this ignored asset when retaining those episodes for replay. Its upstream
source is RoboLab revision `b6cfcf6ccd2113c4cd1216d35b33024a7d5691da`, path
`assets/materials/Base/Wood/Oak/Oak_BaseColor.png`, SHA-256
`58648d35e3b8a9d953948ce0394b56aa95184e1f88b5abab139e7a32a2b44043`.
It retains upstream asset terms; it is not relicensed by this project's MIT code
license. The verified two-stage stacking example uses the original plain table.
