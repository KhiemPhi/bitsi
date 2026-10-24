# Orin: BITSI + SALSA on the real robot (state as of 2026-10-24)

Handoff note for continuing the real-robot work. Code: `ros_bitsi/` on branch `fitpart` of
`github.com:KhiemPhi/bitsi` (last commit `c17f2f6`). Details of every option: `ros_bitsi/README.md`.

## Goal
Run on the robot's Jetson Orin, next to the existing perception pipeline:
segment each detected object into parts (BITSI) -> apply the SALSA screw label (which part to grasp, which screw it
must follow) -> output a sequence of end-effector poses that a separate motion planner ingests later.
Motion planning, IK and execution are NOT part of this container.

## What runs where
| Piece | Where | Notes |
|---|---|---|
| Perception pipeline (RealSense, Qwen, GroundingDINO + SAM) | Orin, its own containers | not ours; publishes `/perception/*` |
| `bitsi-ros:humble` (our node) | Orin, built `FROM ros-realsense:humble` | no pipeline image is modified |
| Web view | node's own HTTP server, port 8090 | pipeline's viewer is on 8080 |

Orin: user `irslorin`, viewer at `http://192.168.1.102:8080` (pipeline) and `:8090` (BITSI/SALSA). Docker needs `sudo`.

## Files (`ros_bitsi/`)
| File | Role |
|---|---|
| `Dockerfile` | `FROM ros-realsense:humble`; numpy 1.24.4, numba 0.58.1, open3d 0.18, sklearn, matplotlib; copies `main.py`, `bitsi_slicer/` (used from PYTHONPATH, not pip), `fitpart/*.py`, `fitpart/bitsi2/`, `fitpart/tasks/*.py` |
| `run.sh` | `build`, `run`, `shell`, `test`, `target "label"`, `version v4`, `salsa "lift 10 cm"` |
| `bitsi_node.py` | ROS 2 node: subscriptions, dedup, target/version/SALSA switching, publishers, web hookup |
| `bitsi_core.py` | BITSI wrapper, no ROS: `run_bitsi(points, version=v0/v3/v4/v5/v6, ...)` |
| `salsa_core.py` | SALSA wrapper, no ROS: `salsa_plan(points, "lift 10 cm")` -> parts, screw, candidate pose sequences |
| `bitsi_web.py` | matplotlib figures + auto-refreshing page; boxes for target / version / SALSA command |

## How to run (on the Orin, from the `bitsi` clone)
```bash
git pull && sudo ros_bitsi/run.sh build
sudo ros_bitsi/run.sh test                      # "imports ok" + /perception topics visible
sudo SALSA_COMMAND="lift 10 cm" BITSI_TARGET="power drill" BITSI_VERSION=v4 \
     BITSI_OUT=$HOME/bitsi_outputs ros_bitsi/run.sh run
# pipeline repo: python3 scripts/system.py detect "power drill"   (or the Detect box on :8080)
sudo ros_bitsi/run.sh target "laptop, power drill"   # while running
sudo ros_bitsi/run.sh version v3
sudo ros_bitsi/run.sh salsa "turn 90 deg"            # "" = SALSA off
```
Put variables AFTER `sudo` (sudo drops the environment); `BITSI_OUT` must be explicit because sudo's `$HOME` is `/root`.
Extra node parameters: `BITSI_ARGS="-p voxel:=0.004 -p gripper_width:=0.085 -p up:=[0.0,0.0,1.0]"`.

## Topics
In: `/perception/clouds` (`seg_interfaces/ObjectClouds`), `/perception/result` (`SegObjects`, labels + 2D box/mask),
`/perception/overlay` (`Image`); control: `/bitsi/target`, `/bitsi/version`, `/salsa/command` (`std_msgs/String`).
Out: `/bitsi/result` (JSON), `/bitsi/segments` (`PointCloud2` with `segment` field), `/salsa/plan` (JSON),
`/salsa/poses` (`PoseArray`, best candidate of the first planned object). Inputs are reliable + transient-local, so the
node can start after a detection. Each (request, object, version, SALSA command) is processed once; changing target,
version or command re-processes the last retained request.

## SALSA output contract (for the motion planner)
- Frame: the cloud's header frame (`base_link`).
- Pose position: pinch point midway between the two finger contacts. Rotation columns: x, y = closing direction,
  z = approach toward the object (fitpart `grasp_frame`). **Add the gripper's pinch-to-TCP offset along z.**
- Sequence per candidate: `pre_grasp` (10 cm back, open) -> `grasp` (close) -> `lift_k` (if the rule lifts first) ->
  `screw_0..11` (hold) -> `release` (open) -> `retreat`. `press`: `pre_press` -> `touch` -> `press_k` -> `retreat`, closed.
- Up to 5 candidates, ordered by a heuristic (top-down approach first, then grasp near the part centre).
  **No IK, reachability, collision or SOCP task-wrench ranking here**; the planner takes the first feasible candidate.
- Commands: `turn, hinge, slide, press, squeeze, pivot, lift, swing, scoop, stir, cut, handover` (+ `fasten` only
  after `fitpart/tasks/screws.py` is committed); amount and sense from the text (`turn 90 deg reverse`), defaults from
  `fitpart/lang/model.py`. Aliases: open/close->hinge, rotate/twist->turn, pick->lift, pull->slide, push->press.

## Pipeline config change made on the robot
`config/system.json` -> `geometry.voxel_m` 0.009 -> 0.003 (and `cluster_eps_m` 0.02 -> 0.015). With 9 mm voxels a
bottle gave 163 points; after the change the drill gave ~2,000 and an open laptop ~20,600. Restart the pipeline after
editing (`scripts/system.py stop` / `start`).

## What we saw on real data (BITSI only; SALSA not yet run on the robot)
| Object | Points | Result |
|---|---|---|
| Water bottle (transparent) | 163 (9 mm voxel) | cloud did not look like a bottle: transparent plastic -> RealSense depth through it; also too sparse |
| Power drill, v0, 4 mm voxel | 1,988 | good: battery / handle / body / thin top sliver (sliver = over-segmentation; IBR 0.445 vs 0.341, tolerance 0.07) |
| Power drill, v0, `voxel:=0.0` | 1,977 | worse: handle and body merged; v0 is sensitive to point density/noise |
| Open laptop, v0 | 20,649 | wrong: sliced along the width (y) instead of at the hinge (x); one big part + two strips along the screen edge |
| Closed laptop, v0 | 3,224 | 5 segments, not inspected |

Synthetic checks (here, not robot): partial drill -> v3/v4 split handle vs barrel cleanly (2 parts, 0.5 s), v0 gave 5
slabs, v5 one part. SALSA `turn 90 deg` on a synthetic bottle grasps the cap from above and rotates about the vertical axis.

## Known issues and open questions
1. **v0 picks one slicing axis before looking for inflections** (`search_dimesion_slice`); L-shaped objects (open laptop)
   get the wrong axis. Option: sweep all three axes and keep the strongest inflection. Not implemented.
2. **Clouds look slightly tilted.** Either real geometry (drill grips are angled), single-view partial surface, or the
   hand-measured `config/camera_pose.yml` rotation. Test: `detect "laptop"` (closed, flat) and check its top surface is
   level in the side views. Proposed: a table-plane tilt diagnostic on the web page. Not implemented.
3. **Transparent objects** need a different depth source (matte spray/tape, depth completion). Test on opaque objects.
4. **Message field names are guessed by introspection** (ids, labels, 2D box/mask). The node logs the `ObjectClouds` and
   `SegObjects` layouts once at start; if labels/boxes look wrong, pin the names from those logs or the `.msg` files.
5. **SALSA rules were developed on full/fused simulation clouds.** RESULTS.md notes BITSI v4 part detection degrades
   on partial clouds; check the red grasp part on the web page before trusting a plan.
6. Gripper defaults: BITSI v0 width/height 0.13/0.07 (main.py); SALSA's screws.py uses `JAW` 0.085 (Kinova 2F-85,
   `P3_ROBOT`). Set to the real gripper.

## Uncommitted local files that the robot build does NOT include
| File | What differs | Effect on the Orin |
|---|---|---|
| `bitsi_slicer/bitsi_slicer/slicing.py` | 2 lines: IBR multiplied by occupancy empty-ratio | robot runs the committed (older) IBR |
| `fitpart/tasks/screws.py` | 66 lines incl. `fasten`, `fuse_parts`, handover fix | `fasten` unavailable; older rules |
| `fitpart/bitsi6.py` | untracked | `version v6` reports "not in this build" |
Commit them (deliberately) if the robot should use them.

## Untested
The ROS node itself has only run on the robot for BITSI (v0, targets, web, 2D box). Not yet run on the robot:
the v3-v5 option, the SALSA mode (`/salsa/plan`, `/salsa/poses`, SALSA web figure), `run.sh version` / `run.sh salsa`.

## Next steps
1. `git pull`, rebuild, run SALSA `lift 10 cm` on the power drill; check the grasp part is the handle and the pose
   path on the web page; echo `/salsa/poses` in RViz.
2. Compare v0 / v3 / v4 on the drill and the open laptop (switch with `run.sh version`; re-runs the same capture).
3. Settle the calibration tilt question (closed-laptop test) before feeding poses to the planner.
4. Agree the TCP offset and frame with whoever owns the motion planner; decide whether they want all candidates or one.
5. Optional work: all-axis slicing for v0, table-tilt diagnostic, sparse-cloud warning, commit the three local files.
