# BITSI + SALSA on the robot (Jetson Orin)

Our ROS 2 node runs next to the robot's perception pipeline. For each detected object it:
1. segments the object's point cloud into parts (**BITSI**),
2. picks the part to grasp and the screw motion to follow for a task command (**SALSA**, e.g. `lift 10 cm`),
3. outputs a sequence of end-effector poses for a separate motion planner.

```
RealSense -> perception pipeline (GroundingDINO + SAM, its own containers)
               | /perception/clouds, /perception/result, /perception/overlay
               v
           bitsi-ros:humble (this folder) --> /bitsi/result, /bitsi/segments      (parts)
                                          --> /salsa/plan, /salsa/poses          (grasp + pose sequence)
                                          --> web view on :8090
```
Motion planning, IK and execution are **not** done here. The image is built `FROM ros-realsense:humble` (ROS 2 Humble +
the pipeline's `seg_interfaces`); no pipeline image or container is modified.

## Quick start (on the Orin)
Terminal 1 is the node; terminal 2 is for detections and live changes. Docker needs `sudo`; put environment
variables **after** `sudo` (sudo drops the environment) and give `BITSI_OUT` as a full path (sudo's `$HOME` is `/root`).

**1. Start the perception pipeline** (in the pipeline repo):
```bash
python3 scripts/system.py start
```
Keep `geometry.voxel_m` at `0.003` in its `config/system.json` (at 0.009 a bottle had only 163 points). After editing:
`scripts/system.py stop` then `start`.

**2. Build and test the node** (in the `bitsi` clone, branch `fitpart`):
```bash
git pull
sudo ros_bitsi/run.sh build     # docker build -f ros_bitsi/Dockerfile -t bitsi-ros:humble .
sudo ros_bitsi/run.sh test      # prints "imports ok" and the /perception/* topics
```
No `/perception` topics: the pipeline is not running or `ROS_DOMAIN_ID` differs (`run.sh` passes `${ROS_DOMAIN_ID:-0}`).

**3. Run the node** (terminal 1):
```bash
sudo SALSA_COMMAND="lift 10 cm" BITSI_TARGET="power drill" BITSI_VERSION=v4 \
     BITSI_OUT=/home/irslorin/bitsi_outputs ros_bitsi/run.sh run
```

**4. Detect an object** (terminal 2, pipeline repo; or the Detect box on `http://<orin-ip>:8080`):
```bash
python3 scripts/system.py detect "power drill"
```
The node's inputs are transient-local, so it can also be started after the detection.

**5. Check the result**
- Web view `http://<orin-ip>:8090` (refreshes every 5 s): camera crop with the detector's box and mask, the input cloud,
  BITSI parts, and the SALSA figure (grasp part in red, screw axis, pose path). For `lift 10 cm` on a drill, the red part
  should be the handle and the path should rise ~10 cm.
- Topics: `ros2 topic echo /salsa/poses --once`, `ros2 topic echo /salsa/plan --once`; `/salsa/poses` and
  `/bitsi/segments` can be shown in RViz (frame `base_link`).
- Files: `BITSI_OUT/` holds a PLY per part and `bitsi.json` per object.

**6. Change things while it runs** (terminal 2; each change re-processes the last capture, no new detection needed):
```bash
sudo ros_bitsi/run.sh version v3           # BITSI algorithm: v0 v3 v4 v5
sudo ros_bitsi/run.sh salsa "turn 90 deg"  # task command; "" turns SALSA off
sudo ros_bitsi/run.sh target "laptop"      # objects to process (comma-separated; "" = all)
```
The web page has the same three controls. `sudo ros_bitsi/run.sh shell` opens a shell in the image with ROS sourced.

## Before trusting a plan
- Opaque objects only: RealSense depth passes through transparent plastic (the water bottle failed).
- Clouds are single-view and partial; SALSA's rules were developed on full simulated clouds. Check the red grasp part.
- Calibration tilt is unresolved: detect a closed laptop and check its top is level in the side view.
- No IK, reachability or collision checks here; candidates are ordered by a heuristic only (see below).
- Poses are at the pinch point; add the gripper's pinch-to-TCP offset along the approach (z) axis.
- Gripper size: BITSI v0 uses `gripper_width` 0.13 / `gripper_height` 0.07, SALSA a 0.085 m jaw (Robotiq 2F-85).

## Troubleshooting
| Symptom | Check |
|---|---|
| `test` lists no `/perception` topics | pipeline started? same `ROS_DOMAIN_ID`? |
| Nothing on :8090 after a detection | `BITSI_TARGET` matches the label? (`run.sh target ""` processes all objects) |
| "only N points after filtering" | object too small / far / transparent, or the pipeline voxel is too coarse |
| Wrong labels or ids | the node logs the `ObjectClouds` / `SegObjects` layouts at start; compare with `seg_interfaces/msg/*.msg` |
| Variables ignored | they must come after `sudo` |

## SALSA: parts -> screw label -> end-effector pose sequence
Give a task command and every processed object also gets a SALSA plan (`salsa_core.py`, using `fitpart/tasks/screws.py`):
1. **Segment:** BITSI v4 with gravity `up` (screws.analyze): coaxial segments along the root spine plus off-axis children.
2. **Screw label:** the task rule picks the part to grasp and the screw it must follow (axis, point, pitch, magnitude;
   Lynch & Park). Tasks: turn, hinge, slide, press, squeeze, pivot, lift, swing, scoop, stir, cut, handover (+ fasten if
   your local screws.py is committed). The command sets amount and sense: `lift 10 cm`, `turn 90 deg`, `turn 90 deg reverse`,
   `hinge` (default amounts as fitpart/lang/model.py). Aliases: open/close -> hinge, rotate/twist -> turn, pick -> lift, pull -> slide, push -> press.
3. **Pose sequence:** antipodal grasps on that part (fitpart/grasping.sample_grasps); per candidate:
   `pre_grasp` (10 cm back along the approach, gripper open) -> `grasp` (close) -> `lift_k` (if the rule lifts first) ->
   `screw_0..11` (the screw motion, gripper holding) -> `release` -> `retreat`. press: `pre_press` -> `touch` -> `press_k` -> `retreat`, fingers closed.

```bash
sudo SALSA_COMMAND="lift 10 cm" BITSI_TARGET="power drill" BITSI_OUT=$HOME/bitsi_outputs ros_bitsi/run.sh run
sudo ros_bitsi/run.sh salsa "turn 90 deg"     # change while running (re-plans the last request); "" = off
```
or the "SALSA command" box on the web page, or `std_msgs/String` on `/salsa/command`.

| Output | Type | Content |
|---|---|---|
| `/salsa/plan` | `std_msgs/String` JSON | per object: command, parts, screw (axis, point, pitch/translation, theta, lift), up to `salsa_candidates` (5) candidates, each `{rank, contacts, width, score, poses: [{name, position, quaternion_xyzw, gripper}]}` |
| `/salsa/poses` | `geometry_msgs/PoseArray` | the best candidate's pose sequence for the first planned object (RViz / planner) |

**Pose convention:** robot frame of the cloud (`base_link`); position = pinch point midway between the two finger contacts;
rotation columns = hand x, y = closing direction, z = approach towards the object (fitpart `grasp_frame`). Add your
gripper's pinch-point-to-TCP offset along z. **Not done here:** IK, reachability, collisions and the SOCP task-wrench
ranking used in simulation (no robot model or object mass on this side). Candidates are ordered by a heuristic (top-down
approach first, then grasp near the part centre); the motion planner should take the first candidate it can execute.

The web page shows, per object, a SALSA figure (parts, grasp part in red, screw axis, the best candidate's path with
approach arrows) and the pose table.

## Choosing the BITSI version
| version | algorithm |
|---|---|
| `v0` (default) | `main.py`: BITC sweep along one bounding-box axis, Chebyshev inflections, IBR fusion |
| `v3` | `fitpart/bitsi3.py`: primitive-relative BITC channels, Chebyshev pieces, MDL cuts |
| `v4` | `fitpart/bitsi4.py` with `up`: v3 + children by code length, gravity-axis candidate (robot setting) |
| `v5` | `fitpart/bitsi5.py` partial-cloud candidate (`axis_rule=residual, spine_coding=free, partial_cuts=True`) |
| `v6` | `fitpart/bitsi6.py`, experimental; only available if that file is committed (it is not in the repo yet) |

```bash
sudo BITSI_VERSION=v4 BITSI_OUT=$HOME/bitsi_outputs ros_bitsi/run.sh run   # at start
sudo ros_bitsi/run.sh version v3                                          # switch while running
```
or the "BITSI version" menu on the web page, or `std_msgs/String` on `/bitsi/version`. Switching re-runs the last request,
so versions can be compared on the same capture. v3-v6 estimate normals (30 neighbours, oriented away from the centroid);
v4-v6 use gravity `up` = `[0, 0, 1]` in the cloud frame (`-p up:="[0.0, 0.0, 1.0]"` if base_link's z is not up). v3+ parts
are named `part_k` (largest first) and have no IBR value. On a synthetic partial drill (handle + barrel), v3 and v4 split
handle from barrel (2 parts, 0.5 s), v0 gave 5 slabs along height, v5 kept one part.

## Choosing the object
By default every object in each request is segmented. To run BITSI only on some objects, give label text (case-insensitive,
matched inside the pipeline's label; comma-separated for several; an object id also works):
```bash
sudo BITSI_TARGET="cup" BITSI_OUT=$HOME/bitsi_outputs ros_bitsi/run.sh run    # at start
sudo ros_bitsi/run.sh target "red cup, phone"                                 # change while running
sudo ros_bitsi/run.sh target ""                                               # back to all objects
```
The web page has the same box ("Run BITSI on"), and any node can publish `std_msgs/String` to `/bitsi/target`.
Changing the target re-checks the last retained request immediately, so objects skipped earlier are processed without a
new detection; objects already processed are not repeated.

## Web view
While the node runs, open `http://<orin-ip>:8090/` in a browser. After every processed request it renders a matplotlib
figure per object (the camera image cropped around the detector's 2D box with the mask tinted, from `/perception/overlay` and `/perception/result`; the full input cloud as received, the segments in 3D, and top x-y / side x-z views with segments coloured over the input in grey, centroids marked) and a table of
segments (points, centroid, oriented-box extent, IBR). The page refreshes every 5 s, newest request first, and keeps the
last 20 requests. Change the port with `BITSI_ARGS="-p web_port:=8091"`; `-p web_port:=0` disables it. The server binds to
all interfaces (host networking), so expose the port only on a trusted network, like the pipeline's own viewer on 8080.

## Other point-cloud topics
Any `sensor_msgs/PointCloud2` topic (one object per message) works too:
```bash
BITSI_ARGS="-p input_type:=pointcloud2 -p clouds_topic:=/your/cloud_topic" ros_bitsi/run.sh run
```
Frames arriving while BITSI is still running on the previous one are dropped. Results use the cloud's own header frame.
Any node parameter can be passed the same way, e.g. `BITSI_ARGS="-p voxel:=0.006 -p gripper_width:=0.085"`.

## Topics
| Direction | Topic | Type | QoS |
|---|---|---|---|
| in | `/perception/clouds` | `seg_interfaces/msg/ObjectClouds` | reliable, transient-local |
| in | `/perception/result` | `seg_interfaces/msg/SegObjects` (labels only) | reliable, transient-local |
| out | `/bitsi/result` | `std_msgs/String`, JSON | reliable, transient-local |
| out | `/bitsi/segments` | `sensor_msgs/PointCloud2`, fields x y z segment (float32), robot frame | reliable, transient-local |

`/bitsi/result` JSON: `{request_id, frame_id, stamp, objects: [{id, label, n_points, n_filtered, thickness, slice_axis,
seconds, segments: [{name, n_points, centroid, ibr_ratio, obb_center, obb_extent, obb_R}]}]}`; an object that cannot be
segmented carries `error` instead of `segments`. All coordinates are in the cloud's header frame (the fixed robot frame).

Each `(request_id, object id)` is processed once, so the pipeline's repeated publications for late subscribers do not
re-run BITSI. If a new request arrives while one is running, it is picked up from the pipeline's next repeat.

## Parameters (`--ros-args -p name:=value`)
`gripper_width` .13, `gripper_height` .07, `epsilon` 5e-3, `strength_threshold` .01, `ibr_tolerance` .07 (main.py values),
`overlay_topic` /perception/overlay (empty disables the image panel), `web_port` 8090 (0 disables), `web_dir` /tmp/bitsi_web, `voxel` .004 m (voxel downsampling + statistical outlier removal before BITSI; 0 disables downsampling), `save_dir` (PLY per
segment + bitsi.json per object), `clouds_topic`, `result_topic`.

## Notes
- Field names of `ObjectClouds`/`SegObjects` are found by introspection (any `PointCloud2`, with `id`/`object_id` and
  `label`/`name` beside it). The node logs both message layouts on first receipt; if ids or labels show up wrong, check
  that log against `seg_interfaces/msg/*.msg` and adjust `ID_KEYS`/`LABEL_KEYS` in `bitsi_node.py`.
- The depth clouds are partial (visible surface), whereas BITSI was developed on full models; expect coarser segments.
- Tested on the robot (2026-10-24): build, BITSI v0, targets, web view (power drill, laptop, water bottle; results in
  `docs/state/ORIN_BITSI_SALSA.md`). Not yet run on the robot: v3-v5, SALSA (`/salsa/plan`, `/salsa/poses`),
  `run.sh version` / `run.sh salsa`.
