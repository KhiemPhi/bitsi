# BITSI Chebyshev consumer for the robot perception pipeline

A separate container on the Orin that reads the pipeline's robot-frame object clouds and runs BITSI Chebyshev segmentation
(`main.py` pipeline: build_cloud_object → auto_thickness → bitsi_metric → Chebyshev fit/inflections → segment tree → fuse →
rename). Motion planning is not done here; it consumes `/bitsi/result` and `/bitsi/segments`.

It is a NEW image built `FROM ros-realsense:humble`, which already holds ROS 2 Humble and the generated `seg_interfaces`
(`/opt/perception_interfaces`). No pipeline container or image is modified.

## On the Orin
```bash
# copy ~/bitsi (main.py, bitsi_slicer/, ros_bitsi/) to the Orin, then from its root:
ros_bitsi/run.sh build          # docker build -f ros_bitsi/Dockerfile -t bitsi-ros:humble .
python3 scripts/system.py start # in the pipeline repo, if not running
ros_bitsi/run.sh test           # imports + checks that /perception/* topics are visible (same ROS_DOMAIN_ID)
ros_bitsi/run.sh run            # start the node; outputs saved under ~/bitsi_outputs (BITSI_OUT to change)
python3 scripts/system.py detect "cup"   # in the pipeline repo; the node processes the clouds of this request
```
`run.sh` uses `--network host --ipc host -e ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}` as the pipeline docs require.

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
- Tested here: `bitsi_core.run_bitsi` on a synthetic partial robot-frame cloud (2.1 s, 5 segments, robot-frame
  centroids). Not tested here: the ROS node itself, the image build on aarch64, and the real seg_interfaces messages.
