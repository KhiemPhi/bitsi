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
figure per object (the full input cloud as received, the segments in 3D, and top x-y / side x-z views with segments coloured over the input in grey, centroids marked) and a table of
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
`web_port` 8090 (0 disables), `web_dir` /tmp/bitsi_web, `voxel` .004 m (voxel downsampling + statistical outlier removal before BITSI; 0 disables downsampling), `save_dir` (PLY per
segment + bitsi.json per object), `clouds_topic`, `result_topic`.

## Notes
- Field names of `ObjectClouds`/`SegObjects` are found by introspection (any `PointCloud2`, with `id`/`object_id` and
  `label`/`name` beside it). The node logs both message layouts on first receipt; if ids or labels show up wrong, check
  that log against `seg_interfaces/msg/*.msg` and adjust `ID_KEYS`/`LABEL_KEYS` in `bitsi_node.py`.
- The depth clouds are partial (visible surface), whereas BITSI was developed on full models; expect coarser segments.
- Tested here: `bitsi_core.run_bitsi` on a synthetic partial robot-frame cloud (2.1 s, 5 segments, robot-frame
  centroids). Not tested here: the ROS node itself, the image build on aarch64, and the real seg_interfaces messages.
