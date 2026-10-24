#!/usr/bin/env bash
# Build and run the BITSI consumer on the Orin (next to the running perception pipeline). Usage: ros_bitsi/run.sh [build|run|shell|test|target "label"]
set -e; cd "$(dirname "$0")/.."; IMG=bitsi-ros:humble; NAME=bitsi-chebyshev; OUT=${BITSI_OUT:-$HOME/bitsi_outputs}
RUN=(docker run --rm --name $NAME --network host --ipc host -e ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0} -e BITSI_SAVE_DIR=/outputs -e BITSI_ARGS="${BITSI_ARGS:-}" -e BITSI_TARGET="${BITSI_TARGET:-}" -v "$OUT":/outputs)
case "${1:-run}" in
  build) docker build -f ros_bitsi/Dockerfile -t $IMG . ;;
  run)   mkdir -p "$OUT"; "${RUN[@]}" -it $IMG ;;
  shell) "${RUN[@]}" -it $IMG bash -lc 'source /opt/ros/humble/setup.bash; source /opt/perception_interfaces/setup.bash; exec bash' ;;
  test)  "${RUN[@]}" $IMG bash -lc 'source /opt/ros/humble/setup.bash; source /opt/perception_interfaces/setup.bash; python3 -c "import bitsi_core, seg_interfaces.msg; print(\"imports ok\")"; ros2 topic list | grep -E "perception|bitsi" || echo "no /perception topics visible: check the pipeline is started and ROS_DOMAIN_ID matches"' ;;
  target) shift; T="$*"   # change the object while the node runs: run.sh target "cup"   (empty = all objects)
         docker exec $NAME bash -lc "source /opt/ros/humble/setup.bash; ros2 topic pub --once /bitsi/target std_msgs/msg/String \"{data: '${T//\'/}'}\"" >/dev/null && echo "target -> ${T:-all objects}" ;;
  *) echo "usage: $0 [build|run|shell|test|target \"label\"]"; exit 1 ;;
esac
