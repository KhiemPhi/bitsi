#!/usr/bin/env bash
# Build and run the BITSI consumer on the Orin (next to the running perception pipeline). Usage: ros_bitsi/run.sh [build|run|shell|test]
set -e; cd "$(dirname "$0")/.."; IMG=bitsi-ros:humble; NAME=bitsi-chebyshev; OUT=${BITSI_OUT:-$HOME/bitsi_outputs}
RUN=(docker run --rm --name $NAME --network host --ipc host -e ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0} -e BITSI_SAVE_DIR=/outputs -e BITSI_ARGS="${BITSI_ARGS:-}" -v "$OUT":/outputs)
case "${1:-run}" in
  build) docker build -f ros_bitsi/Dockerfile -t $IMG . ;;
  run)   mkdir -p "$OUT"; "${RUN[@]}" -it $IMG ;;
  shell) "${RUN[@]}" -it $IMG bash -lc 'source /opt/ros/humble/setup.bash; source /opt/perception_interfaces/setup.bash; exec bash' ;;
  test)  "${RUN[@]}" $IMG bash -lc 'source /opt/ros/humble/setup.bash; source /opt/perception_interfaces/setup.bash; python3 -c "import bitsi_core, seg_interfaces.msg; print(\"imports ok\")"; ros2 topic list | grep -E "perception|bitsi" || echo "no /perception topics visible: check the pipeline is started and ROS_DOMAIN_ID matches"' ;;
  *) echo "usage: $0 [build|run|shell|test]"; exit 1 ;;
esac
