# BITSI Chebyshev consumer for the perception pipeline. Builds a NEW image on top of the existing ros-realsense:humble
# (which already contains ROS 2 Humble and the generated seg_interfaces under /opt/perception_interfaces); nothing in the
# pipeline's images changes. Build context = ~/bitsi:   docker build -f ros_bitsi/Dockerfile -t bitsi-ros:humble .
FROM ros-realsense:humble
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip ros-humble-sensor-msgs-py libgl1 libgomp1 && rm -rf /var/lib/apt/lists/*
# numpy<2 keeps ROS Humble's compiled Python bindings working; open3d publishes aarch64 wheels for Python 3.10
RUN python3 -m pip install --no-cache-dir "numpy<2" "open3d>=0.18,<0.19" scikit-learn scipy matplotlib networkx
COPY bitsi_slicer /opt/bitsi/bitsi_slicer
RUN python3 -m pip install --no-cache-dir --no-deps /opt/bitsi/bitsi_slicer
COPY main.py /opt/bitsi/main.py
COPY ros_bitsi/bitsi_core.py ros_bitsi/bitsi_node.py /opt/bitsi/
WORKDIR /opt/bitsi
ENV PYTHONPATH=/opt/bitsi
CMD ["bash", "-lc", "source /opt/ros/humble/setup.bash && source /opt/perception_interfaces/setup.bash && python3 /opt/bitsi/bitsi_node.py --ros-args -p save_dir:=${BITSI_SAVE_DIR:-/tmp/bitsi_outputs} ${BITSI_ARGS:-}"]
