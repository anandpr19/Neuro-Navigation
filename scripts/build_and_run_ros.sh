#!/bin/bash
# Script to build ros2_ws colcon workspace and launch neuromorphic robot control

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE="$PROJECT_ROOT/ros2_ws"

if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
elif [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
else
    echo "[WARNING] ROS 2 setup.bash not found in /opt/ros/. Please source ROS 2 first."
fi

echo "======================================================================"
echo "Building ROS 2 Workspace: $WORKSPACE"
echo "======================================================================"

cd "$WORKSPACE"
colcon build --symlink-install

source "$WORKSPACE/install/setup.bash"

echo "======================================================================"
echo "ROS 2 Package 'brain_robot_control' Built Successfully!"
echo "To launch standalone ROS 2 Node:"
echo "  ros2 launch brain_robot_control controller.launch.py"
echo ""
echo "To launch Gazebo 3D Simulation + ROS 2 Node:"
echo "  export TURTLEBOT3_MODEL=burger"
echo "  ros2 launch brain_robot_control gazebo_simulation.launch.py"
echo "======================================================================"
