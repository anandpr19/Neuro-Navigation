#!/bin/bash
# Launcher script configured for VMware / Virtual Machine graphics stability

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE="$PROJECT_ROOT/ros2_ws"

if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
fi

source "$WORKSPACE/install/setup.bash"
export TURTLEBOT3_MODEL=burger

# VMware / Virtual Machine GPU Compatibility Environment Variables
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3
export QT_X11_NO_MITSHM=1
export OGRE_RTT_MODE=Copy

MODE="${1:-software_gui}"

echo "======================================================================"
echo "LAUNCHING GAZETBO SIMULATION FOR VM (Mode: $MODE)"
echo "======================================================================"

if [ "$MODE" == "headless" ]; then
    echo "Running Headless Physics Mode (Crash-Proof for VMs)..."
    ros2 launch brain_robot_control gazebo_headless.launch.py
else
    echo "Running Software Rendered GUI Mode..."
    ros2 launch brain_robot_control gazebo_simulation.launch.py
fi
