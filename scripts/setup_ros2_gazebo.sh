#!/bin/bash
# Script to install ROS 2 Jazzy Jalisco and Gazebo Harmonic on Ubuntu 24.04 LTS (Noble)

set -e

echo "======================================================================"
echo "Installing ROS 2 Jazzy Jalisco & Gazebo Harmonic for Neuromorphic Robotics"
echo "======================================================================"

sudo apt update && sudo apt install -y software-properties-common curl
sudo add-apt-repository universe -y

# Add ROS 2 GPG Key & Repository
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-jazzy-ros-base \
                    ros-jazzy-ros-gz \
                    ros-jazzy-ros-gz-sim \
                    ros-jazzy-turtlebot3 \
                    ros-jazzy-turtlebot3-gazebo \
                    ros-jazzy-turtlebot3-simulations \
                    python3-colcon-common-extensions \
                    python3-rosdep

# Initialize rosdep
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    sudo rosdep init
fi
rosdep update

echo "======================================================================"
echo "ROS 2 Jazzy and Gazebo Harmonic Installation Complete!"
echo "Source environment with: source /opt/ros/jazzy/setup.bash"
echo "======================================================================"
