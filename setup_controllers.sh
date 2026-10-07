#!/usr/bin/env bash
# Run after every Gazebo relaunch. Spawns the finger and velocity controllers
# and switches the arm from the trajectory controller to the velocity controller.
# Untested wrapper around the manual commands used during the project.
source /opt/ros/humble/setup.bash
[ -f /root/ur5_project/ur5_comparison_ws/install/setup.bash ] && source /root/ur5_project/ur5_comparison_ws/install/setup.bash

echo "waiting for /controller_manager..."
for i in $(seq 1 60); do
  ros2 control list_controllers >/dev/null 2>&1 && break
  sleep 1
done

ros2 run controller_manager spawner finger_effort_controller -c /controller_manager
ros2 run controller_manager spawner forward_velocity_controller -c /controller_manager --inactive
ros2 control switch_controllers --deactivate joint_trajectory_controller \
  --activate forward_velocity_controller --strict
ros2 control list_controllers
