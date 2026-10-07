import os
root = '/root/ur5_project/ur5_comparison_ws/src/ur5_experiment'

files = {}
files['package.xml'] = '''<?xml version="1.0"?>
<package format="3">
  <name>ur5_experiment</name>
  <version>0.1.0</version>
  <description>UR5e gripper workcell for MuJoCo vs Gazebo comparison</description>
  <maintainer email="you@example.com">charan</maintainer>
  <license>Apache-2.0</license>
  <buildtool_depend>ament_cmake</buildtool_depend>
  <exec_depend>ur_description</exec_depend>
  <exec_depend>ur_simulation_gazebo</exec_depend>
  <export><build_type>ament_cmake</build_type></export>
</package>
'''
files['CMakeLists.txt'] = '''cmake_minimum_required(VERSION 3.8)
project(ur5_experiment)
find_package(ament_cmake REQUIRED)
install(DIRECTORY urdf config launch worlds DESTINATION share/${PROJECT_NAME})
ament_package()
'''
files['urdf/ur5e_gripper.urdf.xacro'] = '''<?xml version="1.0"?>
<robot xmlns:xacro="http://wiki.ros.org/xacro" name="$(arg name)">
  <xacro:include filename="$(find ur_description)/urdf/ur.urdf.xacro"/>

  <link name="gripper_base">
    <visual><origin xyz="0 0 0.01"/><geometry><box size="0.06 0.06 0.02"/></geometry>
      <material name="gdark"><color rgba="0.2 0.2 0.2 1"/></material></visual>
    <collision><origin xyz="0 0 0.01"/><geometry><box size="0.06 0.06 0.02"/></geometry></collision>
    <inertial><origin xyz="0 0 0.01"/><mass value="0.2"/>
      <inertia ixx="6.67e-5" iyy="6.67e-5" izz="1.2e-4" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="gripper_base_joint" type="fixed">
    <parent link="tool0"/><child link="gripper_base"/><origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>

  <xacro:macro name="finger" params="side x axis">
    <link name="finger_${side}">
      <visual><geometry><box size="0.01 0.02 0.08"/></geometry>
        <material name="gdark"/></visual>
      <collision><geometry><box size="0.01 0.02 0.08"/></geometry></collision>
      <inertial><mass value="0.05"/>
        <inertia ixx="2.83e-5" iyy="2.71e-5" izz="2.08e-6" ixy="0" ixz="0" iyz="0"/></inertial>
    </link>
    <joint name="finger_${side}_joint" type="prismatic">
      <parent link="gripper_base"/><child link="finger_${side}"/>
      <origin xyz="${x} 0 0.06"/>
      <axis xyz="${axis}"/>
      <limit lower="0" upper="0.025" effort="100" velocity="1"/>
      <dynamics damping="20"/>
    </joint>
    <gazebo reference="finger_${side}">
      <mu1>0.5</mu1><mu2>0.5</mu2>
    </gazebo>
  </xacro:macro>
  <xacro:finger side="r" x="0.045" axis="-1 0 0"/>
  <xacro:finger side="l" x="-0.045" axis="1 0 0"/>

  <gazebo reference="gripper_base"><mu1>0.5</mu1><mu2>0.5</mu2></gazebo>

  <ros2_control name="gripper" type="system">
    <hardware><plugin>gazebo_ros2_control/GazeboSystem</plugin></hardware>
    <joint name="finger_r_joint">
      <command_interface name="effort"/>
      <state_interface name="position"/><state_interface name="velocity"/>
    </joint>
    <joint name="finger_l_joint">
      <command_interface name="effort"/>
      <state_interface name="position"/><state_interface name="velocity"/>
    </joint>
  </ros2_control>
</robot>
'''
for rel, txt in files.items():
    open(os.path.join(root, rel), 'w').write(txt)

# add the finger controller to our copy of the controllers YAML
p = os.path.join(root, 'config/ur5e_gripper_controllers.yaml')
s = open(p).read()
k = "    forward_position_controller:\n      type: position_controllers/JointGroupPositionController\n"
assert k in s, 'anchor not found in YAML'
assert 'finger_effort_controller' not in s, 'already edited'
s = s.replace(k, k + "\n    finger_effort_controller:\n      type: effort_controllers/JointGroupEffortController\n")
s += "\nfinger_effort_controller:\n  ros__parameters:\n    joints:\n      - finger_r_joint\n      - finger_l_joint\n"
open(p, 'w').write(s)
print('written:', list(files), '+ yaml edited')
