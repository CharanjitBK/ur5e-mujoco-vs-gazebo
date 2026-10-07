import re
R = '/root/ur5_project/ur5_comparison_ws/src/ur5_experiment'

# 1. xacro: copy of the gripper xacro plus a 20x20x100 mm, 0.1 kg peg on the palm
x = open(f'{R}/urdf/ur5e_gripper.urdf.xacro').read()
assert 'name="peg"' not in x
peg = '''
  <link name="peg">
    <visual><geometry><box size="0.02 0.02 0.10"/></geometry>
      <material name="pegc"><color rgba="0.8 0.6 0.1 1"/></material></visual>
    <collision><geometry><box size="0.02 0.02 0.10"/></geometry></collision>
    <inertial><mass value="0.1"/>
      <inertia ixx="8.67e-5" iyy="8.67e-5" izz="6.7e-6" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="peg_joint" type="fixed">
    <parent link="gripper_base"/><child link="peg"/><origin xyz="0 0 0.07" rpy="0 0 0"/>
  </joint>
  <gazebo reference="peg"><mu1>0.5</mu1><mu2>0.5</mu2><kp>1000000</kp><kd>1</kd><minDepth>0</minDepth></gazebo>
</robot>'''
open(f'{R}/urdf/ur5e_peg.urdf.xacro', 'w').write(x.replace('</robot>', peg, 1))

# 2. world: same physics and table, no cube/target, plus a 26 mm square hole made of 4 walls
w = open(f'{R}/worlds/ur5e_experiment.world').read()
for n in ('cube', 'target'):
    w = re.sub(rf'<model name="{n}">.*?</model>', '', w, count=1, flags=re.S)
H, T = 0.013, 0.008
S = ('<surface><friction><ode><mu>0.5</mu><mu2>0.5</mu2></ode></friction>'
     '<contact><ode><kp>1e6</kp><kd>1</kd><min_depth>0</min_depth></ode></contact></surface>')
def wall(n, px, py, sx, sy):
    return (f'<collision name="{n}"><pose>{px} {py} 0 0 0 0</pose><geometry><box><size>{sx} {sy} 0.04</size></box></geometry>{S}</collision>'
            f'<visual name="{n}_v"><pose>{px} {py} 0 0 0 0</pose><geometry><box><size>{sx} {sy} 0.04</size></box></geometry>'
            '<material><ambient>0.3 0.3 0.8 1</ambient></material></visual>')
L = 2 * (H + 2 * T)
fix = f'''
    <model name="fixture">
      <pose>0.5 0.2 0.42 0 0 0</pose>
      <link name="link">
        <gravity>false</gravity><kinematic>true</kinematic>
        <inertial><mass>1</mass><inertia><ixx>0.01</ixx><iyy>0.01</iyy><izz>0.01</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        {wall('wall_px', H+T, 0, 2*T, L)}
        {wall('wall_nx', -(H+T), 0, 2*T, L)}
        {wall('wall_py', 0, H+T, 2*H, 2*T)}
        {wall('wall_ny', 0, -(H+T), 2*H, 2*T)}
      </link>
    </model>
  </world>'''
assert '</world>' in w
open(f'{R}/worlds/ur5e_peg.world', 'w').write(w.replace('</world>', fix, 1))
print('written')
