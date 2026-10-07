D = '/root/ur5_project/mujoco_ur5/models/working_ur5e'
scene = '''<mujoco model="ur5e_tap">
  <include file="ur5e_peg.xml"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual><headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/></visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.8 0.8 0.8 1" friction="1 0.005 0.0001"/>
    <geom name="table" type="box" size="0.25 0.4 0.2" pos="0.55 0 0.2" rgba="0.6 0.5 0.4 1" friction="0.5 0.005 0.0001"/>
    <body name="padbody" pos="0.5 0.2 0.41">
      <geom name="pad" type="box" size="0.05 0.05 0.01" rgba="0.3 0.3 0.8 1" friction="0.5 0.005 0.0001"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="finger_r" joint="finger_r_joint" ctrlrange="-8 8"/>
    <motor name="finger_l" joint="finger_l_joint" ctrlrange="-8 8"/>
  </actuator>
</mujoco>'''
open(f'{D}/scene_tap.xml', 'w').write(scene); print('written')
