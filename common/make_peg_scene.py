D = '/root/ur5_project/mujoco_ur5/models/working_ur5e'
s = open(f'{D}/ur5e.xml').read()
anchor = '<body name="finger_r"'
assert anchor in s
peg = ('<geom name="peg" type="box" size="0.01 0.01 0.05" pos="0 0 0.07" mass="0.1" '
       'friction="0.5 0.005 0.0001" rgba="0.8 0.6 0.1 1"/>\n                    ')
open(f'{D}/ur5e_peg.xml', 'w').write(s.replace(anchor, peg + anchor, 1))

H, T = 0.012, 0.008          # hole half-width (24 mm hole), wall half-thickness
F = 'friction="0.5 0.005 0.0001" rgba="0.3 0.3 0.8 1"'
scene = f'''<mujoco model="ur5e_peg">
  <include file="ur5e_peg.xml"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual><headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/></visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.8 0.8 0.8 1" friction="1 0.005 0.0001"/>
    <geom name="table" type="box" size="0.25 0.4 0.2" pos="0.55 0 0.2" rgba="0.6 0.5 0.4 1" friction="0.5 0.005 0.0001"/>
    <body name="fixture" pos="0.5 0.2 0.42">
      <geom name="wall_px" type="box" size="{T} {H+2*T} 0.02" pos="{H+T} 0 0" {F}/>
      <geom name="wall_nx" type="box" size="{T} {H+2*T} 0.02" pos="{-(H+T)} 0 0" {F}/>
      <geom name="wall_py" type="box" size="{H} {T} 0.02" pos="0 {H+T} 0" {F}/>
      <geom name="wall_ny" type="box" size="{H} {T} 0.02" pos="0 {-(H+T)} 0" {F}/>
    </body>
  </worldbody>
  <actuator>
    <motor name="finger_r" joint="finger_r_joint" ctrlrange="-8 8"/>
    <motor name="finger_l" joint="finger_l_joint" ctrlrange="-8 8"/>
  </actuator>
</mujoco>'''
open(f'{D}/scene_peg.xml', 'w').write(scene)
print('written')
