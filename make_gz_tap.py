import re
R = '/root/ur5_project/ur5_comparison_ws/src/ur5_experiment'
w = open(f'{R}/worlds/ur5e_peg.world').read()
w, k = re.subn(r'<model name="fixture">.*?</model>', '', w, count=1, flags=re.S)
assert k == 1, 'fixture model not found'
S = ('<surface><friction><ode><mu>0.5</mu><mu2>0.5</mu2></ode></friction>'
     '<contact><ode><kp>1e6</kp><kd>1</kd><min_depth>0</min_depth></ode></contact></surface>')
pad = f'''
    <model name="pad">
      <static>true</static>
      <pose>0.5 0.2 0.41 0 0 0</pose>
      <link name="link">
        <collision name="pad_collision"><geometry><box><size>0.1 0.1 0.02</size></box></geometry>{S}</collision>
        <visual name="pad_v"><geometry><box><size>0.1 0.1 0.02</size></box></geometry>
          <material><ambient>0.3 0.3 0.8 1</ambient></material></visual>
      </link>
    </model>
  </world>'''
assert '</world>' in w
open(f'{R}/worlds/ur5e_tap.world', 'w').write(w.replace('</world>', pad, 1))
print('written')
