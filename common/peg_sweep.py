import mujoco, numpy as np, json
D = '/root/ur5_project/mujoco_ur5/models/working_ur5e'
W = json.load(open('/root/ur5_project/common/waypoints_peg.json'))
base = open(f'{D}/scene_peg.xml').read()
import re
T = 0.008
q5 = lambda a, b, T_, t: a + (b - a) * (lambda s: 10*s**3 - 15*s**4 + 6*s**5)(np.clip(t/T_, 0, 1))

def scene(H):
    F = 'friction="0.5 0.005 0.0001" rgba="0.3 0.3 0.8 1"'
    fx = f'''<body name="fixture" pos="0.5 0.2 0.42">
      <geom name="wall_px" type="box" size="{T} {H+2*T} 0.02" pos="{H+T} 0 0" {F}/>
      <geom name="wall_nx" type="box" size="{T} {H+2*T} 0.02" pos="{-(H+T)} 0 0" {F}/>
      <geom name="wall_py" type="box" size="{H} {T} 0.02" pos="0 {H+T} 0" {F}/>
      <geom name="wall_ny" type="box" size="{H} {T} 0.02" pos="0 {-(H+T)} 0" {F}/>
    </body>'''
    return re.sub(r'<body name="fixture".*?</body>', fx, base, flags=re.S)

for clr in (1, 2, 3, 4, 6):                      # clearance per side, mm
    H = 0.010 + clr / 1000
    open(f'{D}/_tmp_peg.xml', 'w').write(scene(H))
    m = mujoco.MjModel.from_xml_path(f'{D}/_tmp_peg.xml'); d = mujoco.MjData(m)
    d.qpos[:6] = W['home']; d.ctrl[:6] = W['home']; d.ctrl[6:8] = -8; mujoco.mj_forward(m, d)
    gid = m.geom('peg').id; fmax = 0.0
    def seg(a, b, Td):
        global fmax
        q0, q1 = np.array(W[a]), np.array(W[b]); t0 = d.time
        while d.time - t0 < Td:
            d.ctrl[:6] = q5(q0, q1, Td, d.time - t0); mujoco.mj_step(m, d)
            for i in range(d.ncon):
                c = d.contact[i]; n = {m.geom(c.geom1).name, m.geom(c.geom2).name}
                if 'peg' in n and any(x.startswith('wall') for x in n):
                    f = np.zeros(6); mujoco.mj_contactForce(m, d, i, f); fmax = max(fmax, abs(f[0]))
    seg('home', 'home', 1.0); seg('home', 'above_hole', 3); seg('above_hole', 'insert', 3); seg('insert', 'insert', 1.0)
    p = d.geom_xpos[gid]
    print(f'clearance {clr} mm/side: tip z={p[2]-0.05:.4f}  xy=({p[0]:.4f},{p[1]:.4f})  peak wall force={fmax:.1f} N  inserted={p[2]-0.05 < 0.415}')
