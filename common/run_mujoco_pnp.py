import mujoco, numpy as np, json, time
XML = '/root/ur5_project/mujoco_ur5/models/working_ur5e/scene_workcell.xml'
W = json.load(open('/root/ur5_project/common/waypoints.json'))
m = mujoco.MjModel.from_xml_path(XML); d = mujoco.MjData(m)
cube = m.body('cube').id
OPEN, CLOSED = -8.0, 8.0

def quintic(a, b, T, t):
    s = np.clip(t / T, 0, 1); return a + (b - a) * (10*s**3 - 15*s**4 + 6*s**5)

d.qpos[:6] = W['home']; d.ctrl[:6] = W['home']; d.ctrl[6:8] = OPEN
mujoco.mj_forward(m, d)
for _ in range(500): mujoco.mj_step(m, d)          # settle 1 s

log = []
def move(name, T, grip=None):
    q0 = d.ctrl[:6].copy(); q1 = np.array(W[name]); t0 = d.time
    while d.time - t0 < T:
        d.ctrl[:6] = quintic(q0, q1, T, d.time - t0)
        if grip is not None: d.ctrl[6:8] = grip
        mujoco.mj_step(m, d)
        log.append([d.time, *d.qpos[:6], *d.site('attachment_site').xpos, *d.xpos[cube]])
def hold(T, grip):
    t0 = d.time; d.ctrl[6:8] = grip
    while d.time - t0 < T: mujoco.mj_step(m, d)

wall = time.time(); sim0 = d.time
move('above_cube', 3, OPEN)
move('grasp', 2, OPEN)
print('fingertip bottom z at grasp:', d.body('finger_r').xpos[2] - 0.04, '(table top 0.40)')
hold(1.0, CLOSED)
move('above_cube', 2, CLOSED)
print('cube z after lift:', d.xpos[cube][2], '(lifted if > 0.5)')
move('above_target', 3, CLOSED)
move('place', 2, CLOSED)
hold(0.5, CLOSED); hold(1.0, OPEN)
move('above_target', 2, OPEN)
hold(1.0, OPEN)
p = d.xpos[cube]; err = np.linalg.norm(p[:2] - [0.5, 0.2])
ok = err < 0.01 and abs(p[2] - 0.42) < 0.01
print(f'cube final: {p.round(4)}  xy_err={err*1000:.1f} mm  SUCCESS={ok}')
print(f'sim time {d.time - sim0:.1f} s  wall {time.time()-wall:.2f} s  RTF {(d.time-sim0)/(time.time()-wall):.1f}')
np.save('/root/ur5_project/mujoco_ur5/results/pnp_trial_test.npy', np.array(log))
