import mujoco, mujoco.viewer, numpy as np, json, time
XML = '/root/ur5_project/mujoco_ur5/models/working_ur5e/scene_workcell.xml'
W = json.load(open('/root/ur5_project/common/waypoints.json'))
m = mujoco.MjModel.from_xml_path(XML); d = mujoco.MjData(m)
OPEN, CLOSED = -8.0, 8.0
def quintic(a, b, T, t):
    s = np.clip(t / T, 0, 1); return a + (b - a) * (10*s**3 - 15*s**4 + 6*s**5)
d.qpos[:6] = W['home']; d.ctrl[:6] = W['home']; d.ctrl[6:8] = OPEN
mujoco.mj_forward(m, d)
with mujoco.viewer.launch_passive(m, d) as v:
    def step():
        t = time.time(); mujoco.mj_step(m, d); v.sync()
        time.sleep(max(0, m.opt.timestep - (time.time() - t)))
    def move(name, T, g):
        q0 = d.ctrl[:6].copy(); q1 = np.array(W[name]); t0 = d.time
        while d.time - t0 < T and v.is_running():
            d.ctrl[:6] = quintic(q0, q1, T, d.time - t0); d.ctrl[6:8] = g; step()
    def hold(T, g):
        t0 = d.time; d.ctrl[6:8] = g
        while d.time - t0 < T and v.is_running(): step()
    hold(1.0, OPEN)
    move('above_cube', 3, OPEN); move('grasp', 2, OPEN); hold(1.0, CLOSED)
    move('above_cube', 2, CLOSED); move('above_target', 3, CLOSED); move('place', 2, CLOSED)
    hold(0.5, CLOSED); hold(1.0, OPEN); move('above_target', 2, OPEN); hold(1.0, OPEN)
    print('cube final:', d.body('cube').xpos.round(4))
    while v.is_running(): step()
