import mujoco, mujoco.viewer, numpy as np, json, time, sys, os
XML = '/root/ur5_project/mujoco_ur5/models/working_ur5e/scene_workcell.xml'
W = json.load(open('/root/ur5_project/common/waypoints.json'))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
RECORD = 'record' in sys.argv
OUT = '/root/ur5_project/results/videos'; os.makedirs(OUT, exist_ok=True)
OPEN, CLOSED = -8.0, 8.0
m = mujoco.MjModel.from_xml_path(XML); d = mujoco.MjData(m)
cadr = m.jnt_qposadr[m.joint('cube_free').id]
dt = m.opt.timestep

def quintic(a, b, T, t):
    s = np.clip(t / T, 0, 1); return a + (b - a) * (10*s**3 - 15*s**4 + 6*s**5)

def reset(k):
    off = np.random.default_rng(42 + k).uniform(-0.005, 0.005, 2)
    mujoco.mj_resetData(m, d)
    d.qpos[:6] = W['home']; d.ctrl[:6] = W['home']; d.ctrl[6:8] = OPEN
    d.qpos[cadr:cadr+3] = [0.5 + off[0], -0.2 + off[1], 0.42]
    mujoco.mj_forward(m, d)

def run(k, frame_cb, alive=lambda: True):
    reset(k)
    def step():
        mujoco.mj_step(m, d); frame_cb()
    def hold(T, g):
        t0 = d.time; d.ctrl[6:8] = g
        while d.time - t0 < T and alive(): step()
    def move(name, T, g):
        q0 = d.ctrl[:6].copy(); q1 = np.array(W[name]); t0 = d.time
        while d.time - t0 < T and alive():
            d.ctrl[:6] = quintic(q0, q1, T, d.time - t0); d.ctrl[6:8] = g; step()
    hold(1.0, OPEN)
    move('above_cube', 3, OPEN); move('grasp', 2, OPEN); hold(1.0, CLOSED)
    move('above_cube', 2, CLOSED); move('above_target', 3, CLOSED); move('place', 2, CLOSED)
    hold(0.5, CLOSED); hold(1.0, OPEN); move('above_target', 2, OPEN); hold(1.0, OPEN)
    p = d.body('cube').xpos; err = np.linalg.norm(p[:2] - [0.5, 0.2])
    ok = err < 0.01 and abs(p[2] - 0.42) < 0.01
    print(f'trial {k}: SUCCESS={ok}  place_err={err*1000:.1f} mm', flush=True)

if RECORD:
    import imageio
    m.vis.global_.offwidth = 1280; m.vis.global_.offheight = 720
    r = mujoco.Renderer(m, 720, 1280)
    cam = mujoco.MjvCamera(); cam.lookat[:] = [0.45, 0.0, 0.5]
    cam.distance = 1.6; cam.azimuth = 140; cam.elevation = -25
    FPS = 30; every = int(round(1 / (FPS * dt))); cnt = [0]
    for k in range(1, N + 1):
        w = imageio.get_writer(f'{OUT}/mujoco_trial_{k:02d}.mp4', fps=FPS)
        def cb():
            cnt[0] += 1
            if cnt[0] % every == 0:
                r.update_scene(d, camera=cam); w.append_data(r.render())
        run(k, cb); w.close()
        print('saved', f'{OUT}/mujoco_trial_{k:02d}.mp4', flush=True)
else:
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = [0.45, 0.0, 0.5]; v.cam.distance = 1.6
        v.cam.azimuth = 140; v.cam.elevation = -25
        state = {'t': time.time(), 'n': 0}
        def cb():
            state['n'] += 1
            if state['n'] % 8 == 0:               # sync about 60 Hz
                v.sync()
                lag = d.time - (time.time() - state['t'])
                if lag > 0: time.sleep(lag)       # real-time pacing
        for k in range(1, N + 1):
            if not v.is_running(): break
            state['t'] = time.time() - d.time * 0   # reset clock per trial
            run(k, cb, v.is_running)
        while v.is_running(): v.sync(); time.sleep(0.03)
