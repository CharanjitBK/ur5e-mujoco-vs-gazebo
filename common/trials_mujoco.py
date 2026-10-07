import mujoco, numpy as np, json, time, csv, os, sys, threading, psutil
XML = '/root/ur5_project/mujoco_ur5/models/working_ur5e/scene_workcell.xml'
W = json.load(open('/root/ur5_project/common/waypoints.json'))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
OUT = '/root/ur5_project/data/raw/mujoco'; os.makedirs(OUT, exist_ok=True)
OPEN, CLOSED = -8.0, 8.0
m = mujoco.MjModel.from_xml_path(XML); d = mujoco.MjData(m)
cadr = m.jnt_qposadr[m.joint('cube_free').id]
site = m.site('attachment_site').id; cube = m.body('cube').id
proc = psutil.Process()

def quintic(a, b, T, t):
    s = np.clip(t / T, 0, 1); return a + (b - a) * (10*s**3 - 15*s**4 + 6*s**5)

def run_trial(k):
    rng = np.random.default_rng(42 + k); off = rng.uniform(-0.005, 0.005, 2)
    mujoco.mj_resetData(m, d)
    d.qpos[:6] = W['home']; d.ctrl[:6] = W['home']; d.ctrl[6:8] = OPEN
    d.qpos[cadr:cadr+3] = [0.5 + off[0], -0.2 + off[1], 0.42]
    mujoco.mj_forward(m, d)
    for _ in range(500): mujoco.mj_step(m, d)          # settle 1 s (not measured)
    rows = []; cpu = []; ram = []; stop = [False]
    def sampler():
        proc.cpu_percent(None)
        while not stop[0]:
            time.sleep(0.2); cpu.append(proc.cpu_percent(None)); ram.append(proc.memory_info().rss / 2**20)
    th = threading.Thread(target=sampler, daemon=True); th.start()
    def step():
        mujoco.mj_step(m, d)
        rows.append([d.time, *d.qpos[:6], *d.qvel[:6], *d.ctrl[:6], *d.site_xpos[site], *d.xpos[cube]])
    def move(name, T, g):
        q0 = d.ctrl[:6].copy(); q1 = np.array(W[name]); t0 = d.time
        while d.time - t0 < T:
            d.ctrl[:6] = quintic(q0, q1, T, d.time - t0); d.ctrl[6:8] = g; step()
    def hold(T, g):
        t0 = d.time; d.ctrl[6:8] = g
        while d.time - t0 < T: step()
    wall0 = time.time(); sim0 = d.time
    move('above_cube', 3, OPEN); move('grasp', 2, OPEN); hold(1.0, CLOSED)
    move('above_cube', 2, CLOSED); lift_z = d.xpos[cube][2]
    move('above_target', 3, CLOSED); move('place', 2, CLOSED)
    hold(0.5, CLOSED); hold(1.0, OPEN); move('above_target', 2, OPEN); hold(1.0, OPEN)
    sim_t = d.time - sim0; wall_t = time.time() - wall0
    stop[0] = True; th.join()
    A = np.array(rows); dt = m.opt.timestep
    p = d.xpos[cube].copy(); err = float(np.linalg.norm(p[:2] - [0.5, 0.2]))
    ee = A[:, 19:22]; qv = A[:, 7:13]
    return A, dict(
        simulator='mujoco', task='pick_place', trial=k, offx_mm=round(off[0]*1e3, 2), offy_mm=round(off[1]*1e3, 2),
        success=int(err < 0.01 and abs(p[2] - 0.42) < 0.01 and lift_z > 0.5),
        place_err_mm=round(err*1e3, 2), lift_z=round(lift_z, 4), sim_time=round(sim_t, 2), wall_time=round(wall_t, 3),
        rtf=round(sim_t / wall_t, 2), path_len_m=round(float(np.linalg.norm(np.diff(ee, axis=0), axis=1).sum()), 4),
        rms_acc=round(float(np.sqrt(np.mean((np.diff(qv, axis=0) / dt) ** 2))), 3),
        track_rms_rad=round(float(np.sqrt(np.mean((A[:, 1:7] - A[:, 13:19]) ** 2))), 5),
        mean_cpu=round(float(np.mean(cpu)) if cpu else 0, 1), max_cpu=round(float(np.max(cpu)) if cpu else 0, 1),
        mean_ram_mb=round(float(np.mean(ram)) if ram else 0, 1), max_ram_mb=round(float(np.max(ram)) if ram else 0, 1))

res = []
for k in range(1, N + 1):
    A, r = run_trial(k); res.append(r)
    np.save(f'{OUT}/pick_place_trial_{k:03d}.npy', A)
    print(k, 'success', r['success'], 'err_mm', r['place_err_mm'], 'rtf', r['rtf'], 'cpu', r['mean_cpu'])
with open(f'{OUT}/pick_place_summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
print('success rate: %d/%d' % (sum(r['success'] for r in res), N))
