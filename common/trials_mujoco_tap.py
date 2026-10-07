import mujoco, numpy as np, json, time, csv, os, sys, psutil
D = '/root/ur5_project/mujoco_ur5/models/working_ur5e'
W = json.load(open('/root/ur5_project/common/waypoints_tap.json'))
args = sys.argv[1:]; VIEW = 'view' in args
nums = [a for a in args if a.isdigit()]; N = int(nums[0]) if nums else 20
CYC = 20
JW = json.load(open('/root/ur5_project/common/waypoints_tap_jitter.json'))
OUT = '/root/ur5_project/data/raw/mujoco'; os.makedirs(OUT, exist_ok=True)
m = mujoco.MjModel.from_xml_path(f'{D}/scene_tap.xml'); d = mujoco.MjData(m)
peg, pad = m.geom('peg').id, m.geom('pad').id; pid = m.body('padbody').id
proc = psutil.Process()
q5 = lambda a, b, T, t: a + (b - a) * (lambda s: 10*s**3 - 15*s**4 + 6*s**5)(np.clip(t/T, 0, 1))

def run_trial(k, v=None):
    jit = np.random.default_rng(42 + k).uniform(-0.001, 0.001)
    mujoco.mj_resetData(m, d); m.body_pos[pid] = [0.5, 0.2, 0.41]
    d.qpos[:6] = W['home']; d.ctrl[:6] = W['home']; d.ctrl[6:8] = -8; mujoco.mj_forward(m, d)
    Wt = dict(W); Wt['tap_down'] = JW[str(k)]['q']
    st = {'rec': False, 'n': 0, 's0': d.time, 'w0': time.time()}; F = []; Q = []
    def step():
        mujoco.mj_step(m, d)
        if st['rec']:
            f = 0.0
            for i in range(d.ncon):
                c = d.contact[i]
                if {c.geom1, c.geom2} == {peg, pad}:
                    w = np.zeros(6); mujoco.mj_contactForce(m, d, i, w); f += abs(w[0])
            F.append((d.time, f)); Q.append(d.qvel[:6].copy())
        if v:
            st['n'] += 1
            if st['n'] % 8 == 0:
                v.sync(); lag = (d.time - st['s0']) - (time.time() - st['w0'])
                if lag > 0: time.sleep(lag)
    def seg(a, b, T):
        q0, q1 = np.array(Wt[a]), np.array(Wt[b]); t0 = d.time
        while d.time - t0 < T and (v is None or v.is_running()):
            d.ctrl[:6] = q5(q0, q1, T, d.time - t0); step()
    seg('home', 'home', 1.0); seg('home', 'lift', 2.5); seg('lift', 'tap_up', 2.5)
    st['rec'] = True; t_start = d.time; sim0 = d.time; wall0 = time.time(); cpu0 = sum(proc.cpu_times()[:2])
    for c in range(CYC): seg('tap_up', 'tap_down', 0.5); seg('tap_down', 'tap_up', 0.5)
    sim_t = d.time - sim0; wall_t = time.time() - wall0
    cpu = 100 * (sum(proc.cpu_times()[:2]) - cpu0) / wall_t; st['rec'] = False
    seg('tap_up', 'lift', 2.0)
    F = np.array(F); Q = np.array(Q); peaks = []; cts = []
    for c in range(CYC):
        s = F[(F[:, 0] >= t_start + c) & (F[:, 0] < t_start + c + 1)]
        peaks.append(s[:, 1].max()); cts.append((s[:, 1] > 1.0).sum() * m.opt.timestep)
    peaks = np.array(peaks); acc = np.diff(Q, axis=0) / m.opt.timestep
    r = dict(simulator='mujoco', task='tapping', trial=k, pad_jitter_mm=round(jit*1e3, 2),
             success=int((peaks > 1.0).all()), cycles_valid=int((peaks > 1.0).sum()),
             peak_force_mean_n=round(peaks.mean(), 2), peak_force_std_n=round(peaks.std(), 2),
             peak_force_max_n=round(peaks.max(), 2), contact_time_mean_s=round(float(np.mean(cts)), 3),
             rms_acc=round(float(np.sqrt(np.mean(acc**2))), 3), sim_time=round(sim_t, 2),
             wall_time=round(wall_t, 3), rtf=round(sim_t/wall_t, 2), mean_cpu=round(cpu, 1),
             mean_ram_mb=round(proc.memory_info().rss / 2**20, 1))
    print(k, 'success', r['success'], 'jit_mm', r['pad_jitter_mm'], 'peak_mean_N', r['peak_force_mean_n'],
          'contact_s', r['contact_time_mean_s'], 'rtf', r['rtf'], flush=True)
    return np.array(F), r

if VIEW:
    import mujoco.viewer
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = [0.5, 0.2, 0.5]; v.cam.distance = 1.0; v.cam.azimuth = 140; v.cam.elevation = -20
        for k in range(1, N + 1):
            if not v.is_running(): break
            run_trial(k, v)
        while v.is_running(): v.sync(); time.sleep(0.03)
else:
    res = []
    for k in range(1, N + 1):
        F, r = run_trial(k); res.append(r); np.save(f'{OUT}/tapping_trial_{k:03d}.npy', F)
    with open(f'{OUT}/tapping_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    print('success rate: %d/%d' % (sum(r['success'] for r in res), N))
