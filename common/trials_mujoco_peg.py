import mujoco, numpy as np, json, time, csv, os, sys, re, threading, psutil
D = '/root/ur5_project/mujoco_ur5/models/working_ur5e'
W = json.load(open('/root/ur5_project/common/waypoints_peg.json'))
args = sys.argv[1:]; VIEW = 'view' in args
nums = [a for a in args if a.isdigit()]; N = int(nums[0]) if nums else 20
CLR = 0.003; H = 0.010 + CLR; T = 0.008          # 3 mm clearance per side
OUT = '/root/ur5_project/data/raw/mujoco'; os.makedirs(OUT, exist_ok=True)

# scene with the 3 mm hole
F = 'friction="0.5 0.005 0.0001" rgba="0.3 0.3 0.8 1"'
fx = f'''<body name="fixture" pos="0.5 0.2 0.42">
      <geom name="wall_px" type="box" size="{T} {H+2*T} 0.02" pos="{H+T} 0 0" {F}/>
      <geom name="wall_nx" type="box" size="{T} {H+2*T} 0.02" pos="{-(H+T)} 0 0" {F}/>
      <geom name="wall_py" type="box" size="{H} {T} 0.02" pos="0 {H+T} 0" {F}/>
      <geom name="wall_ny" type="box" size="{H} {T} 0.02" pos="0 {-(H+T)} 0" {F}/>
    </body>'''
base = open(f'{D}/scene_peg.xml').read()
open(f'{D}/scene_peg_3mm.xml', 'w').write(re.sub(r'<body name="fixture".*?</body>', lambda _: fx, base, flags=re.S))
m = mujoco.MjModel.from_xml_path(f'{D}/scene_peg_3mm.xml'); d = mujoco.MjData(m)
gid = m.geom('peg').id; fid = m.body('fixture').id
walls = {m.geom(n).id for n in ('wall_px', 'wall_nx', 'wall_py', 'wall_ny')}
site = m.site('attachment_site').id
proc = psutil.Process()
OPEN = -8.0
q5 = lambda a, b, Td, t: a + (b - a) * (lambda s: 10*s**3 - 15*s**4 + 6*s**5)(np.clip(t / Td, 0, 1))

def run_trial(k, v=None):
    rng = np.random.default_rng(42 + k); off = rng.uniform(-0.0015, 0.0015, 2)
    hole = np.array([0.5 + off[0], 0.2 + off[1]])
    if v: v.lock().__enter__()
    mujoco.mj_resetData(m, d)
    m.body_pos[fid] = [hole[0], hole[1], 0.42]
    d.qpos[:6] = W['home']; d.ctrl[:6] = W['home']; d.ctrl[6:8] = OPEN
    mujoco.mj_forward(m, d)
    if v: v.lock().__exit__(None, None, None)
    rows = []; cpu = []; ram = []; stop = [False]; state = {'fmax': 0.0, 'tins': None, 'rec': False, 'n': 0}

    def step():
        mujoco.mj_step(m, d)
        tot = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 == gid and c.geom2 in walls) or (c.geom2 == gid and c.geom1 in walls):
                f = np.zeros(6); mujoco.mj_contactForce(m, d, i, f); tot += abs(f[0])
        tip = np.append(d.geom_xpos[gid][:2], d.geom_xpos[gid][2] - 0.05)
        if state['rec']:
            state['fmax'] = max(state['fmax'], tot)
            if state['tins'] is None and state.get('desc') is not None and tip[2] <= 0.405: state['tins'] = d.time - state['desc']
            rows.append([d.time, *d.qpos[:6], *d.ctrl[:6], *tip, tot])
        if v:
            state['n'] += 1
            if state['n'] % 8 == 0:
                v.sync(); lag = (d.time - state['s0']) - (time.time() - state['w0'])
                if lag > 0: time.sleep(lag)

    def seg(a, b, Td):
        q0, q1 = np.array(W[a]), np.array(W[b]); t0 = d.time
        while d.time - t0 < Td and (v is None or v.is_running()):
            d.ctrl[:6] = q5(q0, q1, Td, d.time - t0); step()

    state['s0'] = d.time; state['w0'] = time.time()
    seg('home', 'home', 1.0)                                  # settle (not measured)
    def sampler():
        proc.cpu_percent(None)
        while not stop[0]:
            time.sleep(0.2); cpu.append(proc.cpu_percent(None)); ram.append(proc.memory_info().rss / 2**20)
    if not VIEW: threading.Thread(target=sampler, daemon=True).start()
    state['rec'] = True; state['t0'] = d.time; sim0 = d.time; wall0 = time.time(); cpu0 = sum(proc.cpu_times()[:2])
    seg('home', 'lift', 2.5); seg('lift', 'above_hole', 2.5); state['desc'] = d.time; seg('above_hole', 'insert', 3); seg('insert', 'insert', 1.0)
    sim_t = d.time - sim0; wall_t = time.time() - wall0; stop[0] = True
    tip = np.append(d.geom_xpos[gid][:2], d.geom_xpos[gid][2] - 0.05)
    xy_err = np.linalg.norm(tip[:2] - hole)
    ok = bool(tip[2] <= 0.405 and xy_err < CLR)               # depth >= 35 mm and still inside the hole
    r = dict(simulator='mujoco', task='peg_in_hole', trial=k, clearance_mm=CLR*1e3,
             offx_mm=round(off[0]*1e3, 2), offy_mm=round(off[1]*1e3, 2), success=int(ok),
             tip_z=round(float(tip[2]), 4), xy_err_mm=round(xy_err*1e3, 2),
             insert_time_s=round(state['tins'], 2) if state['tins'] is not None else float('nan'),
             peak_force_n=round(state['fmax'], 1), sim_time=round(sim_t, 2), wall_time=round(wall_t, 3),
             rtf=round(sim_t / wall_t, 2),
             mean_cpu=round(100 * (sum(proc.cpu_times()[:2]) - cpu0) / wall_t, 1), mean_ram_mb=round(proc.memory_info().rss / 2**20, 1))
    print(k, 'success', r['success'], 'tip_z', r['tip_z'], 'xy_err_mm', r['xy_err_mm'],
          'peak_N', r['peak_force_n'], 'rtf', r['rtf'], flush=True)
    return np.array(rows), r

res = []
if VIEW:
    import mujoco.viewer
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = [0.5, 0.2, 0.5]; v.cam.distance = 0.9; v.cam.azimuth = 140; v.cam.elevation = -25
        for k in range(1, N + 1):
            if not v.is_running(): break
            run_trial(k, v)
        while v.is_running(): v.sync(); time.sleep(0.03)
else:
    for k in range(1, N + 1):
        A, r = run_trial(k); res.append(r); np.save(f'{OUT}/peg_in_hole_trial_{k:03d}.npy', A)
    with open(f'{OUT}/peg_in_hole_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    print('success rate: %d/%d' % (sum(r['success'] for r in res), N))
