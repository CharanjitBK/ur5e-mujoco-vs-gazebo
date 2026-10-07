import numpy as np, pandas as pd, json, mujoco, os
from scipy.signal import savgol_filter
from scipy.stats import mannwhitneyu

R = '/root/ur5_project'; HZ = 100.0; N = 20
W = json.load(open(f'{R}/common/waypoints.json'))
m = mujoco.MjModel.from_xml_path(f'{R}/mujoco_ur5/models/working_ur5e/scene_workcell.xml')
d = mujoco.MjData(m); site = m.site('attachment_site').id

PH = [(3, 'above_cube'), (2, 'grasp'), (1, None), (2, 'above_cube'), (3, 'above_target'),
      (2, 'place'), (1.5, None), (2, 'above_target'), (1, None)]
edges = np.concatenate([[0], np.cumsum([p[0] for p in PH])])

def ref(t):
    t = np.asarray(t); out = np.zeros((len(t), 6)); q = np.array(W['home'])
    for (T, wp), a in zip(PH, edges[:-1]):
        q1 = np.array(W[wp]) if wp else q
        s = np.clip((t - a) / T, 0, 1); p = 10*s**3 - 15*s**4 + 6*s**5
        mk = t >= a; out[mk] = q + np.outer(p[mk], q1 - q); q = q1
    return out

G = np.arange(0, edges[-1], 1 / HZ)

def load(sim, k):
    A = np.load(f'{R}/data/raw/{sim}/pick_place_trial_{k:03d}.npy')
    t = A[:, 0] - A[0, 0]; q = A[:, 1:7]
    t, i = np.unique(t, return_index=True); q = q[i]
    return np.column_stack([np.interp(G, t, q[:, j]) for j in range(6)])

# peak reference velocity per joint during the approach move (clip check)
tt = np.arange(0, 3, 0.001); rv = np.abs(np.gradient(ref(tt), 0.001, axis=0)).max(0)
print('peak reference joint speed in approach (rad/s), Gazebo clip = 1.0:')
print('  ', dict(zip(['pan', 'lift', 'elbow', 'w1', 'w2', 'w3'], rv.round(2))))

OFFS = np.arange(-0.10, 0.1001, 0.005)
def best_offset(Q):
    e = [np.sqrt(np.mean((Q - ref(G + o))**2)) for o in OFFS]
    return OFFS[int(np.argmin(e))]

frames = []; apr = {}
for sim in ('mujoco', 'gazebo'):
    s = pd.read_csv(f'{R}/data/raw/{sim}/pick_place_summary.csv')
    s = s[[c for c in s.columns if c not in ('path_len_m', 'rms_acc', 'track_rms_rad')]]
    rows = []; offs = []; apk = []
    for k in s.trial:
        Q = load(sim, k)
        acc = savgol_filter(Q, 15, 3, deriv=2, delta=1 / HZ, axis=0)
        ee = []
        for row in Q:
            d.qpos[:6] = row; mujoco.mj_kinematics(m, d); ee.append(d.site_xpos[site].copy())
        ee = np.array(ee); err = Q - ref(G)
        offs.append(best_offset(Q)); apk.append(np.abs(err[G < 3]).max(0))
        rows.append(dict(path_len_m=np.linalg.norm(np.diff(ee, axis=0), axis=1).sum(),
                         rms_acc=np.sqrt(np.mean(acc**2)),
                         track_rms_rad=np.sqrt(np.mean(err**2)),
                         time_offset_s=offs[-1]))
    frames.append(pd.concat([s, pd.DataFrame(rows)], axis=1))
    apr[sim] = np.mean(apk, axis=0)
    print(f'{sim}: median best time offset = {np.median(offs):+.3f} s (range {min(offs):+.3f}..{max(offs):+.3f})')
print('mean peak |error| per joint in approach (rad):')
for sim in apr: print('  ', sim, dict(zip(['pan', 'lift', 'elbow', 'w1', 'w2', 'w3'], apr[sim].round(3))))

df = pd.concat(frames, ignore_index=True)
df.to_csv(f'{R}/data/processed/pick_place_all.csv', index=False)
M = ['place_err_mm', 'path_len_m', 'rms_acc', 'track_rms_rad', 'rtf', 'mean_cpu', 'mean_ram_mb']
mu, gz = df[df.simulator == 'mujoco'], df[df.simulator == 'gazebo']
rows = [dict(metric='success rate', mujoco=f'{mu.success.sum()}/{len(mu)}', gazebo=f'{gz.success.sum()}/{len(gz)}', p='')]
for c in M:
    a, b = mu[c], gz[c]
    rows.append(dict(metric=c, mujoco=f'{a.mean():.3g} ± {a.std():.2g}', gazebo=f'{b.mean():.3g} ± {b.std():.2g}',
                     p=f'{mannwhitneyu(a, b).pvalue:.3g}'))
T = pd.DataFrame(rows); print(); print(T.to_string(index=False))
T.to_csv(f'{R}/results/pick_place_table.csv', index=False)
