import numpy as np, pandas as pd, mujoco, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

R = '/root/ur5_project'
m = mujoco.MjModel.from_xml_path(f'{R}/mujoco_ur5/models/working_ur5e/scene_workcell.xml')
d = mujoco.MjData(m)
site = m.site('attachment_site').id
HZ = 100.0

def cols(sim, A):
    # mujoco log: [t, q(6), qvel(6), ctrl(6), ee(3), cube(3)]
    # gazebo log: [t, q(6), q_desired(6)]
    if sim == 'mujoco':
        return A[:, 0], A[:, 1:7], A[:, 13:19]
    return A[:, 0], A[:, 1:7], A[:, 7:13]

def metrics(sim, k):
    A = np.load(f'{R}/data/raw/{sim}/pick_place_trial_{k:03d}.npy')
    t, q, qd = cols(sim, A)
    t, i = np.unique(t, return_index=True)
    q, qd = q[i], qd[i]
    g = np.arange(t[0], t[-1], 1 / HZ)                       # common 100 Hz grid
    Q = np.column_stack([np.interp(g, t, q[:, j]) for j in range(6)])
    QD = np.column_stack([np.interp(g, t, qd[:, j]) for j in range(6)])
    acc = np.gradient(np.gradient(Q, 1 / HZ, axis=0), 1 / HZ, axis=0)
    ee = []
    for row in Q:
        d.qpos[:6] = row
        mujoco.mj_kinematics(m, d)
        ee.append(d.site_xpos[site].copy())
    ee = np.array(ee)
    return dict(path_len_m=np.linalg.norm(np.diff(ee, axis=0), axis=1).sum(),
                rms_acc=np.sqrt(np.mean(acc ** 2)),
                track_rms_rad=np.sqrt(np.mean((Q - QD) ** 2)))

frames = []
for sim in ('mujoco', 'gazebo'):
    s = pd.read_csv(f'{R}/data/raw/{sim}/pick_place_summary.csv')
    s = s[[c for c in s.columns if c not in ('path_len_m', 'rms_acc', 'track_rms_rad')]]
    extra = pd.DataFrame([metrics(sim, k) for k in s.trial])
    frames.append(pd.concat([s, extra], axis=1))
df = pd.concat(frames, ignore_index=True)

os.makedirs(f'{R}/data/processed', exist_ok=True)
os.makedirs(f'{R}/results/figures', exist_ok=True)
df.to_csv(f'{R}/data/processed/pick_place_all.csv', index=False)

M = ['place_err_mm', 'path_len_m', 'rms_acc', 'track_rms_rad', 'rtf', 'mean_cpu', 'mean_ram_mb']
mu, gz = df[df.simulator == 'mujoco'], df[df.simulator == 'gazebo']
rows = [dict(metric='success rate',
             mujoco=f'{mu.success.sum()}/{len(mu)}',
             gazebo=f'{gz.success.sum()}/{len(gz)}', p='')]
for c in M:
    a, b = mu[c], gz[c]
    p = mannwhitneyu(a, b).pvalue if len(a) > 1 and len(b) > 1 else float('nan')
    rows.append(dict(metric=c, mujoco=f'{a.mean():.3g} ± {a.std():.2g}',
                     gazebo=f'{b.mean():.3g} ± {b.std():.2g}', p=f'{p:.3g}'))
T = pd.DataFrame(rows)
print(T.to_string(index=False))
T.to_csv(f'{R}/results/pick_place_table.csv', index=False)

fig, ax = plt.subplots(2, 4, figsize=(16, 7))
ax = ax.ravel()
for a, c in zip(ax, M):
    data = [mu[c], gz[c]]
    try:
        a.boxplot(data, tick_labels=['MuJoCo', 'Gazebo'])    # matplotlib >= 3.9
    except TypeError:
        a.boxplot(data, labels=['MuJoCo', 'Gazebo'])         # older versions
    a.set_title(c)
ax[-1].axis('off')
plt.tight_layout()
plt.savefig(f'{R}/results/figures/pick_place_comparison.png', dpi=150)
print('saved figure')
