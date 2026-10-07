import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = '/root/ur5_project'
COL = {'mujoco': '#1f77b4', 'gazebo': '#ff7f0e'}; NAME = {'mujoco': 'MuJoCo', 'gazebo': 'Gazebo'}
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
G = np.linspace(0, 4.0, 400)          # time since start of descent (s)
fig, ax = plt.subplots(figsize=(7.5, 4.8))
for s in ('mujoco', 'gazebo'):
    d = pd.read_csv(f'{R}/data/raw/{s}/peg_in_hole_summary.csv'); curves = []
    for k, ok in zip(d.trial, d.success):
        if not ok: continue
        A = np.load(f'{R}/data/raw/{s}/peg_in_hole_trial_{k:03d}.npy')
        t = A[:, 0]; z = A[:, 15] if s == 'mujoco' else A[:, 9]
        t, i = np.unique(t, return_index=True); z = z[i]
        curves.append(np.interp(G, t - (t[0] + 5.0), z))
    C = np.array(curves)
    ax.fill_between(G, C.min(0), C.max(0), color=COL[s], alpha=0.25)
    ax.plot(G, C.mean(0), color=COL[s], lw=2, label=f'{NAME[s]} (n={len(C)})')
ax.axhline(0.44, c='0.5', ls=':'); ax.text(0.05, 0.442, 'wall tops', color='0.4', fontsize=9)
ax.axhline(0.405, c='r', ls=':'); ax.text(0.05, 0.407, 'success depth', color='r', fontsize=9)
ax.set_xlim(0, 4); ax.set_xlabel('Time since start of descent (s)'); ax.set_ylabel('Peg tip height (m)')
ax.set_title('Peg descent, successful trials (line: mean, band: min-max)'); ax.legend()
fig.text(0.5, 0.01, 'Descent is a fixed 3 s motion; insertion time is about 2.3 s in both simulators.', ha='center', fontsize=9, style='italic')
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f'{R}/results/figures/report/peg_03_insertion_time.png', dpi=200)
print('saved')
