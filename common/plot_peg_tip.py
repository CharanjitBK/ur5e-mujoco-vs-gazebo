import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = '/root/ur5_project'
COL = {'mujoco': '#1f77b4', 'gazebo': '#ff7f0e'}; NAME = {'mujoco': 'MuJoCo', 'gazebo': 'Gazebo'}
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})

def load(s, k):
    A = np.load(f'{R}/data/raw/{s}/peg_in_hole_trial_{k:03d}.npy')
    t = A[:, 0]; z = A[:, 15] if s == 'mujoco' else A[:, 9]
    return t - t[0], z

fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for ax, k, title in zip(axs, (1, 2), ('Trial 1: peg inserts (both simulators)', 'Trial 2: peg jams on wall edge (both simulators)')):
    for s, lw, ls in (('mujoco', 3.0, '-'), ('gazebo', 1.6, '--')):
        t, z = load(s, k)
        ax.plot(t, z, color=COL[s], lw=lw, ls=ls, label=NAME[s])
        ax.annotate(f'{z[-1]:.3f} m', (t[-1], z[-1]), xytext=(-4, 8 if s == 'mujoco' else -16),
                    textcoords='offset points', ha='right', fontsize=10, color=COL[s])
    ax.axhline(0.44, c='0.5', ls=':'); ax.text(1.9, 0.443, 'wall tops (0.44 m)', color='0.4', fontsize=9)
    ax.axhline(0.405, c='r', ls=':'); ax.text(1.9, 0.395, 'success depth (0.405 m)', color='r', fontsize=9)
    ax.axvline(5.0, c='0.85'); ax.text(5.05, 0.64, 'descent starts', color='0.5', fontsize=9, va='top')
    ax.set_title(title, fontsize=12); ax.set_xlabel('Time since start of measured motion (s)')
axs[0].set_ylabel('Peg tip height (m)'); axs[0].legend(loc='upper right')
fig.text(0.5, 0.005, 'Gazebo tip comes from TF at ~100 Hz, so its curve is stepped (a sampling effect, not rough motion).',
         ha='center', fontsize=9, style='italic')
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f'{R}/results/figures/report/peg_05_tip_height.png', dpi=200)
print('saved')
