import numpy as np, pandas as pd, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

R = '/root/ur5_project'
OUT = f'{R}/results/figures/report'; os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(f'{R}/data/processed/pick_place_all.csv')
SIMS = ('mujoco', 'gazebo')
COL = {'mujoco': '#1f77b4', 'gazebo': '#ff7f0e'}
NAME = {'mujoco': 'MuJoCo', 'gazebo': 'Gazebo'}
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
rng = np.random.default_rng(0)

def vals(s, c): return df[df.simulator == s][c].values

def finish(fig, ax, title, ylabel, fname, note=None):
    ax.set_xticks([0, 1]); ax.set_xticklabels([NAME[s] for s in SIMS])
    ax.set_title(title); ax.set_ylabel(ylabel)
    if note: fig.text(0.5, 0.01, note, ha='center', fontsize=9, style='italic')
    fig.tight_layout(rect=(0, 0.05 if note else 0, 1, 1))
    fig.savefig(f'{OUT}/{fname}', dpi=200); plt.close(fig)

def bar(metric, title, ylabel, fname, note=None, fmt='%.3g'):
    """Mean bar (zero-based) + std error bar + every trial as a dot."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    top = max(vals(s, metric).max() for s in SIMS)
    top = max(top, max(vals(s, metric).mean() + vals(s, metric).std() for s in SIMS))
    ax.set_ylim(0, top * 1.25)
    for i, s in enumerate(SIMS):
        v = vals(s, metric)
        ax.bar(i, v.mean(), 0.55, color=COL[s], alpha=0.75, yerr=v.std(), capsize=8)
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color='k', alpha=0.6, zorder=3)
        ax.text(i, top * 1.08, (fmt + ' ± ' + fmt) % (v.mean(), v.std()), ha='center', fontsize=11)
    finish(fig, ax, title, ylabel, fname, note)

def strip(metric, title, ylabel, fname, note=None, limit=None, log=False, fmt='%.3g'):
    """Dot plot: every trial, mean as a horizontal line. Optional limit line / log axis."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    allv = np.concatenate([vals(s, metric) for s in SIMS])
    if log: ax.set_yscale('log')
    for i, s in enumerate(SIMS):
        v = vals(s, metric)
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=28, color=COL[s], edgecolor='k', lw=0.4, zorder=3)
        ax.hlines(v.mean(), i - 0.3, i + 0.3, color='k', lw=2, zorder=4)
        ax.annotate((fmt + ' ± ' + fmt) % (v.mean(), v.std()), (i + 0.33, v.mean()),
                    va='center', fontsize=10)
    if limit is not None:
        ax.axhline(limit, color='r', ls='--', lw=1.2, label=f'Success limit ({limit:g})'); ax.legend(loc='lower right')
        allv = np.append(allv, limit)
    if log: ax.set_ylim(allv.min() / 2, allv.max() * 3)
    else:   ax.set_ylim(0, allv.max() * 1.2)
    ax.set_xlim(-0.6, 1.9)
    finish(fig, ax, title, ylabel, fname, note)

# 1. success rate
fig, ax = plt.subplots(figsize=(5.5, 4.5))
for i, s in enumerate(SIMS):
    v = df[df.simulator == s].success
    ax.bar(i, 100 * v.mean(), 0.55, color=COL[s], alpha=0.75)
    ax.text(i, 100 * v.mean() + 2, f'{v.sum()}/{len(v)}', ha='center', fontsize=12)
ax.set_ylim(0, 115)
finish(fig, ax, 'Pick-and-place success rate', 'Success rate (%)', '01_success_rate.png')

# 2-8. metrics
strip('place_err_mm', 'Placement error (every trial)', 'Cube-to-target error (mm)', '02_placement_error.png',
      limit=10, note='Black line = mean. Same start offsets (seeded) in both simulators.')
bar('path_len_m', 'End-effector path length', 'Path length (m)', '03_path_length.png',
    note='Same trajectory in both simulators; axis starts at zero.', fmt='%.4g')
strip('rms_acc', 'Motion smoothness (RMS joint acceleration)', 'RMS acceleration (rad/s²)', '04_rms_acceleration.png',
      note='Partly reflects the different arm controllers.')
strip('track_rms_rad', 'Trajectory tracking error', 'RMS tracking error (rad)', '05_tracking_error.png',
      note='Partly reflects the different arm controllers.', fmt='%.4f')
strip('rtf', 'Real-time factor (log scale)', 'Sim time / wall time', '06_rtf.png', log=True,
      note='Gazebo capped at real time (update rate 500): not an engine-speed comparison.', fmt='%.2f')
bar('mean_cpu', 'CPU usage', 'Mean CPU (% of one core)', '07_cpu.png',
    note='Gazebo: gzserver + gzclient + runner. MuJoCo: one Python process.')
bar('mean_ram_mb', 'Memory usage', 'Mean RAM (MB)', '08_ram.png',
    note='Gazebo: gzserver + gzclient + runner. MuJoCo: one Python process.')

# 9. placement error across trials
fig, ax = plt.subplots(figsize=(7, 4.5))
for s in SIMS:
    d = df[df.simulator == s]; ax.plot(d.trial, d.place_err_mm, 'o-', color=COL[s], label=NAME[s])
ax.axhline(10, color='r', ls='--', lw=1, label='Success limit (10 mm)')
ax.set_xlabel('Trial'); ax.set_ylabel('Placement error (mm)'); ax.set_title('Placement error per trial')
ax.set_ylim(0, 11); ax.legend(); fig.tight_layout()
fig.savefig(f'{OUT}/09_error_per_trial.png', dpi=200); plt.close(fig)

# 10-11. trajectories from the raw logs (trial 1)
def load(sim, k=1):
    A = np.load(f'{R}/data/raw/{sim}/pick_place_trial_{k:03d}.npy')
    t = A[:, 0] - A[0, 0]
    return t, A[:, 1:7], (A[:, 13:19] if sim == 'mujoco' else A[:, 7:13])

names = ['Shoulder pan', 'Shoulder lift', 'Elbow', 'Wrist 1', 'Wrist 2', 'Wrist 3']
fig, axs = plt.subplots(2, 3, figsize=(13, 7)); axs = axs.ravel()
for j in range(6):
    for s in SIMS:
        t, q, qd = load(s); axs[j].plot(t, q[:, j], color=COL[s], label=NAME[s], lw=1.5)
    t, q, qd = load('gazebo'); axs[j].plot(t, qd[:, j], 'k--', lw=0.8, label='Reference')
    axs[j].set_title(names[j]); axs[j].set_xlabel('Time (s)'); axs[j].set_ylabel('Angle (rad)')
axs[0].legend(); fig.suptitle('Joint trajectories, trial 1'); fig.tight_layout()
fig.savefig(f'{OUT}/10_joint_trajectories.png', dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.5))
for s in SIMS:
    t, q, qd = load(s); ax.plot(t, np.linalg.norm(q - qd, axis=1), color=COL[s], label=NAME[s])
ax.set_xlabel('Time (s)'); ax.set_ylabel('Joint error norm (rad)'); ax.set_title('Tracking error over time, trial 1')
ax.legend(); fig.tight_layout(); fig.savefig(f'{OUT}/11_tracking_error_time.png', dpi=200); plt.close(fig)

print('saved to', OUT); print(sorted(os.listdir(OUT)))
