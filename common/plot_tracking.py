import numpy as np, pandas as pd, json, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

R = '/root/ur5_project'; OUT = f'{R}/results/figures/report'; os.makedirs(OUT, exist_ok=True)
W = json.load(open(f'{R}/common/waypoints.json'))
COL = {'mujoco': '#1f77b4', 'gazebo': '#ff7f0e'}; NAME = {'mujoco': 'MuJoCo', 'gazebo': 'Gazebo'}
SIMS = ('mujoco', 'gazebo'); N = 20
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})

# (phase name, duration, target waypoint or None = hold)
PH = [('approach', 3, 'above_cube'), ('grasp', 2, 'grasp'), ('close', 1, None), ('lift', 2, 'above_cube'),
      ('transfer', 3, 'above_target'), ('place', 2, 'place'), ('release', 1.5, None),
      ('retreat', 2, 'above_target'), ('settle', 1, None)]
edges = np.concatenate([[0], np.cumsum([p[1] for p in PH])])

def ref(t):
    t = np.asarray(t); out = np.zeros((len(t), 6)); q = np.array(W['home'])
    for (name, T, wp), a in zip(PH, edges[:-1]):
        q1 = np.array(W[wp]) if wp else q
        s = np.clip((t - a) / T, 0, 1); p = 10*s**3 - 15*s**4 + 6*s**5
        m = t >= a; out[m] = q + np.outer(p[m], q1 - q)
        q = q1
    return out

def load(sim, k):
    A = np.load(f'{R}/data/raw/{sim}/pick_place_trial_{k:03d}.npy')
    t = A[:, 0] - A[0, 0]; return t, A[:, 1:7]

G = np.arange(0, edges[-1], 0.01)
dev = {s: [] for s in SIMS}; rms = {s: [] for s in SIMS}
for s in SIMS:
    for k in range(1, N + 1):
        t, q = load(s, k); t, i = np.unique(t, return_index=True); q = q[i]
        Q = np.column_stack([np.interp(G, t, q[:, j]) for j in range(6)])
        d = Q - ref(G); dev[s].append(d)
        e = np.linalg.norm(d, axis=1)
        rms[s].append([np.sqrt(np.mean(e[(G >= a) & (G < b)]**2)) for a, b in zip(edges[:-1], edges[1:])])
dev = {s: np.array(v) for s, v in dev.items()}; rms = {s: np.array(v) for s, v in rms.items()}

def shade(ax, labels=True):
    for i, ((n, T, _), a) in enumerate(zip(PH, edges[:-1])):
        if i % 2 == 0: ax.axvspan(a, a + T, color='0.93', zorder=0)
        if labels: ax.text(a + T/2, 1.01, n, transform=ax.get_xaxis_transform(), ha='center', fontsize=8, rotation=30)

# 12. overall tracking error vs time, mean +/- std over 20 trials
fig, ax = plt.subplots(figsize=(10, 4.5))
for s in SIMS:
    e = np.linalg.norm(dev[s], axis=2); m, sd = e.mean(0), e.std(0)
    ax.plot(G, m, color=COL[s], label=NAME[s], lw=1.8); ax.fill_between(G, m - sd, m + sd, color=COL[s], alpha=0.25)
shade(ax); ax.set_xlim(0, edges[-1]); ax.set_ylim(bottom=0)
ax.set_xlabel('Time (s)'); ax.set_ylabel('Joint error norm (rad)'); ax.legend(loc='upper right')
ax.set_title('Tracking error vs analytic reference (mean ± std, 20 trials)', pad=28)
fig.text(0.5, 0.01, 'Reference rebuilt from waypoints.json; same definition for both simulators.', ha='center', fontsize=9, style='italic')
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f'{OUT}/12_tracking_error_mean.png', dpi=200); plt.close(fig)

# 13. tracking RMS per phase
fig, ax = plt.subplots(figsize=(10, 4.5)); x = np.arange(len(PH)); w = 0.38; rng = np.random.default_rng(0)
for j, s in enumerate(SIMS):
    v = rms[s]; xs = x + (j - 0.5) * w
    ax.bar(xs, v.mean(0), w, color=COL[s], alpha=0.75, yerr=v.std(0), capsize=4, label=NAME[s])
    ax.scatter(np.repeat(xs, N) + rng.uniform(-0.06, 0.06, v.size), v.T.ravel(), s=6, color='k', alpha=0.4, zorder=3)
ax.set_xticks(x); ax.set_xticklabels([p[0] for p in PH]); ax.set_ylabel('RMS joint error (rad)')
ax.set_title('Tracking error by task phase (bars: mean ± std, dots: trials)'); ax.legend()
fig.tight_layout(); fig.savefig(f'{OUT}/13_tracking_by_phase.png', dpi=200); plt.close(fig)

# 14. per-joint deviation from reference (mrad), mean over trials
names = ['Shoulder pan', 'Shoulder lift', 'Elbow', 'Wrist 1', 'Wrist 2', 'Wrist 3']
fig, axs = plt.subplots(2, 3, figsize=(14, 7), sharex=True); axs = axs.ravel()
for j in range(6):
    for s in SIMS:
        axs[j].plot(G, 1000 * dev[s][:, :, j].mean(0), color=COL[s], label=NAME[s], lw=1.5)
    axs[j].axhline(0, color='k', lw=0.6); shade(axs[j], labels=False)
    axs[j].set_title(names[j]); axs[j].set_ylabel('Deviation (mrad)')
    if j >= 3: axs[j].set_xlabel('Time (s)')
axs[0].legend(); fig.suptitle('Joint deviation from reference (mean of 20 trials; + = ahead of reference)')
fig.tight_layout(); fig.savefig(f'{OUT}/14_joint_deviation.png', dpi=200); plt.close(fig)

print('overall mean RMS per sim (rad):', {s: round(float(np.sqrt((rms[s]**2).mean())), 4) for s in SIMS})
print(sorted(os.listdir(OUT)))
