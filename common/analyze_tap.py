import numpy as np, pandas as pd, os
from scipy.stats import linregress, mannwhitneyu
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = '/root/ur5_project'; OUT = f'{R}/results/figures/report'; os.makedirs(OUT, exist_ok=True)
COL = {'mujoco': '#1f77b4', 'gazebo': '#ff7f0e'}; NAME = {'mujoco': 'MuJoCo', 'gazebo': 'Gazebo'}; SIMS = ('mujoco', 'gazebo')
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
rng = np.random.default_rng(0)

mu = pd.read_csv(f'{R}/data/raw/mujoco/tapping_summary.csv').set_index('trial')
gz = pd.read_csv(f'{R}/data/raw/gazebo/tapping_summary.csv').set_index('trial')
gf = pd.read_csv(f'{R}/data/raw/gazebo/tapping_force.csv').set_index('trial')
gc = pd.read_csv(f'{R}/data/raw/gazebo/tapping_force_cycles.csv')

# per-cycle peaks: dict sim -> {trial: array(20)}
cyc = {'mujoco': {}, 'gazebo': {k: g.sort_values('cycle').peak_n.values for k, g in gc.groupby('trial')}}
for k in mu.index:
    F = np.load(f'{R}/data/raw/mujoco/tapping_trial_{k:03d}.npy'); t0 = F[0, 0] - 0.002
    cyc['mujoco'][k] = np.array([F[(F[:, 0] >= t0 + c) & (F[:, 0] < t0 + c + 1), 1].max() for c in range(20)])

rows = []
for s, summ in (('mujoco', mu), ('gazebo', gz)):
    for k in summ.index:
        p = cyc[s][k]
        rows.append(dict(simulator=s, trial=k, jitter_mm=summ.loc[k, 'pad_jitter_mm'], peak_median_n=np.nanmedian(p),
                         peak_mean_n=np.nanmean(p), peak_max_n=np.nanmax(p), spikes_over_70n=int(np.nansum(p > 70)),
                         rtf=summ.loc[k, 'rtf'], mean_cpu=summ.loc[k, 'mean_cpu'], mean_ram_mb=summ.loc[k, 'mean_ram_mb']))
df = pd.DataFrame(rows); df.to_csv(f'{R}/data/processed/tapping_all.csv', index=False)
M, G = df[df.simulator == 'mujoco'], df[df.simulator == 'gazebo']

print('success: MuJoCo %d/20, Gazebo %d/20' % (mu.success.sum(), gz.success.sum()))
for s, d in (('mujoco', M), ('gazebo', G)):
    r = linregress(d.jitter_mm, d.peak_median_n)
    print(f'{NAME[s]}: median-of-cycle-peaks slope vs jitter = {r.slope:.2f} N/mm (R2 {r.rvalue**2:.2f}), '
          f'mean {d.peak_median_n.mean():.1f} N, trials with spikes: {(d.spikes_over_70n > 0).sum()}, total spike cycles: {d.spikes_over_70n.sum()}')
T = []
for c, lab in [('peak_median_n', 'median cycle peak force (N)'), ('rtf', 'RTF'), ('mean_cpu', 'CPU (%)'), ('mean_ram_mb', 'RAM (MB)')]:
    a, b = M[c], G[c]
    T.append(dict(metric=lab, mujoco=f'{a.mean():.3g} ± {a.std():.2g}', gazebo=f'{b.mean():.3g} ± {b.std():.2g}', p=f'{mannwhitneyu(a, b).pvalue:.3g}'))
T = pd.DataFrame(T); print(); print(T.to_string(index=False)); T.to_csv(f'{R}/results/tapping_table.csv', index=False)

def finish(fig, ax, title, ylabel, fname, note=None):
    ax.set_title(title); ax.set_ylabel(ylabel)
    if note: fig.text(0.5, 0.01, note, ha='center', fontsize=9, style='italic')
    fig.tight_layout(rect=(0, 0.05 if note else 0, 1, 1)); fig.savefig(f'{OUT}/{fname}', dpi=200); plt.close(fig)

# 1 force vs jitter
fig, ax = plt.subplots(figsize=(7, 4.8))
for s, d in (('mujoco', M), ('gazebo', G)):
    ax.scatter(d.jitter_mm, d.peak_median_n, color=COL[s], s=40, edgecolor='k', lw=0.4, label=NAME[s], zorder=3)
    r = linregress(d.jitter_mm, d.peak_median_n); x = np.array([-1, 1]); ax.plot(x, r.intercept + r.slope*x, color=COL[s], lw=1.5)
ax.set_xlabel('Equivalent extra penetration (mm)'); ax.legend(loc='upper left'); ax.set_ylim(50, 65)
finish(fig, ax, 'Tap force vs penetration depth', 'Median of 20 cycle peaks (N)', 'tap_01_force_vs_depth.png', 'One dot per trial; lines: least-squares fit.')

# 2 per-cycle peaks: Gazebo clean vs spiky trial, MuJoCo trial 4
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.plot(range(1, 21), cyc['mujoco'][4], 'o-', color=COL['mujoco'], label='MuJoCo trial 4')
ax.plot(range(1, 21), cyc['gazebo'][4], 's--', color=COL['gazebo'], label='Gazebo trial 4 (spiky)')
ax.plot(range(1, 21), cyc['gazebo'][1], '^:', color='tab:green', label='Gazebo trial 1 (clean)')
ax.set_xlabel('Tap cycle'); ax.set_ylim(0, None); ax.legend()
finish(fig, ax, 'Peak force in each tap cycle', 'Peak normal force (N)', 'tap_02_cycle_peaks.png', 'Gazebo spikes occur in isolated cycles.')

# 3 all cycle peaks, strip
fig, ax = plt.subplots(figsize=(5.8, 4.8))
for i, s in enumerate(SIMS):
    v = np.concatenate([cyc[s][k] for k in cyc[s]]); v = v[~np.isnan(v)]
    ax.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=6, color=COL[s], alpha=0.5)
    ax.hlines(np.median(v), i - 0.3, i + 0.3, color='k', lw=2)
    ax.annotate(f'median {np.median(v):.1f}\nmax {v.max():.0f}', (i + 0.33, np.median(v)), fontsize=9, va='center')
ax.set_xticks([0, 1]); ax.set_xticklabels([NAME[s] for s in SIMS]); ax.set_xlim(-0.6, 1.9); ax.set_ylim(0, None)
finish(fig, ax, 'All 400 cycle peaks per simulator', 'Peak normal force (N)', 'tap_03_all_cycles.png', 'Black line = median.')

# 4 spikes per trial (Gazebo)
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.bar(G.trial, G.spikes_over_70n, color=COL['gazebo']); ax.set_xlabel('Trial'); ax.set_xticks(range(1, 21, 2))
finish(fig, ax, 'Gazebo: tap cycles with peak force above 70 N', 'Number of cycles (of 20)', 'tap_04_gazebo_spikes.png', f'MuJoCo: {int(M.spikes_over_70n.sum())} such cycles in 400.')

# 5-7 RTF, CPU, RAM
for c, title, yl, fn, note, log in [('rtf', 'Real-time factor (log scale)', 'Sim time / wall time', 'tap_05_rtf.png', 'Gazebo capped at real time (update rate 500).', True),
        ('mean_cpu', 'CPU usage', 'CPU time / wall time (%)', 'tap_06_cpu.png', 'Gazebo: gzserver + gzclient + runner + contact recorder. MuJoCo: one process.', False),
        ('mean_ram_mb', 'Memory usage', 'RAM (MB)', 'tap_07_ram.png', 'Gazebo: gzserver + gzclient + runner. MuJoCo: one process.', False)]:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if log: ax.set_yscale('log')
    for i, s in enumerate(SIMS):
        v = df[df.simulator == s][c].values
        ax.bar(i, v.mean(), 0.55, color=COL[s], alpha=0.75, yerr=v.std(), capsize=8)
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=12, color='k', alpha=0.5, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels([NAME[s] for s in SIMS]); finish(fig, ax, title, yl, fn, note)
print('saved tap_01..tap_07 to', OUT)
