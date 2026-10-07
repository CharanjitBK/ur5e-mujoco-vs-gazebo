import numpy as np, pandas as pd, os
from scipy.stats import binomtest, mannwhitneyu
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

R = '/root/ur5_project'; OUT = f'{R}/results/figures/report'; os.makedirs(OUT, exist_ok=True)
SIMS = ('mujoco', 'gazebo'); COL = {'mujoco': '#1f77b4', 'gazebo': '#ff7f0e'}; NAME = {'mujoco': 'MuJoCo', 'gazebo': 'Gazebo'}
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
rng = np.random.default_rng(0)

def load(sim, k): return np.load(f'{R}/data/raw/{sim}/peg_in_hole_trial_{k:03d}.npy')
def parts(sim, A):   # time, tip(x,y,z)
    return (A[:, 0], A[:, 13:16]) if sim == 'mujoco' else (A[:, 0], A[:, 7:10])

def ins_time(sim, k, hole):
    t, tip = parts(sim, load(sim, k)); desc = t[0] + 5.0     # lift 2.5 s + above_hole 2.5 s
    ok = (t >= desc) & (tip[:, 2] <= 0.405) & (np.linalg.norm(tip[:, :2] - hole, axis=1) < 0.005)
    return float(t[ok][0] - desc) if ok.any() else np.nan

frames = []
for s in SIMS:
    d = pd.read_csv(f'{R}/data/raw/{s}/peg_in_hole_summary.csv')
    d['insert_time_s'] = [ins_time(s, k, np.array([0.5 + ox/1e3, 0.2 + oy/1e3]))
                          for k, ox, oy in zip(d.trial, d.offx_mm, d.offy_mm)]
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
os.makedirs(f'{R}/data/processed', exist_ok=True); df.to_csv(f'{R}/data/processed/peg_in_hole_all.csv', index=False)
g = {s: df[df.simulator == s].set_index('trial') for s in SIMS}

def wilson(k, n, z=1.96):
    p = k / n; c = (p + z*z/(2*n)) / (1 + z*z/n); h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n); return c - h, c + h

# ---- table
rows = []
for s in SIMS:
    d = g[s]; k = int(d.success.sum()); lo, hi = wilson(k, len(d))
    rows.append((s, k, len(d), lo, hi, sorted(d.index[d.success == 0].tolist())))
    print(f'{NAME[s]}: {k}/{len(d)} inserted (95% CI {100*lo:.0f}-{100*hi:.0f}%), jammed trials: {rows[-1][5]}')
a, b = g['mujoco'].success, g['gazebo'].success
only_mu_jam = int(((a == 0) & (b == 1)).sum()); only_gz_jam = int(((a == 1) & (b == 0)).sum())
both_jam = int(((a == 0) & (b == 0)).sum()); both_ok = int(((a == 1) & (b == 1)).sum())
nd = only_mu_jam + only_gz_jam
p_mc = binomtest(min(only_mu_jam, only_gz_jam), nd, 0.5).pvalue if nd else 1.0
print(f'paired outcomes: both inserted {both_ok}, both jammed {both_jam}, MuJoCo-only jam {only_mu_jam}, Gazebo-only jam {only_gz_jam}; exact McNemar p = {p_mc:.3f}')

def ms(s, c, only_ok=False):
    d = g[s]; v = (d[d.success == 1] if only_ok else d)[c].dropna(); return v
T = []
for c, ok, lab in [('insert_time_s', True, 'insertion time (s, successes)'), ('xy_err_mm', True, 'final xy error (mm, successes)'),
                   ('peak_force_n', False, 'peak wall force (N)'), ('rtf', False, 'RTF'),
                   ('mean_cpu', False, 'CPU (%)'), ('mean_ram_mb', False, 'RAM (MB)')]:
    x, y = ms('mujoco', c, ok), ms('gazebo', c, ok)
    p = mannwhitneyu(x, y).pvalue if len(x) > 1 and len(y) > 1 else np.nan
    f = lambda v: f'{v.mean():.3g} ± {v.std():.2g} (n={len(v)})' if len(v) else 'not measured'
    T.append(dict(metric=lab, mujoco=f(x), gazebo=f(y), p=f'{p:.3g}'))
T = pd.DataFrame(T); print(); print(T.to_string(index=False)); T.to_csv(f'{R}/results/peg_in_hole_table.csv', index=False)

def finish(fig, ax, title, ylabel, fname, note=None):
    ax.set_xticks([0, 1]); ax.set_xticklabels([NAME[s] for s in SIMS]); ax.set_title(title); ax.set_ylabel(ylabel)
    if note: fig.text(0.5, 0.01, note, ha='center', fontsize=9, style='italic')
    fig.tight_layout(rect=(0, 0.05 if note else 0, 1, 1)); fig.savefig(f'{OUT}/{fname}', dpi=200); plt.close(fig)

# 1 success rate with Wilson CI
fig, ax = plt.subplots(figsize=(5.5, 4.5))
for i, (s, k, n, lo, hi, _) in enumerate(rows):
    ax.bar(i, 100*k/n, 0.55, color=COL[s], alpha=0.75, yerr=[[100*k/n - 100*lo], [100*hi - 100*k/n]], capsize=8)
    ax.text(i, 100*hi + 2, f'{k}/{n}', ha='center')
ax.set_ylim(0, 120)
finish(fig, ax, 'Peg-in-hole success rate (3 mm clearance)', 'Success rate (%)', 'peg_01_success_rate.png',
       'Bars: observed rate; error bars: 95% Wilson interval. Same 20 seeded hole offsets.')

# 2 outcome map: which hole offsets jam
fig, axs = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
for ax, s in zip(axs, SIMS):
    d = g[s]; ok = d.success == 1
    ax.scatter(d.offx_mm[ok], d.offy_mm[ok], c='tab:green', s=60, label='inserted')
    ax.scatter(d.offx_mm[~ok], d.offy_mm[~ok], c='tab:red', marker='X', s=80, label='jammed')
    for k in d.index: ax.annotate(str(k), (d.offx_mm[k], d.offy_mm[k]), fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax.set_title(NAME[s]); ax.set_xlabel('Hole offset x (mm)'); ax.axhline(0, c='0.8', lw=0.6); ax.axvline(0, c='0.8', lw=0.6)
axs[0].set_ylabel('Hole offset y (mm)'); axs[0].legend(loc='lower left')
fig.suptitle('Outcome vs hole offset (labels = trial number)'); fig.tight_layout()
fig.savefig(f'{OUT}/peg_02_outcome_map.png', dpi=200); plt.close(fig)

# 3-4 insertion time, final xy error (successes only): dot plots
def strip(c, title, ylabel, fname, note):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for i, s in enumerate(SIMS):
        v = ms(s, c, True).values
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=28, color=COL[s], edgecolor='k', lw=0.4, zorder=3)
        if len(v): ax.hlines(v.mean(), i - 0.3, i + 0.3, color='k', lw=2); ax.annotate(f'{v.mean():.3g} ± {v.std():.2g}', (i + 0.33, v.mean()), va='center', fontsize=10)
    ax.set_xlim(-0.6, 1.9); ax.set_ylim(bottom=0); finish(fig, ax, title, ylabel, fname, note)
strip('insert_time_s', 'Insertion time (successful trials)', 'Time from start of descent (s)', 'peg_03_insertion_time.png',
      'Descent is a fixed 3 s motion, so this mostly reflects tracking, not contact.')
strip('xy_err_mm', 'Final tip position error (successful trials)', 'Tip-to-hole-centre error (mm)', 'peg_04_xy_error.png',
      'MuJoCo tip from the physics state; Gazebo tip from TF (tool0 minus 0.12 m).')

# 5 tip height over time: trial 1 (inserts in both) and trial 2 (jams in both)
fig, ax = plt.subplots(figsize=(8, 4.5))
for s in SIMS:
    for k, ls in ((1, '-'), (2, '--')):
        t, tip = parts(s, load(s, k)); ax.plot(t - t[0], tip[:, 2], ls, color=COL[s], label=f'{NAME[s]} trial {k}')
ax.axhline(0.405, c='r', lw=0.8, ls=':'); ax.text(0.2, 0.407, 'success depth', color='r', fontsize=9)
ax.axhline(0.44, c='0.6', lw=0.8, ls=':'); ax.text(0.2, 0.442, 'wall tops', color='0.4', fontsize=9)
ax.axvline(5.0, c='0.8'); ax.set_xlabel('Time since start of measured motion (s)'); ax.set_ylabel('Peg tip height (m)')
ax.set_title('Peg tip height: trial 1 inserts, trial 2 jams'); ax.legend(fontsize=9); fig.tight_layout()
fig.savefig(f'{OUT}/peg_05_tip_height.png', dpi=200); plt.close(fig)

# 6 MuJoCo peak force per trial (Gazebo force not measured yet)
d = g['mujoco']; fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(d.index, d.peak_force_n, color=[('tab:red' if s == 0 else COL['mujoco']) for s in d.success])
ax.set_xlabel('Trial'); ax.set_ylabel('Peak peg-wall force (N)'); ax.set_title('MuJoCo peak contact force per trial (red = jammed)')
fig.text(0.5, 0.01, 'Gazebo contact force not measured yet.', ha='center', fontsize=9, style='italic')
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f'{OUT}/peg_06_force_mujoco.png', dpi=200); plt.close(fig)

# 7-9 RTF, CPU, RAM
def bar(c, title, ylabel, fname, note, log=False):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if log: ax.set_yscale('log')
    for i, s in enumerate(SIMS):
        v = df[df.simulator == s][c].values
        ax.bar(i, v.mean(), 0.55, color=COL[s], alpha=0.75, yerr=v.std(), capsize=8)
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=12, color='k', alpha=0.5, zorder=3)
        ax.text(i, v.mean() * (1.3 if log else 1.02) + (0 if log else v.std()), f'{v.mean():.3g}', ha='center')
    finish(fig, ax, title, ylabel, fname, note)
bar('rtf', 'Real-time factor (log scale)', 'Sim time / wall time', 'peg_07_rtf.png', 'Gazebo capped at real time (update rate 500).', True)
bar('mean_cpu', 'CPU usage', 'CPU time / wall time (%)', 'peg_08_cpu.png', 'Gazebo: gzserver + gzclient + runner. MuJoCo: one Python process.')
bar('mean_ram_mb', 'Memory usage', 'RAM (MB)', 'peg_09_ram.png', 'Gazebo: gzserver + gzclient + runner. MuJoCo: one Python process.')
print('\nsaved peg_01..peg_09 to', OUT)
