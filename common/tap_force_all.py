import re, numpy as np, pandas as pd, os
R = '/root/ur5_project/data/raw/gazebo'
S = pd.read_csv(f'{R}/tapping_summary.csv'); num = r'([-+0-9.e]+)'; per, cyc = [], []
for k, t0 in zip(S.trial, S.t_start):
    p = f'{R}/tap_contacts_trial_{k:03d}.txt'
    if not os.path.exists(p): print(k, 'missing'); continue
    rows = []
    for c in open(p).read().split('\ncontact {\n'):
        if 'peg_collision_2' not in c or 'pad_collision' not in c: continue
        t = re.search(r'time \{\s*sec: (\d+)\s*nsec: (\d+)', c)
        names = re.findall(r'body_([12])_name: "([^"]+)"', c)
        wr = re.findall(r'body_([12])_wrench \{\s*force \{\s*x: ' + num + r'\s*y: ' + num + r'\s*z: ' + num, c)
        if not t or len(names) < 2 or len(wr) < 2: continue
        idx = [i for i, nm in names if 'pad' in nm][0]
        w = [x for x in wr if x[0] == idx][0]
        rows.append((int(t.group(1)) + int(t.group(2))*1e-9, abs(float(w[3]))))
    A = np.array(rows)
    if not len(A): print(k, 'no contacts'); continue
    pk, ct = [], []
    for c in range(20):
        s = A[(A[:, 0] >= t0 + c) & (A[:, 0] < t0 + c + 1)]
        pk.append(s[:, 1].max() if len(s) else np.nan)
        ct.append(len(np.unique(s[s[:, 1] > 1.0, 0])) * 0.002 if len(s) else 0)
        cyc.append(dict(trial=k, cycle=c + 1, peak_n=pk[-1]))
    pk = np.array(pk)
    per.append(dict(trial=k, cycles_with_contact=int(np.sum(~np.isnan(pk))), peak_median_n=round(float(np.nanmedian(pk)), 2),
                    peak_mean_n=round(float(np.nanmean(pk)), 2), peak_max_n=round(float(np.nanmax(pk)), 2),
                    spikes_over_70n=int(np.nansum(pk > 70)), contact_time_mean_s=round(float(np.mean(ct)), 3)))
    print(per[-1], flush=True)
pd.DataFrame(per).to_csv(f'{R}/tapping_force.csv', index=False)
pd.DataFrame(cyc).to_csv(f'{R}/tapping_force_cycles.csv', index=False); print('saved')
