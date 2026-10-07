import re, sys, numpy as np
k = int(sys.argv[1])
txt = open(f'/root/ur5_project/data/raw/gazebo/tap_contacts_trial_{k:03d}.txt').read()
num = r'([-+0-9.e]+)'; rows = []
for c in txt.split('\ncontact {\n'):
    if 'peg_collision_2' not in c or 'pad_collision' not in c: continue
    t = re.search(r'time \{\s*sec: (\d+)\s*nsec: (\d+)', c)
    names = re.findall(r'body_([12])_name: "([^"]+)"', c)
    wr = re.findall(r'body_([12])_wrench \{\s*force \{\s*x: ' + num + r'\s*y: ' + num + r'\s*z: ' + num, c)
    if not t or len(names) < 2 or len(wr) < 2: continue
    idx = [i for i, nm in names if 'pad' in nm][0]
    w = [x for x in wr if x[0] == idx][0]
    rows.append((int(t.group(1)) + int(t.group(2)) * 1e-9, float(np.linalg.norm([float(w[1]), float(w[2]), float(w[3])]))))
R = np.array(rows); print('peg-pad contact samples:', len(R))
if len(R):
    R = R[np.argsort(R[:, 0])]
    # group samples into 1 s cycles from the first contact
    t0 = R[0, 0]; cyc = ((R[:, 0] - t0) // 1.0).astype(int)
    peaks = [R[cyc == c, 1].max() for c in np.unique(cyc)]
    print('first contact at t=%.2f, cycles seen: %d' % (t0, len(peaks)))
    print('per-cycle peak force (N):', np.round(peaks, 1))
    print('overall peak %.1f N, mean of cycle peaks %.1f N' % (R[:, 1].max(), np.mean(peaks)))
