import re, collections
txt = open('/root/ur5_project/contacts.txt').read()
chunks = txt.split('\ncontact {\n')
bins = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0])
num = r'([-+0-9.e]+)'
for c in chunks:
    if 'cube::link::c' not in c or 'finger_' not in c: continue
    f = 'r' if 'finger_r' in c else 'l'
    t = re.search(r'time \{\s*sec: (\d+)\s*nsec: (\d+)', c)
    if not t: continue
    ts = int(t.group(1)) + int(t.group(2)) * 1e-9
    sx = sz = 0.0
    for w in re.finditer(r'body_1_wrench \{\s*force \{\s*x: ' + num + r'\s*y: ' + num + r'\s*z: ' + num, c):
        sx += float(w.group(1)); sz += float(w.group(3))
    sign = 1 if 'collision1: "cube' in c else -1
    b = bins[(round(ts // 0.25 * 0.25, 2), f)]
    b[0] += 1; b[1] += abs(sx); b[2] += sz * sign; b[3] += 0
out = open('/root/ur5_project/contact_summary.txt', 'w')
for (t, f), (n, fx, fz, _) in sorted(bins.items()):
    out.write(f't={t:8.2f} finger_{f} n={n:4d} mean|Fx|={fx/n:6.2f} N  mean Fz_on_cube={fz/n:6.2f} N\n')
out.close()
print('bins:', len(bins))
