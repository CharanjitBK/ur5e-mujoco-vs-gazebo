import re, collections
txt = open('/root/ur5_project/contacts.txt').read()
num = r'([-+0-9.e]+)'
bins = collections.defaultdict(list)
for c in txt.split('\ncontact {\n'):
    if 'cube::link::c' not in c or 'finger_' not in c: continue
    f = 'r' if 'finger_r' in c else 'l'
    names = re.findall(r'body_([12])_name: "([^"]+)"', c)
    wr = re.findall(r'body_([12])_wrench \{\s*force \{\s*x: ' + num + r'\s*y: ' + num + r'\s*z: ' + num, c)
    t = re.search(r'time \{\s*sec: (\d+)\s*nsec: (\d+)', c)
    if len(names) < 2 or len(wr) < 2 or not t: continue
    cube_idx = [i for i, n in names if n.startswith('cube')][0]
    w = [x for x in wr if x[0] == cube_idx][0]
    ts = int(t.group(1)) + int(t.group(2)) * 1e-9
    bins[(round(ts // 0.25 * 0.25, 2), f)].append((float(w[1]), float(w[2]), float(w[3])))
for (t, f), v in sorted(bins.items()):
    n = len(v)
    print(f't={t:8.2f} finger_{f} n={n:4d} cube Fx={sum(a for a,_,_ in v)/n:6.2f} Fy={sum(b for _,b,_ in v)/n:6.2f} Fz={sum(z for _,_,z in v)/n:6.2f} N')
