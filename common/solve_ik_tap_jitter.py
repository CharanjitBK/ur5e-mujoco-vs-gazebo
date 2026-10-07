import mujoco, numpy as np, json
m = mujoco.MjModel.from_xml_path('/root/ur5_project/mujoco_ur5/models/working_ur5e/scene_tap.xml'); d = mujoco.MjData(m)
site = m.site('attachment_site').id
W = json.load(open('/root/ur5_project/common/waypoints_tap.json'))
down = np.array([[1,0,0],[0,-1,0],[0,0,-1]], float)
def solve(target, q0):
    q = np.array(q0, float)
    for _ in range(300):
        d.qpos[:6] = q; mujoco.mj_forward(m, d)
        pe = target - d.site_xpos[site]; R = d.site_xmat[site].reshape(3, 3)
        re = 0.5 * sum(np.cross(R[:, i], down[:, i]) for i in range(3))
        if np.linalg.norm(pe) < 1e-5 and np.linalg.norm(re) < 1e-4: break
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv)); mujoco.mj_jacSite(m, d, jp, jr, site)
        J = np.vstack([jp[:, :6], jr[:, :6]])
        q = q + 0.5 * (J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.concatenate([pe, re])))
    d.qpos[:6] = q; mujoco.mj_forward(m, d)
    return q, np.linalg.norm(target - d.site_xpos[site])
out = {}
for k in range(1, 21):
    jit = float(np.random.default_rng(42 + k).uniform(-0.001, 0.001))
    q, e = solve(np.array([0.5, 0.2, 0.535 - jit]), W['tap_down'])
    out[str(k)] = {'jit_mm': round(jit*1e3, 3), 'q': [round(float(x), 5) for x in q]}
    print(f"trial {k:2d}  jitter {jit*1e3:+.2f} mm  target z {0.535-jit:.4f}  err {e*1000:.3f} mm")
json.dump(out, open('/root/ur5_project/common/waypoints_tap_jitter.json', 'w'), indent=1)
