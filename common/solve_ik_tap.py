import mujoco, numpy as np, json
m = mujoco.MjModel.from_xml_path('/root/ur5_project/mujoco_ur5/models/working_ur5e/scene_tap.xml'); d = mujoco.MjData(m)
site = m.site('attachment_site').id
home = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
down = np.array([[1,0,0],[0,-1,0],[0,0,-1]], float)
def solve(target, q0):
    q = q0.copy()
    for _ in range(300):
        d.qpos[:6] = q; mujoco.mj_forward(m, d)
        pe = target - d.site_xpos[site]; R = d.site_xmat[site].reshape(3, 3)
        re = 0.5 * sum(np.cross(R[:, i], down[:, i]) for i in range(3))
        if np.linalg.norm(pe) < 1e-4 and np.linalg.norm(re) < 1e-3: break
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv)); mujoco.mj_jacSite(m, d, jp, jr, site)
        J = np.vstack([jp[:, :6], jr[:, :6]])
        q = q + 0.5 * (J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.concatenate([pe, re])))
    d.qpos[:6] = q; mujoco.mj_forward(m, d)
    return q, np.linalg.norm(target - d.site_xpos[site])
# site = tip + 0.12: tap_up tip 0.45, tap_down tip 0.415 (pad top 0.42)
out, q = {}, home
for k, p in {'lift': [0.35, 0.0, 0.75], 'tap_up': [0.5, 0.2, 0.57], 'tap_down': [0.5, 0.2, 0.535]}.items():
    q, e = solve(np.array(p), q); out[k] = [round(float(x), 4) for x in q]
    print(f'{k:9s} pos_err={e*1000:.2f} mm  q={out[k]}')
json.dump({'home': home.tolist(), **out}, open('/root/ur5_project/common/waypoints_tap.json', 'w'), indent=1)
