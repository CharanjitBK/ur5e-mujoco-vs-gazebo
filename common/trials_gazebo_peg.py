import sys, json, time, csv, os, re, threading, numpy as np, psutil, rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Float64MultiArray
from gazebo_msgs.srv import SetEntityState
from tf2_ros import Buffer, TransformListener

W = json.load(open('/root/ur5_project/common/waypoints_peg.json'))
JN = ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint']
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
OUT = '/root/ur5_project/data/raw/gazebo'; os.makedirs(OUT, exist_ok=True)
OPEN = -8.0; KP = 10.0; CLR = 0.003

def quintic(a, b, T, t):
    s = np.clip(t / T, 0, 1); d = b - a
    return a + d*(10*s**3 - 15*s**4 + 6*s**5), d*(30*s**2 - 60*s**3 + 30*s**4) / T

class R(Node):
    def __init__(self):
        super().__init__('peg_trials', parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.scli = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.fpub = self.create_publisher(Float64MultiArray, '/finger_effort_controller/commands', 10)
        self.vpub = self.create_publisher(Float64MultiArray, '/forward_velocity_controller/commands', 10)
        self.ref = None; self.qm = None; self.simt = 0.0
        self.tfb = Buffer(); self.tfl = TransformListener(self.tfb, self)
        self.create_timer(0.02, lambda: self.fpub.publish(Float64MultiArray(data=[OPEN, OPEN])))
        self.create_timer(0.005, self.vtick)
        self.create_subscription(Clock, '/clock', lambda m: setattr(self, 'simt', m.clock.sec + m.clock.nanosec*1e-9), qos_profile_sensor_data)
        self.create_subscription(JointState, '/joint_states', self.on_js, qos_profile_sensor_data)
        self.rec = False; self.rows = []
    def on_js(self, m):
        d = dict(zip(m.name, m.position))
        if all(j in d for j in JN):
            self.qm = np.array([d[j] for j in JN])
            if self.rec: self.rows.append([self.simt, *self.qm, *self.tip()])
    def vtick(self):
        if self.ref is None or self.qm is None: return
        q, v = self.ref
        self.vpub.publish(Float64MultiArray(data=np.clip(v + KP*(q - self.qm), -2, 2).tolist()))
    def tool(self):
        try:
            t = self.tfb.lookup_transform('base_link', 'tool0', rclpy.time.Time()).transform.translation
            return np.array([t.x, t.y, t.z])
        except Exception: return np.array([np.nan]*3)
    def tip(self):
        p = self.tool(); return p - np.array([0, 0, 0.12])
    def move(self, q1, T):
        q0 = self.q_cmd; q1 = np.array(q1); t0 = self.simt
        while self.simt - t0 < T:
            self.ref = quintic(q0, q1, T, self.simt - t0); time.sleep(0.002)
        self.ref = (q1, np.zeros(6)); self.q_cmd = q1
    def hold(self, T):
        t0 = self.simt
        while self.simt - t0 < T: time.sleep(0.002)
    def set_fixture(self, x, y):
        rq = SetEntityState.Request(); s = rq.state; s.name = 'fixture'; s.reference_frame = 'world'
        s.pose.position.x, s.pose.position.y, s.pose.position.z = x, y, 0.42; s.pose.orientation.w = 1.0
        f = self.scli.call_async(rq)
        while not f.done(): time.sleep(0.002)

rclpy.init(); n = R(); ex = MultiThreadedExecutor(); ex.add_node(n)
threading.Thread(target=ex.spin, daemon=True).start()
n.scli.wait_for_service()
while n.simt == 0.0 or n.qm is None or np.isnan(n.tool()[0]): time.sleep(0.05)
home = np.array(W['home']); me = psutil.Process()

def procs(): return [me] + [p for p in psutil.process_iter(['name']) if p.info['name'] in ('gzserver', 'gzclient')]

def trial(k):
    off = np.random.default_rng(42 + k).uniform(-0.0015, 0.0015, 2); hole = np.array([0.5 + off[0], 0.2 + off[1]])
    n.q_cmd = n.qm.copy(); n.ref = (n.qm.copy(), np.zeros(6))
    n.move(home, 4.0); n.hold(1.0)
    n.set_fixture(*hole); n.hold(1.0)
    ps = procs(); c0 = sum(sum(p.cpu_times()[:2]) for p in ps)
    n.rows = []; n.rec = True; wall0 = time.time(); sim0 = n.simt
    n.move(W['lift'], 2.5); n.move(W['above_hole'], 2.5)
    if k == 1: print('   above_hole tool0:', n.tool().round(4), '(target 0.5, 0.2, 0.60)')
    desc = n.simt; n.move(W['insert'], 3); n.hold(1.0)
    sim_t = n.simt - sim0; wall_t = time.time() - wall0; n.rec = False
    cpu = 100 * (sum(sum(p.cpu_times()[:2]) for p in ps) - c0) / wall_t
    ram = sum(p.memory_info().rss for p in ps) / 2**20
    tip = n.tip(); xy = float(np.linalg.norm(tip[:2] - hole))
    if k == 1: print('   insert tool0:', n.tool().round(4), '(target 0.5, 0.2, 0.52)')
    n.move(W['above_hole'], 2.0); n.move(W['lift'], 2.5)   # retract (not measured)
    A = np.array(n.rows); tins = float('nan')
    if len(A):
        hit = np.where(A[:, 9] <= 0.405)[0]
        if len(hit): tins = A[hit[0], 0] - desc
    ok = bool(tip[2] <= 0.405 and xy < CLR)
    return A, dict(simulator='gazebo', task='peg_in_hole', trial=k, clearance_mm=CLR*1e3,
        offx_mm=round(off[0]*1e3, 2), offy_mm=round(off[1]*1e3, 2), success=int(ok),
        tip_z=round(float(tip[2]), 4), xy_err_mm=round(xy*1e3, 2), insert_time_s=round(tins, 2),
        peak_force_n=float('nan'), sim_time=round(sim_t, 2), wall_time=round(wall_t, 3),
        rtf=round(sim_t/wall_t, 2), mean_cpu=round(cpu, 1), mean_ram_mb=round(ram, 1))

res = []
for k in range(1, N + 1):
    A, r = trial(k); res.append(r); np.save(f'{OUT}/peg_in_hole_trial_{k:03d}.npy', A)
    print(k, 'success', r['success'], 'tip_z', r['tip_z'], 'xy_err_mm', r['xy_err_mm'], 'ins_t', r['insert_time_s'], 'rtf', r['rtf'], flush=True)
with open(f'{OUT}/peg_in_hole_summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
print('success rate: %d/%d' % (sum(r['success'] for r in res), N))
n.ref = None; rclpy.shutdown(); os._exit(0)
