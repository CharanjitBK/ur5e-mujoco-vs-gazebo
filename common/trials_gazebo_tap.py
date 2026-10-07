import sys, json, time, csv, os, subprocess, threading, numpy as np, psutil, rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Float64MultiArray
from tf2_ros import Buffer, TransformListener

W = json.load(open('/root/ur5_project/common/waypoints_tap.json'))
JW = json.load(open('/root/ur5_project/common/waypoints_tap_jitter.json'))
JN = ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint']
args = sys.argv[1:]; FORCE = 'force' in args
nums = [a for a in args if a.isdigit()]; N = int(nums[0]) if nums else 20
CYC = 20; OPEN = -8.0; KP = 10.0
OUT = '/root/ur5_project/data/raw/gazebo'; os.makedirs(OUT, exist_ok=True)

def quintic(a, b, T, t):
    s = np.clip(t / T, 0, 1); d = b - a
    return a + d*(10*s**3 - 15*s**4 + 6*s**5), d*(30*s**2 - 60*s**3 + 30*s**4) / T

class R(Node):
    def __init__(self):
        super().__init__('tap_trials', parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.fpub = self.create_publisher(Float64MultiArray, '/finger_effort_controller/commands', 10)
        self.vpub = self.create_publisher(Float64MultiArray, '/forward_velocity_controller/commands', 10)
        self.ref = None; self.qm = None; self.simt = 0.0; self.rec = False; self.rows = []
        self.tfb = Buffer(); self.tfl = TransformListener(self.tfb, self)
        self.create_timer(0.02, lambda: self.fpub.publish(Float64MultiArray(data=[OPEN, OPEN])))
        self.create_timer(0.005, self.vtick)
        self.create_subscription(Clock, '/clock', lambda m: setattr(self, 'simt', m.clock.sec + m.clock.nanosec*1e-9), qos_profile_sensor_data)
        self.create_subscription(JointState, '/joint_states', self.on_js, qos_profile_sensor_data)
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
    def tip(self): return self.tool() - np.array([0, 0, 0.12])
    def move(self, q1, T):
        q0 = self.q_cmd; q1 = np.array(q1); t0 = self.simt
        while self.simt - t0 < T:
            self.ref = quintic(q0, q1, T, self.simt - t0); time.sleep(0.002)
        self.ref = (q1, np.zeros(6)); self.q_cmd = q1
    def hold(self, T):
        t0 = self.simt
        while self.simt - t0 < T: time.sleep(0.002)

rclpy.init(); n = R(); ex = MultiThreadedExecutor(); ex.add_node(n)
threading.Thread(target=ex.spin, daemon=True).start()
while n.simt == 0.0 or n.qm is None or np.isnan(n.tool()[0]): time.sleep(0.05)
home = np.array(W['home']); me = psutil.Process()
topic = None
if FORCE:
    lst = subprocess.run(['gz', 'topic', '-l'], capture_output=True, text=True).stdout.split()
    topic = [t for t in lst if t.endswith('/physics/contacts')][0]; print('contacts topic:', topic)

def procs(): return [me] + [p for p in psutil.process_iter(['name']) if p.info['name'] in ('gzserver', 'gzclient')]

def trial(k):
    Wt = dict(W); Wt['tap_down'] = JW[str(k)]['q']; jit = JW[str(k)]['jit_mm']
    n.q_cmd = n.qm.copy(); n.ref = (n.qm.copy(), np.zeros(6))
    n.move(home, 4.0); n.hold(1.0); n.move(Wt['lift'], 2.5); n.move(Wt['tap_up'], 2.5)
    rp = None
    if FORCE:
        f = open(f'{OUT}/tap_contacts_trial_{k:03d}.txt', 'w')
        rp = subprocess.Popen(['gz', 'topic', '-e', topic], stdout=f); time.sleep(1.0)
    ps = procs(); c0 = sum(sum(p.cpu_times()[:2]) for p in ps)
    n.rows = []; n.rec = True; t_start = n.simt; wall0 = time.time()
    for c in range(CYC):
        n.move(Wt['tap_down'], 0.5); n.move(Wt['tap_up'], 0.5)
    sim_t = n.simt - t_start; wall_t = time.time() - wall0; n.rec = False
    cpu = 100 * (sum(sum(p.cpu_times()[:2]) for p in ps) - c0) / wall_t
    ram = sum(p.memory_info().rss for p in ps) / 2**20
    if rp: rp.terminate(); rp.wait(); f.close()
    n.move(Wt['lift'], 2.0)
    A = np.array(n.rows); mins = []
    for c in range(CYC):
        s = A[(A[:, 0] >= t_start + c) & (A[:, 0] < t_start + c + 1)]
        mins.append(s[:, 9].min() if len(s) else np.nan)
    mins = np.array(mins); reached = int((mins < 0.425).sum())
    r = dict(simulator='gazebo', task='tapping', trial=k, pad_jitter_mm=jit, success=int(reached == CYC),
             cycles_valid=reached, tip_min_z_mean=round(float(np.nanmean(mins)), 4), t_start=round(t_start, 3),
             sim_time=round(sim_t, 2), wall_time=round(wall_t, 3), rtf=round(sim_t/wall_t, 2),
             mean_cpu=round(cpu, 1), mean_ram_mb=round(ram, 1))
    print(k, 'success', r['success'], 'jit_mm', jit, 'tip_min_z', r['tip_min_z_mean'], 'rtf', r['rtf'], flush=True)
    return A, r

res = []
for k in range(1, N + 1):
    A, r = trial(k); res.append(r); np.save(f'{OUT}/tapping_trial_{k:03d}.npy', A)
with open(f'{OUT}/tapping_summary.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
print('success rate: %d/%d' % (sum(r['success'] for r in res), N))
n.ref = None; rclpy.shutdown(); os._exit(0)
