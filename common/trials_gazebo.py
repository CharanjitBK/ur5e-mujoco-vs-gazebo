import sys, json, time, csv, os, threading, numpy as np, psutil, rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Float64MultiArray
from gazebo_msgs.srv import GetEntityState, SetEntityState

W = json.load(open('/root/ur5_project/common/waypoints.json'))
JN = ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint']
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
OUT = '/root/ur5_project/data/raw/gazebo'; os.makedirs(OUT, exist_ok=True)
GF = 8.0; OPEN, CLOSED = -GF, GF; KP = 10.0

def quintic(a, b, T, t):
    s = np.clip(t / T, 0, 1); d = b - a
    return a + d*(10*s**3 - 15*s**4 + 6*s**5), d*(30*s**2 - 60*s**3 + 30*s**4) / T

class R(Node):
    def __init__(self):
        super().__init__('trials', parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.gcli = self.create_client(GetEntityState, '/gazebo/get_entity_state')
        self.scli = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.fpub = self.create_publisher(Float64MultiArray, '/finger_effort_controller/commands', 10)
        self.vpub = self.create_publisher(Float64MultiArray, '/forward_velocity_controller/commands', 10)
        self.grip = OPEN; self.ref = None; self.qm = None; self.simt = 0.0; self.rec = False; self.rows = []
        self.create_timer(0.02, lambda: self.fpub.publish(Float64MultiArray(data=[self.grip, self.grip])))
        self.create_timer(0.005, self.vtick)
        self.create_subscription(Clock, '/clock', lambda m: setattr(self, 'simt', m.clock.sec + m.clock.nanosec*1e-9), qos_profile_sensor_data)
        self.create_subscription(JointState, '/joint_states', self.on_js, qos_profile_sensor_data)
    def on_js(self, m):
        d = dict(zip(m.name, m.position))
        if all(j in d for j in JN):
            self.qm = np.array([d[j] for j in JN])
            if 'finger_r_joint' in d: self.fing = (round(d['finger_r_joint'],4), round(d['finger_l_joint'],4))
            if self.rec:
                qd = self.ref[0] if self.ref else self.qm
                self.rows.append([self.simt, *self.qm, *qd])
    def vtick(self):
        if self.ref is None or self.qm is None: return
        q, v = self.ref
        self.vpub.publish(Float64MultiArray(data=np.clip(v + KP*(q - self.qm), -2, 2).tolist()))
    def wait(self, T):
        t0 = self.simt
        while self.simt - t0 < T: time.sleep(0.002)
    def hold(self, T, g): self.grip = g; self.wait(T)
    def move(self, q1, T):
        q0 = self.q_cmd; q1 = np.array(q1); t0 = self.simt
        while self.simt - t0 < T:
            self.ref = quintic(q0, q1, T, self.simt - t0); time.sleep(0.002)
        self.ref = (q1, np.zeros(6)); self.q_cmd = q1
    def rep(self, tag):
        print('   ', tag, 'cube', self.cube().round(4), 'fingers', getattr(self, 'fing', None), flush=True)

    def cube(self):
        rq = GetEntityState.Request(); rq.name = 'cube'; rq.reference_frame = 'world'
        f = self.gcli.call_async(rq)
        while not f.done(): time.sleep(0.002)
        p = f.result().state.pose.position; return np.array([p.x, p.y, p.z])
    def set_cube(self, x, y, z):
        rq = SetEntityState.Request(); s = rq.state; s.name = 'cube'; s.reference_frame = 'world'
        s.pose.position.x, s.pose.position.y, s.pose.position.z = x, y, z; s.pose.orientation.w = 1.0
        f = self.scli.call_async(rq)
        while not f.done(): time.sleep(0.002)

rclpy.init(); n = R(); ex = MultiThreadedExecutor(); ex.add_node(n)
threading.Thread(target=ex.spin, daemon=True).start()
n.gcli.wait_for_service(); n.scli.wait_for_service()
while n.simt == 0.0 or n.qm is None: time.sleep(0.05)
home = np.array(W['home']); me = psutil.Process()

def procs(): return [me] + [p for p in psutil.process_iter(['name']) if p.info['name'] in ('gzserver', 'gzclient')]

def trial(k):
    off = np.random.default_rng(42 + k).uniform(-0.005, 0.005, 2)
    n.q_cmd = n.qm.copy(); n.ref = (n.qm.copy(), np.zeros(6))
    n.grip = OPEN; n.move(home, 4.0); n.hold(1.0, OPEN)
    n.set_cube(0.5 + off[0], -0.2 + off[1], 0.42); n.hold(1.0, OPEN)
    ps = procs(); [p.cpu_percent(None) for p in ps]
    cpu = []; ram = []; stop = [False]
    def sampler():
        while not stop[0]:
            time.sleep(0.2)
            try: cpu.append(sum(p.cpu_percent(None) for p in ps)); ram.append(sum(p.memory_info().rss for p in ps)/2**20)
            except psutil.Error: pass
    th = threading.Thread(target=sampler, daemon=True)
    if 'nosamp' not in sys.argv: th.start()
    n.rows = []; n.rec = True; wall0 = time.time(); sim0 = n.simt
    n.move(W['above_cube'], 3); n.move(W['grasp'], 2); n.rep('grasp pose ')
    n.hold(1.0, CLOSED); n.rep('closed     ')
    n.move(W['above_cube'], 2); lift_z = n.cube()[2]; n.rep('lifted     ')
    n.move(W['above_target'], 3); n.rep('transferred')
    n.move(W['place'], 2)
    n.hold(0.5, CLOSED); n.hold(1.0, OPEN); n.move(W['above_target'], 2); n.hold(1.0, OPEN)
    sim_t = n.simt - sim0; wall_t = time.time() - wall0; n.rec = False; stop[0] = True
    if th.is_alive(): th.join()
    p = n.cube(); err = float(np.linalg.norm(p[:2] - [0.5, 0.2]))
    return np.array(n.rows), dict(simulator='gazebo', task='pick_place', trial=k,
        offx_mm=round(off[0]*1e3, 2), offy_mm=round(off[1]*1e3, 2),
        success=int(err < 0.01 and abs(p[2] - 0.42) < 0.01 and lift_z > 0.5),
        place_err_mm=round(err*1e3, 2), lift_z=round(float(lift_z), 4), sim_time=round(sim_t, 2),
        wall_time=round(wall_t, 3), rtf=round(sim_t/wall_t, 2),
        mean_cpu=round(float(np.mean(cpu)) if cpu else 0, 1), max_cpu=round(float(np.max(cpu)) if cpu else 0, 1),
        mean_ram_mb=round(float(np.mean(ram)) if ram else 0, 1), max_ram_mb=round(float(np.max(ram)) if ram else 0, 1))

res = []
for k in range(1, N + 1):
    A, r = trial(k); res.append(r); np.save(f'{OUT}/pick_place_trial_{k:03d}.npy', A)
    print(k, 'success', r['success'], 'err_mm', r['place_err_mm'], 'rtf', r['rtf'], 'cpu', r['mean_cpu'], flush=True)
with open(f'{OUT}/pick_place_summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
print('success rate: %d/%d' % (sum(r['success'] for r in res), N))
n.ref = None; rclpy.shutdown(); os._exit(0)
