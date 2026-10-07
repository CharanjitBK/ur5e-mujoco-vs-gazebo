import sys, rclpy, json, time, threading, numpy as np
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Float64MultiArray
from rclpy.qos import qos_profile_sensor_data
from gazebo_msgs.srv import GetEntityState, SetEntityState

W = json.load(open('/root/ur5_project/common/waypoints.json'))
JN = ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint']
GF = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
OPEN, CLOSED = -GF, GF
LIFT_T = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
VEL = len(sys.argv) > 3 and sys.argv[3] == 'vel'
KP = 10.0
print('vel mode:', VEL, ' grip effort:', GF, ' lift time:', LIFT_T)
HZ = 100

def quintic(a, b, T, t):
    s = np.clip(t / T, 0, 1); d = b - a
    p = 10*s**3 - 15*s**4 + 6*s**5
    v = (30*s**2 - 60*s**3 + 30*s**4) / T
    ac = (60*s - 180*s**2 + 120*s**3) / T**2
    return a + d*p, d*v, d*ac

class Runner(Node):
    def __init__(self):
        super().__init__('pnp_runner', parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.ac = ActionClient(self, FollowJointTrajectory, '/joint_trajectory_controller/follow_joint_trajectory')
        self.cli = self.create_client(GetEntityState, '/gazebo/get_entity_state')
        self.pub = self.create_publisher(Float64MultiArray, '/finger_effort_controller/commands', 10)
        self.grip = OPEN
        self.create_timer(0.02, lambda: self.pub.publish(Float64MultiArray(data=[self.grip, self.grip])))
        self.simt = 0.0; self.log = []
        self.create_subscription(Clock, '/clock', lambda m: setattr(self, 'simt', m.clock.sec + m.clock.nanosec*1e-9), qos_profile_sensor_data)
        self.create_subscription(JointState, '/joint_states', self.on_js, qos_profile_sensor_data)
        self.q_cmd = None
        self.qm = None; self.ref = None
        self.vpub = self.create_publisher(Float64MultiArray, '/forward_velocity_controller/commands', 10)
        self.create_timer(0.005, self.vtick)

    def on_js(self, m):
        d = dict(zip(m.name, m.position))
        if 'finger_r_joint' in d: self.fing = (round(d['finger_r_joint'],4), round(d['finger_l_joint'],4))
        if all(j in d for j in JN):
            self.log.append([self.simt] + [d[j] for j in JN])
            self.qm = np.array([d[j] for j in JN])

    def wait_sim(self, T):
        t0 = self.simt
        while self.simt - t0 < T: time.sleep(0.002)

    def vtick(self):
        if not VEL or self.ref is None or self.qm is None: return
        q, v = self.ref
        cmd = np.clip(v + KP * (q - self.qm), -2.0, 2.0)
        self.vpub.publish(Float64MultiArray(data=cmd.tolist()))

    def move_vel(self, q1, T):
        q0 = self.q_cmd; q1 = np.array(q1); t0 = self.simt
        while self.simt - t0 < T:
            q, v, _ = quintic(q0, q1, T, self.simt - t0)
            self.ref = (q, v); time.sleep(0.002)
        self.ref = (q1, np.zeros(6)); self.q_cmd = q1
        return 0

    def move(self, q1, T):
        if VEL: return self.move_vel(q1, T)
        q0 = self.q_cmd; q1 = np.array(q1)
        g = FollowJointTrajectory.Goal(); g.trajectory.joint_names = JN
        n = int(round(T*HZ))
        for i in range(1, n+1):
            t = i/HZ; q, v, a = quintic(q0, q1, T, t)
            p = JointTrajectoryPoint(positions=q.tolist(), velocities=v.tolist(), accelerations=a.tolist())
            p.time_from_start = Duration(sec=int(t), nanosec=int(round((t-int(t))*1e9)))
            g.trajectory.points.append(p)
        f = self.ac.send_goal_async(g)
        while not f.done(): time.sleep(0.002)
        gh = f.result(); assert gh.accepted, 'goal rejected'
        r = gh.get_result_async()
        while not r.done(): time.sleep(0.002)
        self.q_cmd = q1
        return r.result().result.error_code

    def hold(self, T, grip):
        self.grip = grip; self.wait_sim(T)

    def set_cube(self, x, y, z):
        if not hasattr(self, 'scli'):
            self.scli = self.create_client(SetEntityState, '/gazebo/set_entity_state')
            self.scli.wait_for_service()
        rq = SetEntityState.Request(); st = rq.state
        st.name = 'cube'; st.reference_frame = 'world'
        st.pose.position.x = x; st.pose.position.y = y; st.pose.position.z = z
        st.pose.orientation.w = 1.0
        f = self.scli.call_async(rq)
        while not f.done(): time.sleep(0.002)

    def ent(self, name):
        rq = GetEntityState.Request(); rq.name = name; rq.reference_frame = 'world'
        f = self.cli.call_async(rq)
        while not f.done(): time.sleep(0.002)
        r = f.result()
        if not r.success: return None
        q = r.state.pose.position
        return np.array([q.x, q.y, q.z]).round(4)

    def cube(self):
        rq = GetEntityState.Request(); rq.name = 'cube'; rq.reference_frame = 'world'
        f = self.cli.call_async(rq)
        while not f.done(): time.sleep(0.002)
        p = f.result().state.pose.position
        return np.array([p.x, p.y, p.z])

rclpy.init(); n = Runner()
ex = MultiThreadedExecutor(); ex.add_node(n)
threading.Thread(target=ex.spin, daemon=True).start()
n.ac.wait_for_server(); n.cli.wait_for_service()
while n.simt == 0.0: time.sleep(0.05)

# go to home (not part of the measured trial), then settle 1 s sim time
home = np.array(W['home'])
js_wait = time.time()
while not n.log: time.sleep(0.05)
n.q_cmd = np.array(n.log[-1][1:7])
print('cube before home move:', n.cube().round(4))
print('home move error_code:', n.move(home, 4.0)); n.hold(1.0, OPEN)
print('cube after home move:', n.cube().round(4))
n.set_cube(0.5, -0.2, 0.42); n.hold(1.0, OPEN)
print('cube start:', n.cube().round(4))

n.log.clear()
rows = []; done = [False]
def sampler():
    while not done[0]:
        try: rows.append((round(n.simt, 2), float(n.cube()[2]), getattr(n, 'fing', None)))
        except Exception: pass
        time.sleep(0.1)
threading.Thread(target=sampler, daemon=True).start()
wall = time.time(); sim0 = n.simt
n.move(W['above_cube'], 3); n.move(W['grasp'], 2)
def rep(tag):
    print(tag, 'cube', n.cube().round(4), 'fingers', getattr(n,'fing',None),
          '| finger_r', n.ent('ur::finger_r'), 'finger_l', n.ent('ur::finger_l'))
rep('1 at grasp pose  ')
n.hold(1.0, CLOSED)
rep('2 after closing  ')
print('palm z:', n.ent('ur::gripper_base'), ' finger z:', n.ent('ur::finger_r'), n.ent('ur::finger_l'))
n.move(W['above_cube'], LIFT_T)
rep('3 after lift     ')
print('cube z after lift:', n.cube()[2], '(lifted if > 0.5)')
n.move(W['above_target'], 3); n.move(W['place'], 2)
n.hold(0.5, CLOSED); n.hold(1.0, OPEN)
n.move(W['above_target'], 2); n.hold(1.0, OPEN)
p = n.cube(); err = np.linalg.norm(p[:2] - [0.5, 0.2])
ok = err < 0.01 and abs(p[2] - 0.42) < 0.01
simT, wallT = n.simt - sim0, time.time() - wall
print(f'cube final: {p.round(4)}  xy_err={err*1000:.1f} mm  SUCCESS={ok}')
print(f'sim time {simT:.1f} s  wall {wallT:.2f} s  RTF {simT/wallT:.2f}')
done[0] = True
for r in rows[::5]: print('t=%.2f cube_z=%.4f fingers=%s' % r)
np.save('/root/ur5_project/gazebo_ur5/pnp_trial_test.npy', np.array(n.log))
n.grip = OPEN; time.sleep(0.2)
rclpy.shutdown()
