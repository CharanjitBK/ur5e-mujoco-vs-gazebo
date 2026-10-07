import rclpy, time, threading
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Float64MultiArray

class T(Node):
    def __init__(self):
        super().__init__('finger_test', parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.pub = self.create_publisher(Float64MultiArray, '/finger_effort_controller/commands', 10)
        self.g = -8.0; self.n = 0
        self.create_timer(0.02, self.tick)
        self.t = 0.0; self.f = None
        self.create_subscription(Clock, '/clock', lambda m: setattr(self, 't', m.clock.sec + m.clock.nanosec*1e-9), qos_profile_sensor_data)
        self.create_subscription(JointState, '/joint_states', self.js, qos_profile_sensor_data)
    def tick(self):
        self.n += 1; self.pub.publish(Float64MultiArray(data=[self.g, self.g]))
    def js(self, m):
        d = dict(zip(m.name, m.position)); self.f = (round(d['finger_r_joint'],4), round(d['finger_l_joint'],4))

rclpy.init(); n = T(); threading.Thread(target=rclpy.spin, args=(n,), daemon=True).start()
while n.t == 0.0 or n.f is None: time.sleep(0.05)
time.sleep(1.0); print('open, fingers:', n.f, ' timer ticks so far:', n.n)
n.g = 8.0; t0 = n.t; k0 = n.n
for i in range(8):
    time.sleep(0.5); print(f'close  sim t+{n.t-t0:4.1f}s  fingers {n.f}  ticks {n.n-k0}')
n.g = -8.0
time.sleep(2); print('reopen, fingers:', n.f)
rclpy.shutdown()
