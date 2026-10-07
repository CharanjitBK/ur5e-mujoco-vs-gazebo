import rclpy, time
from rclpy.node import Node
from gazebo_msgs.srv import GetEntityState
rclpy.init(); n = Node('fv'); c = n.create_client(GetEntityState, '/gazebo/get_entity_state'); c.wait_for_service()
t_end = time.time() + 40
while time.time() < t_end:
    rq = GetEntityState.Request(); rq.name = 'ur::finger_r'; rq.reference_frame = 'world'
    f = c.call_async(rq); rclpy.spin_until_future_complete(n, f)
    s = f.result().state
    print('z=%.4f  vz=%.4f' % (s.pose.position.z, s.twist.linear.z))
    time.sleep(0.5)
