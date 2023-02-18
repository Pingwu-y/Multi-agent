import rclpy
from rclpy.node import Node
from rclpy import time
import numpy as np
from gazebo_msgs.msg import ModelState
import tf_transformations
from geometry_msgs.msg import Twist, TransformStamped
from tf2_ros import TransformBroadcaster 
# from nav_msgs.msg import Odometry
from math import sin,cos,atan2


init_x = 0.0
init_y = 0.0
init_z = 0.0  # 以base_footprint为整个零件的连体基
init_theta = 0.0

class Set_Model_State(Node):
    global init_x, init_y, init_z, init_theta

    # class to package the pose of obj, (x, y, z, theta)
    class State(object):
        def __init__(self, x, y, z, theta):
            self.x = x
            self.y = y
            self.z = z
            self.theta = theta

    ##################################### initializing ####################################
    def __init__(self,name):
        # newly added
        super().__init__(name)  
        self.last_t=0
        self.dt=0
        self.current_time = self.get_clock().now()
        self.state = self.State(init_x, init_y, init_z, init_theta) # 初始xyz
        self.tf_broadcaster = TransformBroadcaster(self)
        self.control_rate=10        # need to be adjusted further according to dt
        # publish the new position of levitator to gazebo
        self.state_pub = self.create_publisher(ModelState,'/leader_tribot/gazebo/set_model_state',10)
        # subscribe control input
        self.rate_sub = self.create_subscription (Twist,'/leader_tribot/cmd_vel',self.MotionCallback,10)
        self.state_sub = self.create_subscription(ModelState,"/leader_tribot/gazebo/set_model_state", self.broadcast_odom_tf,10)

    def broadcast_odom_tf(self,model_state):
        transform = TransformStamped() 
        transform.header.stamp = self.get_clock().now().to_msg()      # 设置坐标变换消息的时间戳
        transform.header.frame_id = 'world'                           # 设置一个坐标变换的源坐标系
        transform.child_frame_id = 'leader_tribot/base_footprint'                   # 设置一个坐标变换的目标坐标系
        transform.transform.translation.x = model_state.pose.position.x                     # 设置坐标变换中的X、Y、Z向的平移
        transform.transform.translation.y = model_state.pose.position.y
        transform.transform.translation.z = model_state.pose.position.z
        #q = tf_transformations.quaternion_from_euler(0, 0, model_state.theta) # 将欧拉角转换为四元数（roll, pitch, yaw）
        transform.transform.rotation.x = model_state.pose.orientation.x                        # 设置坐标变换中的X、Y、Z向的旋转（四元数）
        transform.transform.rotation.y = model_state.pose.orientation.y
        transform.transform.rotation.z = model_state.pose.orientation.z
        transform.transform.rotation.w = model_state.pose.orientation.w
        # Send the transformation
        self.tf_broadcaster.sendTransform(transform)     # 广播坐标变换，海龟位置变化后，将及时更新坐标变换信息
        
    # main function
    # receive the control signal and generate the modelstate, modify the modelstate
    # @return   NULL
    def MotionCallback(self,cmd_vel):
        state_t = ModelState()
        state_t.model_name = "leader_tribot"
        self.current_time = self.get_clock().now().seconds_nanoseconds()[0]
        self.dt = self.current_time - self.last_t
        
        if (self.last_t==0):
            self.last_t = self.current_time
            initial_state = self.state
            initial_state_pub = self.Generate_State(initial_state,"leader_tribot")
            # rospy.loginfo(initial_state_pub)
            self.state_pub.publish(initial_state_pub)
        else :
            self.last_t = self.current_time
            self.state = self.forward_dynamics(self.state, cmd_vel)
            state_t_pub=self.Generate_State(self.state,"leader_tribot")
            # rospy.loginfo(state_t_pub)
            self.state_pub.publish(state_t_pub)

    # generate ModelState msg by given pose, which fits gazebo env
    # @param    pose: given pose
    #           name: model_name
    # @return   ModelState()
    def Generate_State(self, state, name):
        state_p = ModelState()
        state_p.model_name = name

        qtn = tf_transformations.quaternion_from_euler(0, 0, state.theta)
        state_p.pose.orientation.x = qtn[0]
        state_p.pose.orientation.y = qtn[1]
        state_p.pose.orientation.z = qtn[2]
        state_p.pose.orientation.w = qtn[3]

        state_p.pose.position.x = state.x
        state_p.pose.position.y = state.y
        state_p.pose.position.z = state.z
        return state_p
        
    # update state by given control signal
    # @param    state_now: now state， control input
    # @return   state_next
    def forward_dynamics(self, state_now, cmd_vel):
        v = cmd_vel.linear.x
        omega = cmd_vel.angular.z
        x_next = state_now.x + v*cos(state_now.theta)*self.dt
        y_next = state_now.y + v*sin(state_now.theta)*self.dt
        z_next = state_now.z
        theta_next = state_now.theta + omega*self.dt
        # limit theta into the range of (-pi,pi)
        theta_next = atan2(sin(theta_next), cos(theta_next))
        state_next = self.State(x_next, y_next, z_next, theta_next)
        return state_next
'''   
if __name__ == '__main__':
    try:
        rclpy.init(args='set_kinematic')
        state_set = Set_Model_State()
        rclpy.spin()
    except rclpy.ROSInterruptException:
        pass
'''
def main(args=None):                                 # ROS2节点主入口main函数
    rclpy.init(args='set_kinematic')                            # ROS2 Python接口初始化
    node = Set_Model_State("tribot")        # 创建ROS2节点对象并进行初始化
    rclpy.spin(node)                                 # 循环等待ROS2退出
    node.destroy_node()                              # 销毁节点对象
    rclpy.shutdown()                                 # 关闭ROS2 Python接口
