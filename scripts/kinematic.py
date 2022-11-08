#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist, TransformStamped
# from nav_msgs.msg import Odometry
from math import sin,cos,atan2
import tf

init_x = 0
init_y = 0
init_z = 0  # 以base_footprint为整个零件的连体基
init_theta = 0

class Set_Model_State(object):
    global init_x, init_y, init_z, init_theta

    # class to package the pose of obj, (x, y, z, theta)
    class State(object):
        def __init__(self, x, y, z, theta):
            self.x = x
            self.y = y
            self.z = z
            self.theta = theta

    ##################################### initializing ####################################
    def __init__(self):
        # newly added
        self.last_t=0
        self.dt=0
        self.current_time = rospy.get_time()
        self.state = self.State(init_x, init_y, init_z, init_theta) # 初始xyz
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.control_rate=10        # need to be adjusted further according to dt
        # publish the new position of levitator to gazebo
        self.state_pub = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=10)
        # subscribe control input
        rospy.Subscriber("/cmd_vel", Twist, self.MotionCallback)
        rospy.Subscriber("gazebo/set_model_state", ModelState, self.broadcast_odom_tf)
    
    def broadcast_odom_tf(self,model_state):
        parent_frame = 'odom'
        child_frame = 'base_footprint'                 
        time_now = rospy.Time.now()
        translation = (model_state.pose.position.x, model_state.pose.position.y, model_state.pose.position.z)
        rotation = (model_state.pose.orientation.x,model_state.pose.orientation.y,model_state.pose.orientation.z,model_state.pose.orientation.w)
        # Send the transformation
        self.tf_broadcaster.sendTransform(translation, rotation, time_now,child_frame,parent_frame)

    # main function
    # receive the control signal and generate the modelstate, modify the modelstate
    # @return   NULL
    def MotionCallback(self,cmd_vel):
        state_t = ModelState()
        state_t.model_name = "tribot"
        self.current_time = rospy.get_time()
        self.dt = self.current_time - self.last_t
        
        if (self.last_t==0):
            self.last_t = self.current_time
            initial_state = self.state
            initial_state_pub = self.Generate_State(initial_state,"tribot")
            # rospy.loginfo(initial_state_pub)
            self.state_pub.publish(initial_state_pub)
        else :
            self.last_t = self.current_time
            self.state = self.forward_dynamics(self.state, cmd_vel)
            state_t_pub=self.Generate_State(self.state,"tribot")
            # rospy.loginfo(state_t_pub)
            self.state_pub.publish(state_t_pub)

    # generate ModelState msg by given pose, which fits gazebo env
    # @param    pose: given pose
    #           name: model_name
    # @return   ModelState()
    def Generate_State(self, state, name):
        state_p = ModelState()
        state_p.model_name = name

        qtn = tf.transformations.quaternion_from_euler(0, 0, state.theta)
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
        
if __name__ == '__main__':
    try:
        rospy.init_node('set_kinematic', anonymous=True)
        state_set = Set_Model_State()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
