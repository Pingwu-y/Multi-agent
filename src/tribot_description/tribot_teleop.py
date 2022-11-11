'''
FileName: 
Description: 
Autor: Liujunjie/Aries-441
StudentNumber: 521021911059
Date: 2022-11-10 19:20:24
E-mail: sjtu.liu.jj@gmail.com/sjtu.1518228705@sjtu.edu.cn
LastEditTime: 2022-11-11 00:33:20
'''

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos_event import SubscriptionEventCallbacks
import sys, select, termios, tty

msg = """
Control tribot!
---------------------------
Moving around:
   u    i    o
   j    k    l
   m    ,    .

q/z : increase/decrease max speeds by 10%
w/x : increase/decrease only linear speed by 10%
e/c : increase/decrease only angular speed by 10%
space key, k : force stop
anything else : stop smoothly

CTRL-C to quit
"""

moveBindings = {
        'i':(1,0),
        'o':(1,-1),
        'j':(0,1),
        'l':(0,-1),
        'u':(1,1),
        ',':(-1,0),
        '.':(-1,1),
        'm':(-1,-1),
           }

speedBindings={
        'q':(1.1,1.1),
        'z':(.9,.9),
        'w':(1.1,1),
        'x':(.9,1),
        'e':(1,1.1),
        'c':(1,.9),
          }
'''
def getKey():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, termios.tcgetattr(sys.stdin))
    return key

speed = .2
turn = 1

def vels(speed,turn):
    return "currently:\tspeed %s\tturn %s " % (speed,turn)

#settings = termios.tcgetattr(sys.stdin)
'''
class Keyboard(Node):
    def _init_(self, name):
        super()._init_(name)
        pub = self.create_publisher(Twist,'cmd_vel',10)
        self.timer=self.create_timer(0.1, self.timer_callback)
    def timer_callback(self):
        x = 0
        th = 0
        status = 0
        count = 0
        acc = 0.1
        target_speed = 0
        target_turn = 0
        control_speed = 0
        control_turn = 0
        try:
            print(msg)
            print(self.vels(speed,turn))
            while(1):
                key = self.getKey()
                # 运动控制方向键（1：正方向，-1负方向）
                if key in moveBindings.keys():
                    x = moveBindings[key][0]
                    th = moveBindings[key][1]
                    count = 0
                # 速度修改键
                elif key in speedBindings.keys():
                    speed = speed * speedBindings[key][0]  # 线速度增加0.1倍
                    turn = turn * speedBindings[key][1]    # 角速度增加0.1倍
                    count = 0

                    print(self.vels(speed,turn))
                    if (status == 14):
                        print(msg)
                    status = (status + 1) % 15
                # 停止键
                elif key == ' ' or key == 'k' :
                    x = 0
                    th = 0
                    control_speed = 0
                    control_turn = 0
                else:
                    count = count + 1
                    if count > 4:
                        x = 0
                        th = 0
                    if (key == '\x03'):
                        break

                # 目标速度=速度值*方向值
                target_speed = speed * x
                target_turn = turn * th

                # 速度限位，防止速度增减过快
                if target_speed > control_speed:
                    control_speed = min( target_speed, control_speed + 0.02 )
                elif target_speed < control_speed:
                    control_speed = max( target_speed, control_speed - 0.02 )
                else:
                    control_speed = target_speed

                if target_turn > control_turn:
                    control_turn = min( target_turn, control_turn + 0.1 )
                elif target_turn < control_turn:
                    control_turn = max( target_turn, control_turn - 0.1 )
                else:
                    control_turn = target_turn

                # 创建并发布twist消息
                twist = Twist()
                twist.linear.x = control_speed; 
                twist.linear.y = 0; 
                twist.linear.z = 0
                twist.angular.x = 0; 
                twist.angular.y = 0; 
                twist.angular.z = control_turn
                self.pub.publish(twist)

        except:
            print(1,1.1)

        finally:
            twist = Twist()
            twist.linear.x = 0.0; twist.linear.y = 0.0; twist.linear.z = 0.0
            twist.angular.x = 0.0; twist.angular.y = 0.0; twist.angular.z = 0.0
            self.pub.publish(twist)

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, termios.tcgetattr(sys.stdin))

    def getKey():
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, termios.tcgetattr(sys.stdin))
        return key

    speed = .2
    turn = 1

    def vels(speed,turn):
        return "currently:\tspeed %s\tturn %s " % (speed,turn)

def main(args=None):                                 # ROS2节点主入口main函数
    rclpy.init(args=args)                            # ROS2 Python接口初始化
    node = Keyboard("tribot_description")        # 创建ROS2节点对象并进行初始化
    rclpy.spin(node)                                 # 循环等待ROS2退出
    node.destroy_node()                              # 销毁节点对象
    rclpy.shutdown()                                 # 关闭ROS2 Python接口