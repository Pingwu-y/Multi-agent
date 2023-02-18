import rclpy
import os

#from ament_index_python.packages import get_package_share_directory
import launch
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import xacro


def generate_launch_description():
    scale_linear_launch_arg = DeclareLaunchArgument(
      'scale_linear', default_value=TextSubstitution(text='0.1')     # 创建一个Launch文件内参数（arg）background_r
   )
    scale_angular_launch_arg = DeclareLaunchArgument(
      'scale_angular', default_value=TextSubstitution(text='0.4')     # 创建一个Launch文件内参数（arg）background_r
   )
    
    # Launch!
    return LaunchDescription([
      scale_linear_launch_arg,
      scale_angular_launch_arg,
      Node(
         package='tribot',
         executable='tribot_teleop',
         output='screen',
         parameters=[{
          'scale_linear': LaunchConfiguration('scale_linear'),   # 创建参数background_r
          'scale_angular': LaunchConfiguration('scale_angular')
            }]
      ),
   ])