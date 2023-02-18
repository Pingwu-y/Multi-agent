import os
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.actions import GroupAction
from launch_ros.actions import PushRosNamespace
from launch.actions import DeclareLaunchArgument
from tribot import *

def generate_launch_description():
    robot_name = LaunchConfiguration('robot_name')
    robot_name_launch_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='tribot1',
        description='configure the namespace'
    )

    package_name='tribot' #<--- CHANGE ME
    world_file_path = 'worlds/neighborhood.world'
    pkg_path = os.path.join(get_package_share_directory(package_name))
    world_path = os.path.join(pkg_path, world_file_path)  
    
    # Pose where we want to spawn the robot
    spawn_x_val = '0.3'
    spawn_y_val = '0.0'
    spawn_z_val = '0.0'
    spawn_yaw_val = '0.0'
    
    # Create a robot_state_publisher node
    tribot_node = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory(package_name),'launch','tribot_follower.launch.py'
                )]), launch_arguments={'use_sim_time': 'true', 'world':world_path}.items()
    )

    kinematic_node = IncludeLaunchDescription(
      PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('tribot'), 'launch'),
         '/set_kinematic.launch.py'])
      )

    # Include the Gazebo launch file, provided by the gazebo_ros package
    # Run the spawner node from the gazebo_ros package. The entity name doesn't really matter if you only have a single robot.
    spawn_entity = Node(package='gazebo_ros', executable='spawn_entity.py',
                        arguments=['-topic', 'robot_description',
                                   '-entity', robot_name,
                                   '-x', spawn_x_val,
                                   '-y', spawn_y_val,
                                   '-z', spawn_z_val,
                                   '-Y', spawn_yaw_val],
                        output='screen')

    node_with_namespace = GroupAction(
     actions=[
        PushRosNamespace(robot_name),
        tribot_node,
        spawn_entity,
        robot_name_launch_arg,
        #kinematic_node
      ]
   )

    # Launch them all!
    return LaunchDescription([
        node_with_namespace
    ])