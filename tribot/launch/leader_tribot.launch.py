import os
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import GroupAction
from launch_ros.actions import PushRosNamespace
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
import xacro

def generate_launch_description():
   # Include the robot_state_publisher launch file, provided by our own package. Force sim time to be enabled
   # !!! MAKE SURE YOU SET THE PACKAGE NAME CORRECTLY !!!
    package_name='tribot' #<--- CHANGE ME
    world_file_path = 'worlds/empty_world.model'  
    pkg_path = os.path.join(get_package_share_directory(package_name))
    world_path = os.path.join(pkg_path, world_file_path)   
    # Pose where we want to spawn the robot
    spawn_x_val = '0.0'
    spawn_y_val = '0.0'
    spawn_z_val = '0.0'
    spawn_yaw_val = '0.0'

    tribot = IncludeLaunchDescription(
               PythonLaunchDescriptionSource([os.path.join(
                  get_package_share_directory(package_name),'launch','tribot.launch.py'
               )]), launch_arguments={'use_sim_time': 'true', 'world':world_path,'robot_name':'leader_tribot'}.items()
   )
   # Include the Gazebo launch file, provided by the gazebo_ros package
    gazebo = IncludeLaunchDescription(
               PythonLaunchDescriptionSource([os.path.join(
                  get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')]),
            )
    # Run the spawner node from the gazebo_ros package. The entity name doesn't really matter if you only have a single robot.
    spawn_entity = Node(package='gazebo_ros', executable='spawn_entity.py',
                     arguments=['-topic', '/robot_description',
                                 '-entity','leader_tribot',                     
                                 '-x', spawn_x_val,
                                 '-y', spawn_y_val,
                                 '-z', spawn_z_val,
                                 '-Y', spawn_yaw_val],
                     output='screen')

    kinematic_node = Node(package='tribot', executable='kinematic',
                        output='screen')
   
 
    return LaunchDescription([
         tribot,
         #gazebo,
         spawn_entity,
         kinematic_node 
        
   ])