需要四个终端
终端1：roslaunch tribot_description view_tribot_gazebo_world.launch
将小车生成在一个空的世界中。

终端2：roslaunch tribot_description  set_kinematic.launch
小车的运动学模型节点，根据控制输入和运动学模型，输出在gazebo中的位置

终端3：roslaunch tribot_description tribot_teleop.launch
键盘控制运动节点

终端4：rviz或者其他可视化节点
