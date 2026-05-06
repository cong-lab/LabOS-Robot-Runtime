from launch import LaunchDescription
from ament_index_python.packages import get_package_share_path
from launch_ros.actions import Node

def generate_launch_description():
    # 定义包名和文件路径
    package_name = 'zwhand_17dof_right'  # 替换为实际URDF包名
    urdf_file = 'zwhand_17dof_right.urdf'  # URDF文件名
    rviz_config_file = 'display.rviz'    # RViz配置文件（可选）

    # 获取URDF和RViz配置文件的绝对路径
    urdf_path = get_package_share_path(package_name) / 'urdf' / urdf_file
    rviz_config_path = get_package_share_path(package_name) / 'rviz' / rviz_config_file

    # 读取URDF内容（XML格式）
    with open(urdf_path, 'r') as f:
        urdf_content = f.read()

    # 节点列表
    nodes = [
        # 1. 机器人状态发布节点（发布URDF到参数服务器）
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': urdf_content}],
            output='screen'
        ),

        # 2. 关节状态发布节点（带GUI界面，可手动调节关节角度）
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen'
        ),

        
        # 3. RViz2可视化节点（可选，需提前创建RViz配置文件）
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', str(rviz_config_path)],
            output='screen'
        )
    ]

    return LaunchDescription(nodes)