"""
learning-nav-irl: Learning human-aware robot navigation via IRL.
Copyright (C) 2020  Marina Kollmitz

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import copy
import numpy as np

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, OccupancyGrid


class NavMap:
    """
    Maintain environment map and provide some helpful conversions.
    """

    def __init__(self, ros_inflated_map):
        """
        Initialize.
        :param ros_inflated_map: inflated occupancy map from ros
        """

        map_shape = (ros_inflated_map.info.height, ros_inflated_map.info.width)
        self.grid = np.array(ros_inflated_map.data).reshape(map_shape).T

        self.map_info = ros_inflated_map.info
        self.frame_id = ros_inflated_map.header.frame_id
        self.ros_occupancy_map = ros_inflated_map

    def publish(self):
        """
        Publish environment map as ros message.
        """

        map_pub = rospy.Publisher("/inflated_map", OccupancyGrid, latch=True, queue_size=1)
        map_pub.publish(self.ros_occupancy_map)

    def visualize_grid(self, viz_array, publisher):
        """
        Publish array as ros OccupancyGrid for visualization.
        :param viz_array: array for visualization
        :param publisher: ros publisher
        """

        viz_map = OccupancyGrid()
        viz_map.info = copy.deepcopy(self.map_info)
        viz_map.info.origin.position.z = self.map_info.origin.position.z + 0.1

        viz_map.header.frame_id = self.frame_id
        viz_map.header.stamp = rospy.Time.now()

        viz_array = copy.deepcopy(viz_array)

        # limit value range for OccupancyGrid type
        viz_array[np.isnan(viz_array)] = 99
        viz_array[viz_array > 99] = 99
        viz_array[viz_array < -99] = -99

        viz_array = viz_array.astype(int)
        viz_map.data = viz_array.T.flatten().tolist()

        publisher.publish(viz_map)

    def cell2world(self, cell_x, cell_y):
        """
        Convert from cell to world coordinates.
        :param cell_x: x-coordinate of cell
        :param cell_y: y-coordinate of cell
        :return: world coordinates
        """

        map_info = self.map_info
        world_x = cell_x * map_info.resolution + map_info.resolution / 2. + map_info.origin.position.x
        world_y = cell_y * map_info.resolution + map_info.resolution / 2. + map_info.origin.position.y

        return world_x, world_y

    def world2cell(self, world_x, world_y, clip=True):
        """
        Convert from world to cell coordinates.
        :param world_x: x-coordinate in world CS
        :param world_y: y-coordinate in world CS
        :param clip: whether to clip cell coordinates to integer values
        :return: cell coordinates
        """

        map_info = self.map_info
        cell_x = (world_x - map_info.origin.position.x) / map_info.resolution
        cell_y = (world_y - map_info.origin.position.y) / map_info.resolution

        if clip:
            return int(cell_x), int(cell_y)
        else:
            return cell_x, cell_y

    def convert_to_ros_path(self, path_cells, timestamp):
        """
        Convert path in cell coordinates to ros path.
        :param path_cells: path with cell coordinates
        :param timestamp: timestamp of path
        :return: converted ros path
        """

        nav_path = Path()
        nav_path.header.frame_id = "/map"  # self.frame_id
        nav_path.header.stamp = timestamp

        for cell in path_cells:
            world_pos = self.cell2world(cell[0], cell[1])

            one_pose = PoseStamped()
            one_pose.header = nav_path.header
            one_pose.pose.position.x = world_pos[0]
            one_pose.pose.position.y = world_pos[1]
            one_pose.pose.orientation.w = 1.0
            nav_path.poses.append(one_pose)

        return nav_path
