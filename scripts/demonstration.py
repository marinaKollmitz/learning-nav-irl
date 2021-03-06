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

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from path_discretizer import grid_discretize_path


class Demonstration:
    """
    Maintain a demonstration trajectory.
    """

    def __init__(self, world_poses, nav_map, grid_actions, people_pos):
        """
        Initialize demonstration trajectory.
        :param world_poses: recorded trajectory in world coordinates
        :param nav_map: map of the environment
        :param grid_actions: MDP actions
        :param people_pos: cell coordinates of people
        """

        self.world_poses = world_poses
        self.nav_map = nav_map
        self.people_pos = people_pos
        self.grid_path = grid_discretize_path(world_poses, nav_map, grid_actions)

    def get_rviz_marker(self, marker_id, timestamp):
        """
        Get rviz visualization Marker for demonstration trajectory.
        :param marker_id: ID of marker
        :param timestamp: timestamp for publishing
        :return: visualization marker
        """

        demo_marker = Marker()
        demo_marker.header.frame_id = self.nav_map.frame_id
        demo_marker.header.stamp = timestamp
        demo_marker.type = Marker.LINE_STRIP
        demo_marker.action = Marker.MODIFY
        demo_marker.color.g = 1.0
        demo_marker.color.a = 1.0
        demo_marker.scale.x = 0.05
        demo_marker.pose.orientation.w = 1.0
        demo_marker.id = marker_id

        for position in self.world_poses:
            pnt = Point()
            pnt.x = position[0]
            pnt.y = position[1]
            pnt.z = 0.12
            demo_marker.points.append(pnt)

        return demo_marker
