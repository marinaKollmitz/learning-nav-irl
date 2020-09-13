"""
learning-nav-irl: Learning human-aware robot navigation via IRL.
Copyright (C) 2020  Torsten Koller, Marina Kollmitz

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

from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid


def publish_demo_markers(demos, timestamp, publisher):
    """
    Publish marker array of demonstrations for visualization in rviz.
    :param demos:
    :return:
    """
    markers = MarkerArray()
    for i, demo in enumerate(demos):
        markers.markers.append(demo.get_rviz_marker(i, timestamp))
    publisher.publish(markers)


def visualize_grid(viz_array, nav_map, timestamp, publisher):
    """
    Publish array as ros OccupancyGrid for visualization.
    :param viz_array: array for visualization
    :param publisher: ros publisher
    """

    viz_map = OccupancyGrid()
    viz_map.info = copy.deepcopy(nav_map.map_info)
    viz_map.info.origin.position.z = nav_map.map_info.origin.position.z + 0.1

    viz_map.header.frame_id = nav_map.frame_id
    viz_map.header.stamp = timestamp

    viz_array = copy.deepcopy(viz_array)

    # limit value range for OccupancyGrid type
    viz_array[np.isnan(viz_array)] = 99
    viz_array[viz_array > 99] = 99
    viz_array[viz_array < -99] = -99

    viz_array = viz_array.astype(int)
    viz_map.data = viz_array.T.flatten().tolist()

    publisher.publish(viz_map)


def publish_people_marker(people_pos, nav_map, timestamp, publisher):
    """
    Get rviz visualization marker for people poses.
    :return: visualization marker
    """

    people_markers = MarkerArray()
    person_marker = Marker()

    person_marker.header.frame_id = nav_map.frame_id
    person_marker.header.stamp = timestamp
    person_marker.type = Marker.MESH_RESOURCE
    person_marker.mesh_resource = "package://learning-nav-irl/meshes/human/meshes/standing.dae"
    person_marker.mesh_use_embedded_materials = True
    person_marker.scale.x = 1.0
    person_marker.scale.y = 1.0
    person_marker.scale.z = 1.0

    for i, person_pos in enumerate(people_pos):
        person_marker.id = i
        person_x, person_y = nav_map.cell2world(person_pos[0], person_pos[1])
        person_marker.pose.position.x = person_x
        person_marker.pose.position.y = person_y
        # person orientation hard-coded for visualization
        person_marker.pose.orientation.z = 0.7071068
        person_marker.pose.orientation.w = 0.7071068
        people_markers.markers.append(copy.copy(person_marker))

    publisher.publish(people_markers)
