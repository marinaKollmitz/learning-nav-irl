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


import numpy as np


def find_closest(point, polyline):
    """
    Find closest point on polyline from query point.
    :param point: query point
    :param polyline: polyline
    :return: polyline index of closest point, closest distance to polyline,
             support points for cutting polyline at closest point
    """

    p = np.array([point[0], point[1], 0.0])

    closest_dist = np.float("inf")
    closest_idx = -1
    closest_support_points = []

    if len(polyline) > 1:

        for i in range(len(polyline) - 1):

            # check line segment
            x0 = np.array([polyline[i][0], polyline[i][1], 0.0])
            x1 = np.array([polyline[i + 1][0], polyline[i + 1][1], 0.0])

            # coordinate on line between x0 and x1
            t = -(x0 - p).dot(x1 - x0) / (np.linalg.norm(x1 - x0) ** 2)
            support_points = []

            if 0.0 < t < 1.0:
                # closest point to segment is between x0 and x1
                d = np.linalg.norm(np.cross((x1 - x0), (x0 - p))) / np.linalg.norm(x1 - x0)
                pnt = x0 + t * (x1 - x0)
                support_points.append([pnt[0], pnt[1]])
                support_points.append([x1[0], x1[1]])
            else:
                # closest point to segment is x0 or x1
                d0 = np.linalg.norm(x0 - p)
                d1 = np.linalg.norm(x1 - p)

                if d0 < d1:
                    d = d0
                    support_points.append(x0)
                    support_points.append(x1)
                else:
                    d = d1
                    support_points.append(x1)

            if d < closest_dist:
                closest_dist = d
                closest_idx = i
                closest_support_points = support_points

    elif len(polyline) == 1:
        x0 = np.array([polyline[0][0], polyline[0][1], 0.0])
        closest_dist = np.linalg.norm(x0 - p)
        closest_idx = 0
        closest_support_points = [polyline[0]]

    else:
        return None

    return closest_idx, closest_dist, closest_support_points


def convert_to_cell_cs(world_path, nav_map):
    """
    Convert path in world coordinate system to continuous-valued path in cell
    coordinate system.
    :param world_path: path in world coordinates
    :param nav_map: map of the environment
    :return: path converted to cell coordinate system
    """

    conti_cell_path = []

    for world_pos in world_path:
        conti_cell_path.append(nav_map.world2cell(world_pos[0], world_pos[1], clip=False))

    return conti_cell_path


def grid_discretize_path(world_path, nav_map, grid_actions):
    """
    Discretize continuous path to grid cells.
    :param world_path: path in world coordinates
    :param nav_map: map of the environment
    :param grid_actions: allowed grid move actions
    :return: discretized cell coordinates of path
    """

    conti_cell_path = convert_to_cell_cs(world_path, nav_map)
    discrete_path = []

    if len(world_path) > 0:
        start_cell = (int(conti_cell_path[0][0]), int(conti_cell_path[0][1]))
        end_cell = (int(conti_cell_path[-1][0]), int(conti_cell_path[-1][1]))

        discrete_path.append(start_cell)
        cell = start_cell

        # walk along grid from start to goal, checking neighbors according
        # to grid_actions and adding neighbor closest to path
        while cell != end_cell:

            min_d = np.float("inf")
            min_idx = -1
            min_support_points = []
            best_next = None

            # check grid neighbors
            for action in grid_actions:
                next_x = cell[0] + action[0]
                next_y = cell[1] + action[1]
                next_cell = (next_x, next_y)

                # avoid loops
                if len(discrete_path) <= 1 or next_cell != discrete_path[-2]:
                    idx, d, support_points = find_closest([next_x + 0.5, next_y + 0.5], conti_cell_path)
                    if d < min_d:
                        min_d = d
                        min_idx = idx
                        min_support_points = support_points
                        best_next = next_cell

            # append cell to discrete path
            cell = best_next
            discrete_path.append(cell)

            # "eat up" used path to avoid loops
            conti_cell_path = min_support_points + conti_cell_path[min_idx + 2::]

    return discrete_path
