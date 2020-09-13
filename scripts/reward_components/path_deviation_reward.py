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

import numpy as np

import rospy
import torch
from mdp import GridMDP
from nav_msgs.msg import OccupancyGrid
from reward_function import RewardFunction

from obstacle_reward import ObstacleReward
from reward_component import RewardComponent
from step_reward import StepReward
from viz_helper import visualize_grid


class PathDeviationReward(RewardComponent):
    """
    Calculate reward for deviating from the direct path between a start and
    a goal cell that takes only static obstacles into account. Since multiple paths
    can be optimal in a grid, calculate path deviation reward based on the
    minimum distance to any of them, as described below in compute_path_dist.
    """

    def __init__(self, reward_shape, grid_actions, name="path_dev_reward", viz=True):
        """
        Initialize path deviation reward component
        :param reward_shape: distance reward shape for path deviation reward
        :param grid_actions: MDP actions
        :param name: reward component name, for debugging
        :param viz: whether the reward component should be included for visualization
        """

        RewardComponent.__init__(self, name, viz)

        # MDP for calculating direct path policy
        reward_direct_path = RewardFunction()
        reward_direct_path.add_component(ObstacleReward())
        reward_direct_path.add_component(
            StepReward(-1.0, [np.float('-inf'), 0], "direct_path_step"))
        self.direct_path_client = GridMDP(grid_actions, reward_direct_path)

        # MDP for calculating path distance based on direct path policy
        reward_path_dist = RewardFunction()
        self.step_weight = -1.0
        reward_path_dist.add_component(
            StepReward(self.step_weight, [np.float('-inf'), 0], "path_dist_step"))
        self.path_dist_client = GridMDP(GridMDP.DIAGONAL_ACTIONS, reward_path_dist)

        # reward shape for distance to path
        self.dist_reward = reward_shape
        self.params = self.dist_reward.params
        self.valid_params_ranges = self.dist_reward.valid_params_ranges

        self.direct_path_pub = rospy.Publisher("direct_path_cells", OccupancyGrid, queue_size=1)

    def compute_path_dist(self, nav_map, start, goal):
        """
        For every state, calculate the minimum distance to the direct path
        between start and goal, taking only static obstacles into account.
        :param nav_map: map of environment
        :param start: start cell
        :param goal: goal cell
        :return: distance map

        To solve the direct path ambiguity that multiple paths between start and
        goal can have minimal length, we first compute the optimal policy
        through MDP with static obstacles, using the path_dist_client mdp.
        Propagating the optimal policy from the start to the goal determines the
        states along all shortest paths, direct_path_cells. Afterwards, we perform
        a wavefront expansion from the direct_path_cells outwards to determine
        the distance of each state to any direct path. This is achieved by
        performing value iteration with the path_dist_client mdp with a step
        weight of -1.0 and treating the direct_path_cells as goal cells. Each
        state will have the minimum step reward to the direct path, and we
        can calculate the path distance with the step weight.
        """

        # solve direct_path_client mdp for optimal policy
        values, q_values = \
            self.direct_path_client.value_iteration(nav_map, [], start, goal,
                                                    softmax=False)
        direct_path_policy = \
            self.direct_path_client.compute_policy(values, q_values, stochastic=False)

        # propagate policy for states along direct paths
        direct_path_cells = self.direct_path_client.propagate_policy(nav_map,
                                                                     start, goal,
                                                                     direct_path_policy,
                                                                     viz=False)
        direct_path_cells[goal] = 1.0

        # visualize direct path cells
        visualize_grid((direct_path_cells > 0) * 100, nav_map, rospy.Time.now(),
                       self.direct_path_pub)

        # perform wavefront expansion from direct_path_cells outwards by solving
        # the path_dist_client mdp
        wavefront_start = direct_path_cells > 0
        step_values, _ = \
            self.path_dist_client.value_iteration(nav_map, [], None, wavefront_start,
                                                  softmax=False)

        # convert step values to path distances
        path_dists = torch.from_numpy(self.step_weight * np.expand_dims(step_values, axis=2))

        return path_dists

    def compute_state_reward(self, nav_map, people_pos, start, goal):
        """
        Compute path deviation state reward.
        :param nav_map: map of environment
        :param people_pos: cell coordinates of people (unused)
        :param start: start cell
        :param goal: goal cell
        :return: path deviation reward
        """

        direct_path_dist = self.compute_path_dist(nav_map, start, goal)
        deviation_reward = self.dist_reward.compute_reward(direct_path_dist)

        return deviation_reward
