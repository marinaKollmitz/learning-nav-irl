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

import torch

from reward_component import RewardComponent


class ObstacleReward(RewardComponent):
    """
    Reward for static obstacles in the map.
    """

    def __init__(self, name="obstacle_reward", viz=True):
        """
        Initialize obstacle reward component.
        :param name: reward component name, for debugging
        :param viz: whether the reward component should be included for visualization
        """
        RewardComponent.__init__(self, name, viz)

    def compute_state_reward(self, nav_map, people_pos, start, goal):
        """
        Compute obstacle state reward.
        :param nav_map: map of the environment
        :param people_pos: cell coordinates of people (unused)
        :param start: start cell (unused)
        :param goal: goal cell (unused)
        :return: obstacle state reward
        """

        nav_grid = nav_map.grid

        # assign -inf reward to all inflated occupied cells
        obstacle_reward = np.zeros([nav_grid.shape[0], nav_grid.shape[1], 1])
        occupied = (nav_map.grid >= 99)  # inflated lethal obstacle cost
        obstacle_reward[occupied] = np.float("-inf")

        return torch.from_numpy(obstacle_reward)
