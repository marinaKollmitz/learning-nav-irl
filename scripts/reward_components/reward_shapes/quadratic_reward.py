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

import torch

from reward_shape import RewardShape


class QuadraticReward(RewardShape):
    """
    Calculate reward based on the distance of the state to a target
    (could be the nominal path, for example) using a quadratic distance function.
    """

    def __init__(self, init_scale, scale_range, scale_for_grad=0.01):
        """
        Initialize quadratic reward shape.
        :param init_scale: initial scale parameter of the reward
        :param scale_range: valid parameter range for scale
        :param scale_for_grad: reward scale factor for gradient magnitude
        """

        RewardShape.__init__(self)

        self.reward_scale = torch.tensor([init_scale])
        self.scale_for_grad = scale_for_grad

        # model parameters and parameter ranges
        self.params = [self.reward_scale]
        self.valid_params_ranges = [scale_range]

    def compute_reward(self, distance_map):
        """
        Compute quadratic distance reward.
        :param distance_map: array with distance value for each state
        :return: (scaled) quadratic distance reward
        """

        quad_reward = self.reward_scale * torch.pow(distance_map, 2)

        # scale reward for gradient magnitudes
        quad_reward_scaled = self.scale_for_grad * quad_reward

        return quad_reward_scaled
