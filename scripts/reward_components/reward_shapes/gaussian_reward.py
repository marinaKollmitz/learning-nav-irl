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

import torch

from reward_shape import RewardShape


class GaussianReward(RewardShape):
    """
    Calculate reward based on the distance of the state to a target
    (could be a person, for example), using a  squared-exponential distance
    function (also called Gaussian).
    """

    def __init__(self, init_scale, init_width, scale_range, width_range):
        """
        Initialize Gaussian reward shape.
        :param init_scale: initial scale parameter of the Gaussian
        :param init_width: initial width parameter of the Gaussian
        :param scale_range: valid parameter range for scale
        :param width_range: valid parameter range for width
        """

        RewardShape.__init__(self)

        self.gaussian_scale = torch.tensor([init_scale])
        self.gaussian_width = torch.tensor([init_width])

        # model parameters and parameter ranges
        self.params = [self.gaussian_scale, self.gaussian_width]
        self.valid_params_ranges = [scale_range, width_range]

    def compute_reward(self, distance_map):
        """
        Compute Gaussian distance reward.
        :param distance_map: array with distance value for each state
        :return: Gaussian distance reward
        """

        gaussian_reward = self.gaussian_scale * \
            torch.exp(-torch.pow(distance_map, 2) / (2 * torch.pow(self.gaussian_width, 2)))

        return gaussian_reward
