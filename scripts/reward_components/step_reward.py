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


class StepReward(RewardComponent):
    """
    Reward for taking a step in the MDP grid.
    """

    def __init__(self, init_step_weight, step_weight_range, name="step_reward", viz=False):
        """
        Initialize step reward component.
        :param init_step_weight: initial step weight parameter
        :param step_weight_range: valid step weight parameter range
        :param name: reward component name, for debugging
        :param viz: whether the reward component should be included for visualization
        """

        RewardComponent.__init__(self, name, viz)

        self.step_param = torch.tensor([init_step_weight])
        self.params = [self.step_param]
        self.valid_params_ranges = [step_weight_range]

    def compute_action_reward(self, actions):
        """
        Compute step action reward.
        :param actions: MDP actions
        :return: step reward
        """

        # reward: param * size of step
        r_step = self.step_param * torch.from_numpy(np.hypot(actions[:, 0], actions[:, 1]))

        return r_step
