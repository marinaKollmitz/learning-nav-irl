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

from logging_helper import setup_logger


class RewardComponent:
    """
    Interface for reward function components. Reward components can overwrite
    the required methods.
    """

    def __init__(self, name, viz):
        """
        Initialize component.
        :param name: reward component name, for debugging
        :param viz: whether the reward component should be included for visualization
        """

        self.name = name
        self.logger = setup_logger(name)
        self.viz = viz

        self.params = []
        self.valid_params_ranges = []

    def get_params(self):
        """
        Get component's trainable parameters.
        :return: parameter set
        """

        return self.params

    def compute_state_reward(self, nav_map, people_pos, start, goal):
        """
        Compute state reward of component, implementation optional.
        :param nav_map: map of the environment
        :param people_pos: cell coordinates of people
        :param start: start cell
        :param goal: goal cell
        :return: state reward, torch tensor of shape [nav_map.shape[0], nav_map.shape[1], 1]
        """

        pass

    def compute_action_reward(self, actions):
        """
        Compute action reward of component, implementation optional.
        :param actions: MDP actions
        :return: action reward, torch tensor of shape [len(actions)]
        """

        pass
