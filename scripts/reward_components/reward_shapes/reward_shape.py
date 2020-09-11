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


class RewardShape:
    """
    Interface for distance reward shapes. Reward shapes can overwrite
    the required methods.
    """

    def __init__(self):
        """
        Initialize reward shape.
        """

        self.params = []

    def get_params(self):
        """
        Get trainable shape parameters.
        :return: parameters
        """

        return self.params

    def compute_reward(self, distance_map):
        """
        Compute rewards for all states according to the distances in distance_map.
        :param distance_map: array with distance value for each state
        :return: distance reward
        """

        pass
