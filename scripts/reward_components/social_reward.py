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


def compute_person_dist(nav_grid, person_pos):
    """
    Compute distance map between states and person
    :param nav_grid: MDP states
    :param person_pos: cell coordinate of person
    :return: distance map
    """

    xs = np.arange(nav_grid.shape[0])
    ys = np.arange(nav_grid.shape[1])
    xs = np.expand_dims(xs, axis=1)
    ys = np.expand_dims(ys, axis=0)

    person_dist = np.hypot(xs - person_pos[0], ys - person_pos[1])
    person_dist = torch.from_numpy(np.expand_dims(person_dist, axis=2))

    return person_dist


class SocialReward(RewardComponent):
    """
    Social reward evaluating interaction distances.
    """

    def __init__(self, reward_shape, name="social_reward",
                 combine_people_rewards="accumulate", viz=True):
        """
        Initialize social reward component.
        :param reward_shape: distance reward shape for social reward.
        :param name: reward component name, for debugging
        :param combine_people_rewards: method to combine rewards of multiple
                                       people ("accumulate" or "minimum")
        :param viz: whether the reward component should be included for visualization
        """

        RewardComponent.__init__(self, name, viz)

        self.reward_shape = reward_shape
        self.params = self.reward_shape.params
        self.valid_params_ranges = self.reward_shape.valid_params_ranges

        self.combine_method = combine_people_rewards

    def compute_state_reward(self, nav_map, people_pos, start, goal):
        """
        Compute social state reward.
        :param nav_map: map of the environment
        :param people_pos: cell coordinates of people
        :param start: start cell
        :param goal: goal cell
        :return: social reward
        """

        nav_grid = nav_map.grid
        people_reward = torch.zeros([nav_grid.shape[0], nav_grid.shape[1], 1])

        for person_pos in people_pos:
            person_dist = compute_person_dist(nav_grid, person_pos)
            person_reward = self.reward_shape.compute_reward(person_dist)

            if self.combine_method == "minimum":
                people_reward = np.minimum(people_reward, person_reward)
            else:
                people_reward += person_reward
                if self.combine_method != "accumulate":
                    self.logger.warn("reward combine method %s not available, "
                                     "defaulting to accumulate", self.combine_method)

        return people_reward
