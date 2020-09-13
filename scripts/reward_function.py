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

import rospy
import torch
from nav_msgs.msg import OccupancyGrid

from logging_helper import setup_logger
from viz_helper import visualize_grid


class RewardFunction:
    """
    Reward function assigning real-valued rewards to state-action pairs, can
    be used in MDPs.
    """

    def __init__(self):
        """
        Initialize reward function with empty reward components array and empty
        parameter array.
        """

        self.reward_components = []
        self.reward_params = []
        self.valid_params_ranges = []

        self.logger = setup_logger("reward_function")

        # publish reward as occupancy grid for visualization in rviz
        self.reward_pub = rospy.Publisher("reward", OccupancyGrid, queue_size=1)

    def add_component(self, reward_component):
        """
        Add reward component.
        :param reward_component: reward component (following RewardComponent
                                 interface from reward_components/reward_component.py)
        """

        self.reward_components.append(reward_component)
        self.reward_params.extend(reward_component.get_params())
        self.valid_params_ranges.extend(reward_component.valid_params_ranges)

    def compute_state_action_rewards(self, nav_map, actions, people_pos, start, goal, viz=True):
        """
        Calculate state and action reward from all components.
        :param nav_map: map of the environment
        :param actions: MDP actions
        :param people_pos: cell coordinates of people
        :param start: start cell
        :param goal: goal cell
        :param viz: whether reward should be visualized in rviz
        :return: state reward and action reward
        """

        nav_grid = nav_map.grid

        state_reward = torch.zeros([nav_grid.shape[0], nav_grid.shape[1], 1])
        action_reward = torch.zeros([len(actions)])
        viz_reward = torch.zeros([nav_grid.shape[0], nav_grid.shape[1], 1])

        for reward_component in self.reward_components:
            state_reward_comp = reward_component.compute_state_reward(nav_map, people_pos, start, goal)
            action_reward_comp = reward_component.compute_action_reward(actions)

            # Add state reward of reward component, if implemented
            if state_reward_comp is not None:

                # Verify that reward has correct shape
                if state_reward_comp.shape == state_reward.shape:
                    state_reward += state_reward_comp

                    if reward_component.viz:
                        viz_reward += state_reward_comp

                else:
                    self.logger.warn("state reward of component %s has wrong shape",
                                     reward_component.name)

            # Add action reward of reward component, if implemented.
            if action_reward_comp is not None:

                # Verify that reward has correct shape.
                if action_reward_comp.shape == action_reward.shape:
                    action_reward += action_reward_comp
                else:
                    self.logger.warn("action reward of component %s has wrong shape",
                                     reward_component.name)

        if viz:
            visualize_grid(-5 * viz_reward.detach().numpy()[:, :, 0], nav_map,
                           rospy.Time.now(), self.reward_pub)

        return state_reward, action_reward

    def compute_reward(self, nav_map, actions, people_pos, start, goal, viz=False,
                       grad=False):
        """
        Calculate combined state-action rewards from reward components
        :param nav_map: map of environment
        :param actions: mdp grid actions
        :param people_pos: cell coordinates of people
        :param start: start cell
        :param goal: goal cell
        :param viz: whether reward should be visualized in rviz
        :param grad: whether reward function gradients should be computed
        :return: state-action reward array
        """

        for param in self.reward_params:
            param.requires_grad_(grad)

        state_reward, action_reward = \
            self.compute_state_action_rewards(nav_map, actions, people_pos, start, goal, viz=viz)

        # Bring state and action rewards in correct state-action shape:
        # states x actions.

        nav_grid = nav_map.grid

        # broadcast action reward to state-action shape
        action_rewards = torch.zeros([nav_grid.shape[0], nav_grid.shape[1], len(actions)]) + action_reward

        # expand state reward to state-action shape
        state_rewards = state_reward.repeat(1, 1, len(actions))

        # calculate combined state-action reward
        state_action_rewards = state_rewards + action_rewards

        return state_action_rewards
