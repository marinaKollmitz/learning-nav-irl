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

import copy
import numpy as np
from scipy.misc import logsumexp

import rospy
from nav_msgs.msg import OccupancyGrid

from logging_helper import setup_logger


class GridMDP:
    """
    Markov Decision Process (MDP) in a grid. This MDP considers only
    deterministic state transitions.
    """

    # grid actions for moving to neighbors on Manhattan grid
    MANHATTAN_ACTIONS = np.array([[-1, 0], [0, -1], [0, 1], [1, 0]])

    # grid actions for moving to neighbors in diagonal fashion
    DIAGONAL_ACTIONS = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1],
                                 [0, 1], [1, -1], [1, 0], [1, 1]])

    def __init__(self, grid_actions, reward_function):
        """
        Initialize MDP.
        :param grid_actions: array of tuples with index increments for actions
        :param reward_function: MDP reward function
        """

        self.logger = setup_logger('mdp')

        self.actions = grid_actions
        self.reward_function = reward_function

        self.state_viz_pub = rospy.Publisher("state_visitations", OccupancyGrid, queue_size=1)

    def plan_path(self, nav_map, people_pos, start, goal, softmax=True,
                  stochastic_policy=True):
        """
        Plan a path through MDP from start to goal.
        :param nav_map: map of the environment
        :param people_pos: cell coordinates of people
        :param start: start cell
        :param goal: goal cell
        :param softmax: use MaxEnt IRL softmax in value iteration
                        (False: use max like in standard MDPs)
        :param stochastic_policy: stochastic policy from MaxEnt IRL
                                  (False: deterministic policy from max Q values)
        :return: planned path
        """

        path = []

        # solve MDP
        values, q_values = self.value_iteration(nav_map, people_pos, start, goal, softmax=softmax)
        policy = self.compute_policy(values, q_values, stochastic=stochastic_policy)

        # follow policy to generate navigation path
        state = start

        while state != goal:
            path.append(state)

            probs = policy[state]

            # check for invalid states (-inf value): no action has >0 probability
            if np.sum(probs) == 0:
                self.logger.warn("planning state not valid, is the cell occupied?")
                return [start]

            # get next state according to policy. We use likelihood sampling
            # in case there is more than 1 non-zero action probability
            idx = np.random.choice(range(len(probs)), p=probs)
            best_action = self.actions[idx]

            next_x = state[0] + best_action[0]
            next_y = state[1] + best_action[1]
            state = (next_x, next_y)

        path.append(goal)
        return path

    def propagate_to_neighbors(self, state_visits_t, policy):
        """
        Propagate state visitations through MDP along policy for one time step t
        :param state_visits_t: state visitations at time step t
        :param policy: policy through MDP
        :return: state visitations at time step t+1
        """

        actions = self.actions
        grid_shape = state_visits_t.shape
        state_visits_tp1 = np.zeros(grid_shape)

        # for padding of neighborhood action probs
        p = np.max(abs(actions))

        for a, action in enumerate(actions):
            # calculate p(a|s)*p_t(s) for that action
            neighbor_probs_action = policy[:, :, a] * state_visits_t

            padded = np.pad(neighbor_probs_action, (p, p), 'constant', constant_values=0.0)

            # crop padded array for neighbors of action, similar to calculating
            # values_from_neighbors(..)
            crop = np.array([[p, p + grid_shape[0]], [p, p + grid_shape[1]]])
            crop[0] -= action[0]
            crop[1] -= action[1]

            # add for all actions
            state_visits_tp1 += padded[crop[0, 0]:crop[0, 1], crop[1, 0]:crop[1, 1]]

        return state_visits_tp1

    def propagate_policy(self, nav_map, start, goal, policy, min_prob_mass=0.05,
                         viz=True):
        """
        Propagate policy trough MDP from start to goal. This is often referred
        to as calculating the state visitation (frequencies).
        :param nav_map: map of the environment
        :param start: start cell
        :param goal: goal cell
        :param policy: policy through MDP
        :param min_prob_mass: probability mass threshold for policy propagation
        :param viz: whether to publish state visitations for rviz
        :return: accumulated state visitation frequencies
        """

        grid_shape = nav_map.grid.shape
        state_visits = np.zeros(grid_shape)  # state visitation counts
        state_visits_t = np.zeros(grid_shape)  # at time step t

        # initial state distribution
        state_visits_t[start] = 1.0

        state_visits += state_visits_t
        prob_mass = 1.0

        # propagate state visitations state_visits_t to neighbors according to policy
        while prob_mass > min_prob_mass:
            # propagate state visitations for 1 step
            state_visits_tp1 = self.propagate_to_neighbors(state_visits_t, policy)

            # absorb in terminal state
            state_visits_tp1[goal] = 0

            # update overall state visitation and for time step t
            state_visits += state_visits_tp1
            state_visits_t = copy.copy(state_visits_tp1)

            prob_mass = np.sum(state_visits_tp1)

        if viz:
            nav_map.visualize_grid(50 * state_visits, self.state_viz_pub)

        return state_visits

    def values_from_neighbors(self, values):
        """
        Calculate the neighbor values for all states by padding and shifting
        the state array onto neighbor positions.
        :param values: state values array
        :return: array with neighbor values
        """

        actions = self.actions
        v_shape = values.shape
        neighbor_values = np.zeros([v_shape[0], v_shape[1], len(actions)])

        # pad value array
        p = np.max(abs(actions))
        values = np.pad(values, (p, p), 'constant', constant_values=(np.float("-inf")))

        for a, action in enumerate(actions):
            # crop padded array for neighbors of action
            crop = np.array([[p, p + v_shape[0]], [p, p + v_shape[1]]])
            crop[0] += action[0]
            crop[1] += action[1]

            neighbor_values[:, :, a] = values[crop[0, 0]:crop[0, 1], crop[1, 0]:crop[1, 1]]

        return neighbor_values

    def value_iteration(self, nav_map, people_pos, start, goal, softmax=True):
        """
        Performs value iteration to solve the MDP.
        :param nav_map: map of the environment
        :param people_pos: cell coordinates of people
        :param start: start cell
        :param goal: goal cell
        :param softmax: use softmax in value iteration
                        (False: deterministic max)
        :return: converged values and Q-values
        """

        states = nav_map.grid

        # get state-action reward_components
        rewards = self.reward_function.compute_reward(nav_map, self.actions, people_pos,
                                                      start, goal).detach().numpy()

        # initialize values array to -inf
        values = np.float("-inf") * np.ones(states.shape)

        # set value of goal state to zero (no more reward possible)
        values[goal] = 0.0

        converged = False
        while not converged:

            # calculate Q-values (with deterministic state transitions)
            v_neighbors = self.values_from_neighbors(values)
            q_values = v_neighbors + rewards

            if softmax:
                # new values as softmax over actions of reward_components and previous values
                next_values = logsumexp(q_values, axis=2)

            else:  # deterministic max
                # new values as max over actions of reward_components and previous values
                next_values = np.max(q_values, axis=2)

            next_values = np.array(next_values)

            # set value of goal state to zero (no more reward possible)
            next_values[goal] = 0.0

            # check for unbounded values, causing divergence
            if np.max(next_values) > 0:
                self.logger.error("MDP unbounded values!")
                return None

            # check for convergence
            converged = np.allclose(values, next_values, atol=0.01)

            values = copy.copy(next_values)

        return values, q_values

    def compute_policy(self, values, q_values, stochastic=True):
        """
        Computes policy from values and Q-values.
        :param values: state values array
        :param q_values: state-action Q-values array
        :param stochastic: compute stochastic policy
                           (False: deterministic policy)
        :return: computed policy
        """

        # bring values in the shape of the q_values
        values_qshape = np.expand_dims(values, axis=2)
        values_qshape = np.repeat(values_qshape, len(self.actions), axis=2)

        if stochastic:
            # stochastic (softmax) policy is exp(q_value - value)
            policy = np.exp(np.nan_to_num(q_values) - np.nan_to_num(values_qshape))

        else:  # deterministic policy
            # take action with highest Q-value
            max_val = np.max(q_values, axis=2, keepdims=True)

            # find entries where q_values are maximum (can be more than 1)
            is_max = (q_values == max_val).astype(float)

            # normalize to probabilities (if there is more than 1 max, all choices are equally likely)
            policy = is_max / np.sum(is_max, axis=2, keepdims=True)

        # set policy for -inf q_values to zero: never take that action!
        policy[q_values == np.float("-inf")] = 0

        return policy
