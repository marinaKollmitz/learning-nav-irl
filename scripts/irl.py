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
import os

import matplotlib
import rospy
import torch
from visualization_msgs.msg import MarkerArray

from logging_helper import setup_logger

# use matplotlib in non-gui mode
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = setup_logger("irl")
people_pub = rospy.Publisher("people_viz", MarkerArray, queue_size=1)


def compute_likelihood(demo_path, policy, actions):
    """
    Compute log-likelihood of demonstration.
    :param demo_path: demonstration trajectory
    :param policy: MDP policy
    :param actions: MDP actions
    :return: demonstration log-likelihood
    """
    log_likelihood = 0

    # log-likelihood of trajectory is given by the sum of log-likelihoods
    # of the state-action pairs in the trajectory, which translate to the
    # log-likelihoods of taking the actions from the states.
    if len(demo_path) > 1:
        for i in range(1, len(demo_path)):
            state = demo_path[i - 1]
            next_state = demo_path[i]
            action = np.array([next_state[0] - state[0], next_state[1] - state[1]])

            # find index of action
            a_idx = np.argwhere(np.sum(abs(actions - action), axis=1) == 0)
            p_action = policy[state[0], state[1], a_idx[0, 0]]

            if p_action > 0:
                # log-likelihood of taking the action from the state
                log_likelihood += np.log(p_action)

    return log_likelihood


def compute_demo_reward(demo_path, actions, reward_map):
    """
    Compute accumulated reward along demonstration.
    :param demo_path: demonstration trajectory
    :param actions: MDP actions
    :param reward_map: state-action rewards
    :return: reward along demonstration
    """

    # torch array to mask state-actions in demonstration
    demo_mask = torch.zeros(reward_map.shape)

    if len(demo_path) > 1:
        for i in range(1, len(demo_path)):
            state = demo_path[i - 1]
            next_state = demo_path[i]
            action = np.array([next_state[0] - state[0], next_state[1] - state[1]])

            # find index of action
            a_idx = np.argwhere(np.sum(abs(actions - action), axis=1) == 0)
            demo_mask[state[0], state[1], a_idx] = 1

    demo_reward = torch.mul(demo_mask, reward_map)

    # filter nans
    demo_reward[demo_reward != demo_reward] = 0

    sum_demo_reward = torch.sum(demo_reward)

    return sum_demo_reward


def compute_expected_reward(state_visitations, policy, reward_map):
    """
    Compute expected reward of agent through MDP.
    :param state_visitations: state visitations
    :param policy: MDP policy
    :param reward_map: state-action rewards
    :return: expected reward through MDP
    """
    state_visitations = state_visitations[:, :, np.newaxis]
    exp_states_actions = torch.from_numpy(np.multiply(policy, state_visitations))

    exp_reward = torch.mul(exp_states_actions, reward_map)

    # filter nans
    exp_reward[exp_reward != exp_reward] = 0

    sum_exp_reward = torch.sum(exp_reward)

    return sum_exp_reward


def compute_gradient(loss, reward_function):
    """
    Compute gradient of loss function.
    :param loss: loss
    :param reward_function: reward function
    :return: gradient
    """

    # backpropagate for gradients
    loss.backward()
    grad = []
    for param in reward_function.reward_params:
        grad.append(param.grad)

    return grad


def sort_demos(demonstrations):
    """
    Sort demos with same start and goal together
    :param demonstrations: demonstrations
    :return: sorted demonstration set
    """

    demo_sets = []

    for demo in demonstrations:
        start = demo.grid_path[0]
        goal = demo.grid_path[-1]
        people_pos = demo.people_pos

        if len(demo_sets) == 0 \
                or goal != demo_sets[-1]["goal"] \
                or start != demo_sets[-1]["start"] \
                or people_pos != demo_sets[-1]["people_pos"]:
            # make new demo set
            demo_dict = {"start": start, "goal": goal, "people_pos": people_pos, "demos": [demo]}
            demo_sets.append(demo_dict)
        else:
            # append demo to demo dict
            demo_sets[-1]["demos"].append(demo)

    return demo_sets


def irl(nav_map, mdp, demonstrations, max_iters=500, epsilon_objective=0.01,
        epsilon_grad=0.1, lr=0.1):
    """
    Perform inverse reinforcement learning to find most likely reward function
    parameters from demonstrations.
    :param nav_map: map of the environment
    :param mdp: MDP for navigation task
    :param demonstrations: demonstrations
    :param max_iters: maximum number of optimization steps
    :param epsilon_objective: objective convergence threshold
    :param epsilon_grad: gradient convergence threshold
    :param lr: learning rate
    :return: optimized reward function parameters
    """

    dir_path = os.path.dirname(os.path.realpath(__file__))

    if len(demonstrations) == 0:
        logger.warn("no demonstrations available")
        return

    demo_sets = sort_demos(demonstrations)

    # for plotting objective
    likelihoods = [-np.float('inf')]
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    plt.title("log likelihood of demonstrations")

    # optimizer for adapting reward function parameters
    optimizer = torch.optim.Adam(mdp.reward_function.reward_params, lr=lr, weight_decay=0.01 * lr)
    logger.info("reward_params: {}".format(mdp.reward_function.reward_params))

    # optimization loop
    for i in range(max_iters):

        likelihood = 0
        reward_diff = torch.tensor(0.0)
        optimizer.zero_grad()

        for demo_set in demo_sets:

            start = demo_set["start"]
            goal = demo_set["goal"]
            people_pos = demo_set["people_pos"]
            people_pub.publish(demo_set["demos"][0].get_people_marker(rospy.Time.now()))

            # solve (stochastic) MDP with current reward parameters
            values, q_values = mdp.value_iteration(nav_map, people_pos, start, goal, softmax=True)
            policy = mdp.compute_policy(values, q_values, stochastic=True)

            # calculate state visitations based on stochastic policy
            state_visitations = mdp.propagate_policy(nav_map, start, goal, policy)
            reward_map = mdp.reward_function.compute_reward(nav_map, mdp.actions, people_pos,
                                                            start, goal, viz=True, grad=True)

            # calculate objective (Likelihood of demonstrations), reward of demos and expected reward
            demo_r = torch.tensor(0.0)
            for demo in demo_set["demos"]:
                likelihood += compute_likelihood(demo.grid_path, policy, mdp.actions)
                demo_r += compute_demo_reward(demo.grid_path, mdp.actions, reward_map)

            exp_r = compute_expected_reward(state_visitations, policy, reward_map)

            # reward diff for gradient calculation
            reward_diff += (demo_r - len(demo_set["demos"]) * exp_r)

        # calculate gradient with respect to loss (negate because we want to maximize objective)
        grad = compute_gradient(-1 * reward_diff, mdp.reward_function)
        logger.info("grad: {}".format(grad))

        # adapt parameters
        optimizer.step()

        # clamp parameters to valid ranges for stability
        with torch.no_grad():
            for k in range(len(mdp.reward_function.reward_params)):
                mdp.reward_function.reward_params[k].clamp_(mdp.reward_function.valid_params_ranges[k][0],
                                                            mdp.reward_function.valid_params_ranges[k][1])

        logger.info("reward_params: {}".format(mdp.reward_function.reward_params))

        # adapt learning rate
        if (i != 0) and (i % 100 == 0):
            lr = lr * 0.8
            optimizer = torch.optim.Adam(mdp.reward_function.reward_params, lr=lr, weight_decay=0.01 * lr)
            logger.info("adapting learning rate to %f", lr)

        # save loss plot to file
        likelihoods.append(likelihood)
        ax.clear()
        ax.plot(likelihoods)
        fig.savefig(os.path.join(dir_path, '../demo_likelihood.png'))
        plt.pause(0.01)

        # check for convergence
        max_grad = max(np.abs(grad))
        if abs(likelihoods[-1] - likelihoods[-2]) < epsilon_objective and max_grad < epsilon_grad:
            logger.info("training converged")
            break

        # check for ctrl-c
        if rospy.is_shutdown():
            return None

    logger.info("Training done. Such IRL magic!")
    return mdp.reward_function.reward_params
