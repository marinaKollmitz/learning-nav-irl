#! /usr/bin/env python

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
import pickle
import sys

import rospy
from visualization_msgs.msg import MarkerArray

from irl import irl
from mdp import GridMDP
from reward_function import RewardFunction
from reward_components.obstacle_reward import ObstacleReward
from reward_components.path_deviation_reward import PathDeviationReward
from reward_components.social_reward import SocialReward
from reward_components.step_reward import StepReward
from reward_components.reward_shapes.gaussian_reward import GaussianReward
from reward_components.reward_shapes.quadratic_reward import QuadraticReward


def setup_reward_function(grid_actions):
    """
    Setup reward function with experiment reward components.
    :param grid_actions: MDP actions
    :return: reward function object
    """
    reward_function = RewardFunction()
    reward_function.add_component(StepReward(-5.0, [np.float("-inf"), 0]))
    reward_function.add_component(SocialReward(
        GaussianReward(-5.0, 5.0, [np.float("-inf"), 0], [0, np.float("inf")])))
    reward_function.add_component(PathDeviationReward(
        QuadraticReward(-0.01, [np.float("-inf"), 0]), grid_actions))
    reward_function.add_component(ObstacleReward())

    return reward_function


def setup_mdp():
    """
    Setup MDP with diagonal grid actions.
    :return: MDP object
    """
    grid_actions = GridMDP.DIAGONAL_ACTIONS
    reward_function = setup_reward_function(grid_actions)
    grid_mdp = GridMDP(grid_actions, reward_function)

    return grid_mdp


def publish_demo_markers(demos):
    """
    Publish marker array of demonstrations for visualization in rviz.
    :param demos:
    :return:
    """
    markers = MarkerArray()
    marker_pub = rospy.Publisher("demos", MarkerArray, latch=True, queue_size=1)
    for i, demo in enumerate(demos):
        markers.markers.append(demo.get_rviz_marker(i, rospy.Time.now()))
    marker_pub.publish(markers)


def retrain_irl(experiment_path):
    """
    Retrain on recorded demonstrations from experiments.
    :param experiment_path: Path to experiment run
    """
    nav_map_path = os.path.join(experiment_path, "nav_map.pkl")
    nav_map = pickle.load(open(nav_map_path))
    nav_map.publish()

    demo_path = os.path.join(experiment_path, "force_demos.pkl")
    demos = pickle.load(open(demo_path))

    grid_mdp = setup_mdp()

    # first passing cycle
    irl_demos = demos['passing_1']
    publish_demo_markers(irl_demos)
    irl(nav_map, grid_mdp, irl_demos)

    # second passing cycle
    irl_demos = demos['passing_2']
    publish_demo_markers(irl_demos)
    irl(nav_map, grid_mdp, irl_demos)


if __name__ == '__main__':
    rospy.init_node('learn_from_demo_node')
    rospy.loginfo("learn from demonstration node started")

    if len(sys.argv) < 2:
        rospy.logwarn("usage: rosrun learning-nav-irl experiment_demo_irl.py <experiment_folder_path>")
    else:
        exp_path = sys.argv[1]
        if os.path.exists(exp_path):
            retrain_irl(exp_path)
        else:
            rospy.logerr("path %s does not exist", exp_path)
