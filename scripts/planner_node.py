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

import actionlib
import rospy
import numpy as np
import os
import pickle
import sys

from mdp import GridMDP
from reward_function import RewardFunction
from nav_map import NavMap

from move_base_msgs.msg import MoveBaseAction
from nav_msgs.msg import OccupancyGrid, Path
from nav_msgs.srv import GetPlan, GetPlanResponse
from visualization_msgs.msg import MarkerArray

from reward_components.step_reward import StepReward
from reward_components.social_reward import SocialReward
from reward_components.obstacle_reward import ObstacleReward
from reward_components.path_deviation_reward import PathDeviationReward

from reward_components.reward_shapes.gaussian_reward import GaussianReward
from reward_components.reward_shapes.quadratic_reward import QuadraticReward


def load_reward_params(parameter_file):
    params = pickle.load(open(parameter_file))

    # unpack parameters in order
    step_weight = params[0].detach().numpy()[0]
    gaussian_scale = params[1].detach().numpy()[0]
    gaussian_width = params[2].detach().numpy()[0]
    path_dev_weight = params[3].detach().numpy()[0]

    return step_weight, gaussian_scale, gaussian_width, path_dev_weight


def setup_reward_function(grid_actions, step_weight, gaussian_scale,
                          gaussian_width, path_dev_weight):
    """
    Setup reward function with experiment reward components.
    :param grid_actions: MDP actions
    :param step_weight: step reward weight
    :param gaussian_scale: social reward gaussian scale
    :param gaussian_width: social reward gaussian width
    :param path_dev_weight: path deviation reward weight
    :return: reward function object
    """
    reward_function = RewardFunction()
    reward_function.add_component(StepReward(step_weight, [np.float("-inf"), 0]))
    reward_function.add_component(SocialReward(
        GaussianReward(gaussian_scale, gaussian_width, [np.float("-inf"), 0], [0, np.float("inf")])))
    reward_function.add_component(PathDeviationReward(
        QuadraticReward(path_dev_weight, [np.float("-inf"), 0]), grid_actions))
    reward_function.add_component(ObstacleReward())

    return reward_function


def setup_mdp(grid_actions, reward_function):
    """
    Setup MDP with diagonal grid actions.
    :param grid_actions: MDP actions
    :param reward_function: mdp reward function
    :return: MDP object
    """

    grid_mdp = GridMDP(grid_actions, reward_function)

    return grid_mdp


class SocialPlanner:

    def __init__(self, parameter_file):
        self.map = None
        self.map_info = None

        grid_actions = GridMDP.DIAGONAL_ACTIONS

        # load reward parameters from file
        step_weight, gaussian_scale, gaussian_width, path_dev_weight = \
            load_reward_params(parameter_file)

        # setup reward function with loaded parameters
        reward_function = setup_reward_function(grid_actions, step_weight,
                                                gaussian_scale, gaussian_width,
                                                path_dev_weight)

        # setup Markov Decision Process (MDP) with reward function
        self.mdp = setup_mdp(grid_actions, reward_function)

        # grid poses of people in the environment
        # TODO people poses from ros parameter or publisher
        self.people_pos = [[52, 26]]
        self.people_pub = rospy.Publisher("people_viz", MarkerArray, queue_size=1,
                                          latch=True)

        # move base link for sending navigation goals
        rospy.loginfo("waiting for move base action server to come up...")
        self.move_base_client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.move_base_client.wait_for_server()

        # link to ROS navigation
        rospy.Service("/redirect_plan", GetPlan, self.make_plan_server)

        # map callback
        rospy.Subscriber("/move_base/global_costmap/costmap", OccupancyGrid,
                         self.map_callback)

    def map_callback(self, static_map):
        rospy.loginfo("static map received")
        self.map = NavMap(static_map)

        # TODO publish people marker

    def make_plan_server(self, req):
        rospy.loginfo("planning requested")
        res = GetPlanResponse()

        if self.map is not None:
            grid_path = self.plan_path(req.start, req.goal)
            path = self.map.convert_to_ros_path(grid_path, req.start.header.stamp)
            res.plan = path
        else:
            rospy.logerr("No map received, cannot plan a path")
            res.plan = Path()

        return res

    def plan_path(self, start, goal):
        # get start and goal in grid coordinates
        start_cell = self.map.world2cell(start.pose.position.x, start.pose.position.y)
        goal_cell = self.map.world2cell(goal.pose.position.x, goal.pose.position.y)

        grid_path = self.mdp.plan_path(self.map, self.people_pos, start_cell, goal_cell,
                                       softmax=True, stochastic_policy=True)

        return grid_path


if __name__ == '__main__':
    rospy.init_node('social_planner_node')
    rospy.loginfo("social planner node started")

    if len(sys.argv) < 2:
        rospy.logwarn("usage: rosrun learning-nav-irl planner_node.py <reward_param_file>")

    else:
        reward_param_file = sys.argv[1]

        if os.path.exists(reward_param_file):
            fp = SocialPlanner(reward_param_file)
        else:
            rospy.logerr("planner node: reward parameter file {} not found".format(reward_param_file))

    rospy.spin()
