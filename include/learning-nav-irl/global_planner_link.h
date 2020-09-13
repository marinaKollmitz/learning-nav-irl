/** include the libraries you need in your planner here */
/** for global path planner interface */
#include <ros/ros.h>
#include <costmap_2d/costmap_2d_ros.h>
#include <costmap_2d/costmap_2d.h>
#include <nav_core/base_global_planner.h>
#include <geometry_msgs/PoseStamped.h>
#include <angles/angles.h>
#include <base_local_planner/world_model.h>
#include <base_local_planner/costmap_model.h>
#include <nav_msgs/GetPlan.h>
#include <geometry_msgs/PoseArray.h>

using std::string;

#ifndef GLOBAL_PLANNER_CPP
#define GLOBAL_PLANNER_CPP

namespace planner_link
{

class GlobalPlanner : public nav_core::BaseGlobalPlanner {
  /** Global planner plugin that redirects planning requests from move_base
to python global planner via service call. **/

public:

  GlobalPlanner();
  GlobalPlanner(std::string name, costmap_2d::Costmap2DROS* costmap_ros);

  /** overridden classes from interface nav_core::BaseGlobalPlanner **/
  void initialize(std::string name, costmap_2d::Costmap2DROS* costmap_ros);
  bool makePlan(const geometry_msgs::PoseStamped& start,
                const geometry_msgs::PoseStamped& goal,
                std::vector<geometry_msgs::PoseStamped>& plan
                );

private:
  ros::ServiceClient path_client_;
  ros::Publisher plan_pub_;
  ros::Publisher plan_poses_pub_;
};

}
#endif
