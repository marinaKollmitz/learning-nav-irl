#include <pluginlib/class_list_macros.h>
#include <learning-nav-irl/global_planner_link.h>

//register this planner as a BaseGlobalPlanner plugin
PLUGINLIB_EXPORT_CLASS(planner_link::GlobalPlanner, nav_core::BaseGlobalPlanner)

using namespace std;

namespace planner_link
{

//Default Constructor
GlobalPlanner::GlobalPlanner (){

}

GlobalPlanner::GlobalPlanner(std::string name, costmap_2d::Costmap2DROS* costmap_ros)
{
  initialize(name, costmap_ros);
}


void GlobalPlanner::initialize(std::string name, costmap_2d::Costmap2DROS* costmap_ros)
{
  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~/" + name);

  plan_pub_ = private_nh.advertise<nav_msgs::Path>("plan", 1);
  plan_poses_pub_ = private_nh.advertise<geometry_msgs::PoseArray>("plan_poses", 1);

  path_client_ = nh.serviceClient<nav_msgs::GetPlan>("/redirect_plan");
}

bool GlobalPlanner::makePlan(const geometry_msgs::PoseStamped& start, const geometry_msgs::PoseStamped& goal,  std::vector<geometry_msgs::PoseStamped>& plan )
{
  nav_msgs::GetPlan plan_srv;
  geometry_msgs::PoseArray plan_poses;

  plan_srv.request.start = start;
  plan_srv.request.goal = goal;

  if(!path_client_.waitForExistence())
    ROS_WARN("path client does not exists");

  ROS_INFO("global_planner_link: redirecting planning request");
  if(path_client_.call(plan_srv))
  {
    plan = plan_srv.response.plan.poses;

    plan_poses.header = plan.front().header;
    for(int i=0; i<plan.size(); i++)
    {
      plan.at(i).pose.orientation = goal.pose.orientation;
      plan_poses.poses.push_back(plan.at(i).pose);
    }

    plan_pub_.publish(plan_srv.response.plan);
    plan_poses_pub_.publish(plan_poses);

    return true;
  }

  else
  {
    ROS_ERROR("global_planner_link: could not redirect planning request");
    return false;
  }
}

}
