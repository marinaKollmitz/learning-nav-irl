cmake_minimum_required(VERSION 2.8.3)
project(learning-nav-irl)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  pluginlib
  nav_core
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(global_planner_link
   src/global_planner_link.cpp
)

target_link_libraries(global_planner_link ${catkin_LIBRARIES})
