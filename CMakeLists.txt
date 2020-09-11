cmake_minimum_required(VERSION 3.0.2)
project(learning-nav-irl)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

