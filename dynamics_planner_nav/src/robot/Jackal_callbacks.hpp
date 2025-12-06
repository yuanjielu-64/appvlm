// Jackal_callbacks.hpp — ROS callback handlers
// Manages all ROS subscriber callbacks for Robot_config

#ifndef DYNAMICS_PLANNER_NAV_JACKAL_CALLBACKS_HPP
#define DYNAMICS_PLANNER_NAV_JACKAL_CALLBACKS_HPP

#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/OccupancyGrid.h>
#include <move_base_msgs/MoveBaseActionGoal.h>
#include <std_msgs/Float64MultiArray.h>

// Forward declaration
class Robot_config;

class JackalCallbacks {
public:
    explicit JackalCallbacks(Robot_config* robot);

    ~JackalCallbacks() = default;

    // ROS callback handlers
    void robotStatusCallback(const nav_msgs::Odometry::ConstPtr& msg);

    void laserScanCallback(const sensor_msgs::LaserScan::ConstPtr& msg);

    void costmapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg);

    void globalPathCallback(const nav_msgs::Path::ConstPtr& msg);

    void arrayCallback(const std_msgs::Float64MultiArray::ConstPtr& msg);

    void paramsCallback(const std_msgs::Float64MultiArray::ConstPtr& msg);

    void goalCallback(const move_base_msgs::MoveBaseActionGoal::ConstPtr& msg);

    void velocityCallback(const nav_msgs::Odometry::ConstPtr& msg);

private:
    Robot_config* robot_;  // Pointer to parent robot instance
};

#endif // DYNAMICS_PLANNER_NAV_JACKAL_CALLBACKS_HPP
