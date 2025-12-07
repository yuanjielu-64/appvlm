// Jackal.cpp — Robot state manager and ROS interface
#include "Jackal.hpp"
#include <ros/ros.h>
#include <cmath>
#include <algorithm>
#include <nav_msgs/GetPlan.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Twist.h>
#include "std_srvs/Empty.h"
#include "Utility.hpp"

bool Robot_config::setup() {
    if (!checkGazeboPaused() && getRobotState() != INITIALIZING && getPoseState().valid_ && getMapData() && can_move &&
        getGoal) {
        return true;
    }

    setRobotState(RobotState::IDLE);
    return false;
}

bool Robot_config::checkGazeboPaused() const {
    std_msgs::String state_msg;

    bool is_paused = false;
    if (nh.getParam("/gazebo/is_paused", is_paused)) {
        if (is_paused) {
            state_msg.data = "PAUSED";
            robot_state_pub.publish(state_msg);
            return true;
        }
    }

    publishRobotState();
    return false;
}

// Load map data from specified source and verify availability
bool Robot_config::getMapData() {
    const RobotState state = getRobotState();
    MapSource mapSource;
    map.clear();

    if (state == BACKWARD)
        mapSource = Robot_config::ONLY_COSTMAP_RECEIVED;
    else
        mapSource = Robot_config::ONLY_LASER_RECEIVED;

    // Determine primary and fallback data sources
    const auto &primaryData = (mapSource == ONLY_COSTMAP_RECEIVED) ? costmapData : getLaserData();
    const auto &fallbackData = (mapSource == ONLY_COSTMAP_RECEIVED) ? getLaserData() : costmapData;

    // Use primary source if available, otherwise fallback
    if (!primaryData.empty()) {
        map = primaryData;
        currentMap = mapSource;
    } else if (!fallbackData.empty()) {
        map = fallbackData;
        currentMap = (mapSource == ONLY_COSTMAP_RECEIVED) ? ONLY_LASER_RECEIVED : ONLY_COSTMAP_RECEIVED;
    } else {
        // No data available from either source
        currentMap = NO_ANY_RECEIVED;
        setRobotState(NO_MAP_PLANNING);
        return true;
    }

    return !map.empty();
}


// Get robot footprint (physical dimensions for collision checking)
Robot_config::Footprint Robot_config::getFootprint() const {
    const RobotState state = getRobotState();

    // Use point-mass approximation for backward maneuvers (tight spaces)
    if (state == BACKWARD) {
        return {POINT_MASS_LENGTH, POINT_MASS_WIDTH};
    }

    // Use point-mass for non-laser maps (costmap-only or no map - less reliable)
    if (currentMap != ONLY_LASER_RECEIVED) {
        return {POINT_MASS_LENGTH, POINT_MASS_WIDTH};
    }

    // Default: use full robot dimensions
    return {ROBOT_LENGTH, ROBOT_WIDTH};
}

// Get velocity constraints (speed limits for trajectory generation)
Robot_config::VelocityLimits Robot_config::getVelocityLimits() const {
    const RobotState state = getRobotState();

    // Backward maneuvers: only allow negative velocities
    if (state == BACKWARD) {
        return {-2.0, 0.0, -2.0, 2.0};
    }

    if (state == FORWARD) {
        return {0.0, 2.0, -2.0, 2.0};
    }

    return {0.0, max_vel_x, -max_vel_theta, max_vel_theta};
}

std::vector<std::vector<double> > Robot_config::getLaserData() {
    std::vector<std::vector<double> > out;
    out.reserve(laserData.size());
    for (const auto &p: laserData) {
        out.push_back({static_cast<double>(p.x()), static_cast<double>(p.y())});
    }
    return out;
}


//==============================================================================
// Constructor: Initialize state and setup ROS communication
//==============================================================================
Robot_config::Robot_config()
    : algorithm(DWA),
      currentState(INITIALIZING),
      currentMap(ONLY_LASER_RECEIVED),
      getGoal(false),
      can_move(false),
      param_received(false),
      canBeSolved(true),
      rotating_angle(0.0),
      dt(0.05),
      latter_obs(INFINITY),
      front_obs(INFINITY),
      recover_times(0) {
    // Reserve capacity for goal vectors
    global_goal.reserve(2);
    local_goal.reserve(2);
    local_goal_odom.reserve(2);

    // Initialize state
    local_goal = {0.0, 0.0};
    robot_state = PoseState(0.0, 0.0, 0.0, 0.0, 0.0, false);
    actions = {{0.0, 0.0}};

    // ---- Create Callback Handler ----
    callbacks_ = std::make_shared<JackalCallbacks>(this);

    // ---- Create Async Task Executor (for heavy callbacks) ----
    async_executor_ = std::make_shared<AsyncTaskExecutor>(num_threads);

    // ---- ROS Subscribers ----
    robot_pose_sub = nh.subscribe("/odometry/filtered", 10, &Robot_config::robotStatusCallback, this);
    laser_scan_sub = nh.subscribe("/front/scan", 10, &Robot_config::laserScanCallback, this);
    goal_sub = nh.subscribe("/move_base/goal", 10, &Robot_config::goalCallback, this);
    costmap_update_sub = nh.subscribe("/move_base/local_costmap/costmap", 10, &Robot_config::costmapCallback, this);
    velocity_sub = nh.subscribe("/odometry/filtered", 10, &Robot_config::velocityCallback, this);
    global_path_sub = nh.subscribe<nav_msgs::Path>("/move_base/NavfnROS/plan", 10, &Robot_config::globalPathCallback,
                                                   this);
    array_dt_sub = nh.subscribe("/dy_dt", 1, &Robot_config::arrayCallback, this);
    params_sub = nh.subscribe("/params", 1, &Robot_config::paramsCallback, this);

    // ---- ROS Publish
    trajectory_pub = nh.advertise<nav_msgs::Path>("trajectory", 10);
    global_path_pub = nh.advertise<nav_msgs::Path>("global_path", 10);
    local_goal_pub = nh.advertise<visualization_msgs::Marker>("local_goal", 1);
    global_goal_pub = nh.advertise<visualization_msgs::Marker>("global_goal", 1);
    cmd_vel_pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
    robot_state_pub = nh.advertise<std_msgs::String>("/robot_mode", 1);

    // ---- ROS Service Clients ----
    global_path_clt = nh.serviceClient<nav_msgs::GetPlan>("/move_base/NavfnROS/make_plan");
    clear_costmaps_clt = nh.serviceClient<std_srvs::Empty>("/move_base/clear_costmaps");

    ROS_INFO("Robot_config initialized successfully");
    ROS_INFO("All planners will use %d parallel threads", num_threads);
}


double Robot_config::calculateTheta(const PoseState &state, const std::vector<double> &y) {
    const double deltaX = y[0] - state.x_;
    const double deltaY = y[1] - state.y_;
    const double theta = std::atan2(deltaY, deltaX);
    const double normalizedTheta = normalize_angle(state.theta_);
    return std::fabs(normalize_angle(theta - normalizedTheta));
}

//==============================================================================
// Get current tuning parameters snapshot
//==============================================================================
Robot_config::TuningParams Robot_config::setTuningParams() const {
    TuningParams params{};
    params.max_vel_x = max_vel_x;
    params.max_vel_y = max_vel_y;
    params.acc_lim_theta = max_vel_theta;
    params.vx_sample = static_cast<int>(vx_sample);
    params.vTheta_samples = static_cast<int>(vTheta_samples);
    params.path_distance_bias = path_distance_bias;
    params.goal_distance_bias = goal_distance_bias;
    params.nr_pairs_ = static_cast<int>(nr_pairs_);
    params.nr_steps_ = static_cast<int>(nr_steps_);
    params.linear_stddev = linear_stddev;
    params.angular_stddev = angular_stddev;
    params.lambda = lambda;
    params.local_goal_distance = local_goal_distance;
    params.distance = distance;
    params.robot_radius_ = robot_radius_;
    params.dt = dt;
    return params;
}


void Robot_config::update_angular_velocity() {
    // Optional post-processing for angular velocity limits
}
