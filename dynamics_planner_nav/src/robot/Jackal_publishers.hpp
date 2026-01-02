#ifndef DYNAMICS_PLANNER_NAV_ROBOT_VIEW_HPP
#define DYNAMICS_PLANNER_NAV_ROBOT_VIEW_HPP

#include <vector>
#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include "Jackal.hpp"

// Lightweight visualizer for Robot_config-related markers and paths.
class RobotVisualizer {
public:
    static void publishGoals(const ros::Publisher &global_goal_pub,
                             const ros::Publisher &local_goal_pub,
                             const std::vector<double> &global,
                             const std::vector<double> &local);

    static void publishTrajectoryFromState(const ros::Publisher &traj_pub,
                                           const Robot_config::PoseState &state,
                                           std::vector<Robot_config::PoseState> &trajectories,
                                           int nr_steps_, double theta_offset,
                                           const std::vector<double> &t);

    static void publishTrajectory(const ros::Publisher &traj_pub,
                                  std::vector<Robot_config::PoseState> &trajectories,
                                  int nr_steps_,
                                  const std::vector<double> &t);

    static void publishSmoothedGlobalPath(const ros::Publisher &smoothed_path_pub,
                                          const std::vector<double> &xhat,
                                          const std::vector<double> &yhat);
};

#endif // DYNAMICS_PLANNER_NAV_ROBOT_VIEW_HPP
