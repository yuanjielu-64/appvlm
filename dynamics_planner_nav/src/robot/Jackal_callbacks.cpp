// Jackal_callbacks.cpp — ROS callback implementations

#include "Jackal_callbacks.hpp"
#include "Jackal.hpp"
#include "Utility.hpp"
#include <cmath>
#include <algorithm>

JackalCallbacks::JackalCallbacks(Robot_config* robot) : robot_(robot) {}

//==============================================================================
// ODOMETRY CALLBACK
//==============================================================================

void JackalCallbacks::robotStatusCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    double q1 = msg->pose.pose.orientation.x;
    double q2 = msg->pose.pose.orientation.y;
    double q3 = msg->pose.pose.orientation.z;
    double q0 = msg->pose.pose.orientation.w;

    robot_->robot_state.x_ = msg->pose.pose.position.x;
    robot_->robot_state.y_ = msg->pose.pose.position.y;
    robot_->robot_state.theta_ = atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3));
    robot_->robot_state.velocity_ = msg->twist.twist.linear.x;
    robot_->robot_state.angular_velocity_ = msg->twist.twist.angular.z;

    robot_->robot_state.valid_ = true;
}

//==============================================================================
// LASER SCAN CALLBACK
//==============================================================================

void JackalCallbacks::laserScanCallback(const sensor_msgs::LaserScan::ConstPtr& msg) {
    robot_->laserData.clear();
    robot_->laserDataDistance.clear();

    const auto& ranges = msg->ranges;
    const int n = static_cast<int>(ranges.size());
    robot_->laserData.reserve(n);
    robot_->laserDataDistance.reserve(n);

    const double angle_min = msg->angle_min;
    const double inc = msg->angle_increment;
    const double rmin = msg->range_min;
    const double rmax = msg->range_max;

    // Precompute front sector index window for [-pi/4, pi/4]
    int start_idx = static_cast<int>(std::ceil((-M_PI / 4.0 - angle_min) / inc));
    int end_idx = static_cast<int>(std::floor((M_PI / 4.0 - angle_min) / inc));
    start_idx = std::max(0, std::min(n - 1, start_idx));
    end_idx = std::max(0, std::min(n - 1, end_idx));
    const bool has_front = start_idx <= end_idx;

    // Iterative cos/sin update
    double c = std::cos(angle_min);
    double s = std::sin(angle_min);
    const double c_inc = std::cos(inc);
    const double s_inc = std::sin(inc);

    double last_x = std::numeric_limits<double>::infinity();
    double last_y = std::numeric_limits<double>::infinity();

    robot_->front_obs = std::numeric_limits<double>::infinity();

    for (int i = 0; i < n; ++i) {
        const double r = ranges[i];
        if (r > rmin && r < rmax && std::isfinite(r)) {
            const double x = r * c;
            const double y = r * s;

            if (std::isfinite(last_x)) {
                const double dx = x - last_x;
                const double dy = y - last_y;
                if (dx * dx + dy * dy < 1e-4) { // (0.01 m)^2
                    const double c_new = c * c_inc - s * s_inc;
                    const double s_new = s * c_inc + c * s_inc;
                    c = c_new;
                    s = s_new;
                    continue;
                }
            }

            robot_->laserData.emplace_back(static_cast<float>(x), static_cast<float>(y));
            robot_->laserDataDistance.emplace_back(r);
            last_x = x;
            last_y = y;

            if (has_front && i >= start_idx && i <= end_idx) {
                if (r < robot_->front_obs) robot_->front_obs = r;
            }
        }

        const double c_new = c * c_inc - s * s_inc;
        const double s_new = s * c_inc + c * s_inc;
        c = c_new;
        s = s_new;
    }

    if (std::isfinite(robot_->front_obs)) {
        robot_->front_obs = std::max(0.0, robot_->front_obs - 0.33);
    }
}

//==============================================================================
// COSTMAP CALLBACK
//==============================================================================

void JackalCallbacks::costmapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg) {
    robot_->costmapData.clear();

    if (robot_->getRobotState() == Robot_config::LOW_SPEED_PLANNING ||
        robot_->getRobotState() == Robot_config::NORMAL_PLANNING) {
        return;
    }

    const int width = msg->info.width;
    const int height = msg->info.height;
    const double resolution = msg->info.resolution;
    const geometry_msgs::Pose origin = msg->info.origin;
    const Robot_config::PoseState& robotPose = robot_->getPoseState();

    robot_->latter_obs = INFINITY;

    if (robotPose.valid_) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = x + y * width;
                int8_t value = msg->data[index];

                if (value >= 0 && value != 0) {
                    double obs_x = origin.position.x + x * resolution;
                    double obs_y = origin.position.y + y * resolution;

                    std::vector<double> lg = transform_lg(
                        obs_x, obs_y,
                        robot_->robot_state.x_, robot_->robot_state.y_, robot_->robot_state.theta_);

                    robot_->costmapData.push_back(lg);

                    double dx = obs_x - robotPose.x_;
                    double dy = obs_y - robotPose.y_;
                    double distance = std::sqrt(dx * dx + dy * dy);

                    double angle = std::atan2(dy, dx) - robotPose.theta_;
                    angle = normalize_angle(angle);

                    if (angle >= M_PI - M_PI_4 && angle <= M_PI + M_PI_4)
                        robot_->latter_obs = std::min(robot_->latter_obs, distance);
                }
            }
        }
    }
}

//==============================================================================
// GOAL CALLBACK
//==============================================================================

void JackalCallbacks::goalCallback(const move_base_msgs::MoveBaseActionGoal::ConstPtr& msg) {
    ROS_INFO("Received goal to move to position x: %f, y: %f",
             msg->goal.target_pose.pose.position.x,
             msg->goal.target_pose.pose.position.y);

    robot_->global_goal_odom.clear();
    robot_->global_goal_odom = {msg->goal.target_pose.pose.position.x,
                                 msg->goal.target_pose.pose.position.y};
    robot_->setRobotState(Robot_config::NORMAL_PLANNING);
    robot_->can_move = true;
}

//==============================================================================
// ARRAY CALLBACK (Dynamic Time Interval)
//==============================================================================

void JackalCallbacks::arrayCallback(const std_msgs::Float64MultiArray::ConstPtr& msg) {
    if (msg->data.empty()) {
        ROS_WARN("Received empty dynamics data");
        return;
    }

    robot_->timeInterval.clear();
    robot_->timeInterval = msg->data;
}

//==============================================================================
// PARAMS CALLBACK (Tuning Parameters)
//==============================================================================

void JackalCallbacks::paramsCallback(const std_msgs::Float64MultiArray::ConstPtr& msg) {
    if (robot_->getAlgorithm() == Robot_config::DWA) {
        if (msg->data.empty()) {
            ROS_WARN("Received empty dynamics data");
            return;
        }

        robot_->max_vel_x = msg->data[0];
        robot_->max_vel_theta = msg->data[1];
        robot_->vx_sample = msg->data[2];
        robot_->vTheta_samples = msg->data[3];
        robot_->path_distance_bias = msg->data[4];
        robot_->goal_distance_bias = msg->data[5];
    }

    if (robot_->getAlgorithm() == Robot_config::MPPI ||
        robot_->getAlgorithm() == Robot_config::MPPI_DDP) {
        if (msg->data.empty()) {
            ROS_WARN("Received empty dynamics data");
            return;
        }

        robot_->max_vel_x = msg->data[0];
        robot_->max_vel_theta = msg->data[1];
        robot_->nr_pairs_ = msg->data[2];
        robot_->nr_steps_ = msg->data[3];
        robot_->linear_stddev = msg->data[4];
        robot_->angular_stddev = msg->data[5];
        robot_->lambda = msg->data[6];
    }

    if (robot_->getAlgorithm() == Robot_config::DDP) {
        if (msg->data.empty()) {
            ROS_WARN("Received empty dynamics data");
            return;
        }

        robot_->max_vel_x = msg->data[0];
        robot_->max_vel_theta = msg->data[1];
        robot_->nr_pairs_ = msg->data[2];
        robot_->distance = msg->data[3];
        robot_->robot_radius_ = msg->data[4];
    }

    robot_->param_received = true;
}

//==============================================================================
// GLOBAL PATH CALLBACK
//==============================================================================

void JackalCallbacks::globalPathCallback(const nav_msgs::Path::ConstPtr& msg) {
    // Start to find local and global goal
    if (robot_->global_goal_odom.empty()) {
        return;
    }

    robot_->getGoal = true;

    robot_->local_paths.clear();
    robot_->local_paths_odom.clear();
    std::vector<double> goals;
    goals.reserve(2);

    if ((int)msg->poses.size() == 0) {
        if (robot_->local_goals_history.size() >= 2) {
            std::vector<double> lg = transform_lg(
                robot_->local_goals_history[0][0], robot_->local_goals_history[0][1],
                robot_->robot_state.x_, robot_->robot_state.y_, robot_->robot_state.theta_);

            robot_->setLocalGoal(lg, robot_->local_goals_history[0][0], robot_->local_goals_history[0][1]);

            robot_->local_paths_history.erase(robot_->local_paths_history.begin());
            robot_->local_goals_history.erase(robot_->local_goals_history.begin());

            robot_->view_Goal(robot_->global_goal_odom, robot_->local_goal_odom);

            return;
        }

        if (!robot_->local_paths_history.empty()) {
            // We have the last goal here
            goals = {robot_->global_goal_odom[0], robot_->global_goal_odom[1]};

            std::vector<double> lg;
            std::vector<double> X, Y;

            int close_id = -1;
            double min_distance = INFINITY;
            double threads = 1;
            for (size_t i = 0; i < robot_->local_paths_history[0].size(); ++i) {
                lg = transform_lg(robot_->local_paths_history[0][i][0], robot_->local_paths_history[0][i][1],
                                 robot_->robot_state.x_, robot_->robot_state.y_, robot_->robot_state.theta_);
                double distance = sqrt(lg[0] * lg[0] + lg[1] * lg[1]);
                if (distance < min_distance) {
                    min_distance = distance;
                    close_id = (int)i;
                }
                X.push_back(lg[0]);
                Y.push_back(lg[1]);
            }

            std::vector<std::vector<double>> paths = {{robot_->robot_state.x_, robot_->robot_state.y_}};
            for (size_t i = close_id; i < X.size(); ++i) {
                std::vector<double> vector = {robot_->local_paths_history[0][i][0], robot_->local_paths_history[0][i][1]};
                paths.push_back(vector);
            }

            robot_->local_paths_odom = paths;

            bool flag = false;
            double length = l2_distance(X[close_id], Y[close_id], 0, 0);

            for (size_t i = close_id; i < X.size(); ++i) {
                double dist = l2_distance(X[i], Y[i], X[i - 1], Y[i - 1]);

                length += dist;

                if (length >= std::max(threads - 0.08 * robot_->re, 0.2) && flag == false) {
                    lg = {X[i], Y[i]};
                    // Safety check to prevent out-of-bounds access
                    if (i < robot_->local_paths_history[0].size()) {
                        robot_->setLocalGoal(lg, robot_->local_paths_history[0][i][0], robot_->local_paths_history[0][i][1]);
                    } else {
                        ROS_WARN("Index %zu out of bounds for local_paths_history[0] (size: %zu)",
                                i, robot_->local_paths_history[0].size());
                    }
                    flag = true;
                    break;
                }
            }

            if (!flag) {
                lg = transform_lg(robot_->global_goal_odom[0], robot_->global_goal_odom[1],
                                 robot_->robot_state.x_, robot_->robot_state.y_, robot_->robot_state.theta_);
                robot_->setLocalGoal(lg, robot_->global_goal_odom[0], robot_->global_goal_odom[1]);
            }

            robot_->view_Goal(goals, robot_->local_goal_odom);
            robot_->update_angular_velocity();
            return;
        }
    }

    std::vector<std::pair<double, double>> path_points;

    std::vector<double> X, Y;
    for (const auto& pose : msg->poses) {
        X.push_back(pose.pose.position.x);
        Y.push_back(pose.pose.position.y);
    }

    std::vector<double> xhat = savgolFilter(X, 9, 2);
    std::vector<double> yhat = savgolFilter(Y, 9, 2);

    std::vector<double> lg = transform_lg(robot_->global_goal_odom[0],
                                         robot_->global_goal_odom[1],
                                         robot_->robot_state.x_,
                                         robot_->robot_state.y_,
                                         robot_->robot_state.theta_);

    robot_->global_goal = lg;
    goals = {robot_->global_goal_odom[0], robot_->global_goal_odom[1]};

    std::vector<double> last_point = {INFINITY, INFINITY};

    bool flag = false;
    double thresholdSq = 0;

    double length = 0;

    for (size_t i = 1; i < xhat.size(); ++i) {
        double dist = l2_distance(xhat[i], yhat[i], xhat[i - 1], yhat[i - 1]);

        length += dist;

        if (robot_->getAlgorithm() == Robot_config::DWA || robot_->getAlgorithm() == Robot_config::DWA_DDP) {
            thresholdSq = 2 * robot_->max_vel_x + 1;

            if (length >= thresholdSq && flag == false) {
                lg = transform_lg(xhat[i], yhat[i], robot_->robot_state.x_, robot_->robot_state.y_, robot_->robot_state.theta_);
                robot_->setLocalGoal(lg, xhat[i], yhat[i]);
                flag = true;
                break;
            }
        } else if (robot_->getAlgorithm() == Robot_config::MPPI || robot_->getAlgorithm() == Robot_config::MPPI_DDP) {
            thresholdSq = 1.5 * robot_->max_vel_x;

            if (length >= thresholdSq && flag == false) {
                lg = transform_lg(xhat[i], yhat[i], robot_->robot_state.x_, robot_->robot_state.y_, robot_->robot_state.theta_);
                robot_->setLocalGoal(lg, xhat[i], yhat[i]);
                flag = true;
                break;
            }
        } else {
            if (robot_->getRobotState() == Robot_config::NORMAL_PLANNING) {
                thresholdSq = 2 * robot_->max_vel_x + 2;
                if (length >= thresholdSq && flag == false) {
                    lg = transform_lg(xhat[i], yhat[i], robot_->robot_state.x_, robot_->robot_state.y_, robot_->robot_state.theta_);
                    robot_->setLocalGoal(lg, xhat[i], yhat[i]);
                    flag = true;
                    break;
                }
            } else if (robot_->getRobotState() == Robot_config::LOW_SPEED_PLANNING) {
                thresholdSq = 1 * robot_->max_vel_x + 1;

                if (length >= thresholdSq && flag == false) {
                    lg = transform_lg(xhat[i], yhat[i], robot_->robot_state.x_, robot_->robot_state.y_, robot_->robot_state.theta_);
                    robot_->setLocalGoal(lg, xhat[i], yhat[i]);
                    flag = true;
                    break;
                }
            } else if (robot_->getRobotState() == Robot_config::NO_MAP_PLANNING) {
                thresholdSq = robot_->max_vel_x;
                if (length >= thresholdSq && flag == false) {
                    lg = transform_lg(xhat[i], yhat[i], robot_->robot_state.x_, robot_->robot_state.y_, robot_->robot_state.theta_);
                    robot_->setLocalGoal(lg, xhat[i], yhat[i]);
                    flag = true;
                    break;
                }
            } else {
                thresholdSq = 0.8;
                if (length >= thresholdSq && flag == false) {
                    lg = transform_lg(xhat[i], yhat[i], robot_->robot_state.x_, robot_->robot_state.y_, robot_->robot_state.theta_);
                    robot_->setLocalGoal(lg, xhat[i], yhat[i]);
                    flag = true;
                    break;
                }
            }
        }
    }

    if (!flag) {
        lg = transform_lg(robot_->global_goal_odom[0], robot_->global_goal_odom[1],
                         robot_->robot_state.x_, robot_->robot_state.y_, robot_->robot_state.theta_);
        robot_->setLocalGoal(lg, robot_->global_goal_odom[0], robot_->global_goal_odom[1]);
    }

    for (size_t i = 1; i < xhat.size(); ++i) {
        if (last_point[0] != INFINITY) {
            double dx = xhat[i] - last_point[0];
            double dy = yhat[i] - last_point[1];
            double distance = sqrt(dx * dx + dy * dy);

            if (distance >= 0.1) {
                lg = transform_lg(xhat[i], yhat[i], robot_->robot_state.x_, robot_->robot_state.y_, robot_->robot_state.theta_);
                robot_->local_paths.emplace_back(std::vector<double>{lg[0], lg[1]});
                robot_->local_paths_odom.emplace_back(std::vector<double>{xhat[i], yhat[i]});
                last_point = {xhat[i], yhat[i]};
            }
        } else {
            last_point = {xhat[i], yhat[i]};
        }
    }

    if (!robot_->local_paths.empty() && !robot_->local_goal_odom.empty()) {
        robot_->local_paths_history.push_back(robot_->local_paths_odom);
        robot_->local_goals_history.push_back(robot_->local_goal_odom);

        if (robot_->local_paths_history.size() > 30) {
            robot_->local_paths_history.erase(robot_->local_paths_history.begin());
        }

        if (robot_->local_goals_history.size() > 30) {
            robot_->local_goals_history.erase(robot_->local_goals_history.begin());
        }
    }

    robot_->view_Goal(goals, robot_->local_goal_odom);
    robot_->update_angular_velocity();
}

//==============================================================================
// VELOCITY CALLBACK
//==============================================================================

void JackalCallbacks::velocityCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    if (robot_->getAlgorithm() == Robot_config::DWA ||
        robot_->getAlgorithm() == Robot_config::DWA_DDP ||
        robot_->getAlgorithm() == Robot_config::MPPI ||
        robot_->getAlgorithm() == Robot_config::MPPI_DDP)
        return;

    double linear_speed = fabs(msg->twist.twist.linear.x);

    double LOW_SPEED_THRESHOLD = robot_->max_vel_x * 0.8 + 0.05;
    double LOW_SPEED_HYSTERESIS = 0.05;
    double HIGH_SPEED_THRESHOLD = robot_->max_vel_x * 0.5 + 0.1;
    double BRAKE_WAIT_TIME = 0.5;

    if (robot_->getRobotState() == Robot_config::NORMAL_PLANNING) {
        robot_->re = 1;
        robot_->low_to_normal_active = false;
        robot_->low_to_brake_active = false;

        robot_->is_stopped = false;

        if (linear_speed < HIGH_SPEED_THRESHOLD) {
            if (!robot_->normal_to_low_active) {
                robot_->normal_to_low_time = ros::Time::now();
                robot_->normal_to_low_active = true;
            } else if ((ros::Time::now() - robot_->normal_to_low_time).toSec() >= 0.5) {
                ROS_INFO("The robot is back to LOW_SPEED_PLANNING after 0.5s in high speed.");
                robot_->setRobotState(Robot_config::LOW_SPEED_PLANNING);

                robot_->normal_to_low_active = false;
            }
        } else {
            robot_->normal_to_low_active = false;
        }
    } else if (robot_->getRobotState() == Robot_config::LOW_SPEED_PLANNING) {

        robot_->normal_to_low_active = false;

        if (linear_speed >= LOW_SPEED_THRESHOLD + LOW_SPEED_HYSTERESIS) {
            if (!robot_->low_to_normal_active) {
                robot_->low_to_normal_time = ros::Time::now();
                robot_->low_to_normal_active = true;
            } else if ((ros::Time::now() - robot_->low_to_normal_time).toSec() >= 0.5) {
                ROS_INFO("The robot is back to NORMAL_PLANNING after 0.5s in low speed.");
                robot_->setRobotState(Robot_config::NORMAL_PLANNING);
                robot_->low_to_normal_active = false;
            }
        } else {
            robot_->low_to_normal_active = false;
        }

        if (linear_speed < Robot_config::MIN_SPEED) {
            if (!robot_->low_to_brake_active) {
                robot_->low_to_brake_time = ros::Time::now();
                robot_->low_to_brake_active = true;
            } else if ((ros::Time::now() - robot_->low_to_brake_time).toSec() > BRAKE_WAIT_TIME * Robot_config::STOPPED_TIME_THRESHOLD) {
                ROS_INFO("The robot needs to brake after 1 second in low speed");
                robot_->setRobotState(Robot_config::BRAKE_PLANNING);
                robot_->low_to_brake_active = false;
            }
        } else {
            robot_->low_to_brake_active = false;
        }

    } else {  // recover
        robot_->normal_to_low_active = false;
        robot_->low_to_normal_active = false;
        robot_->low_to_brake_active = false;

        if (robot_->re >= 5)
            robot_->re = 4;
    }
}

