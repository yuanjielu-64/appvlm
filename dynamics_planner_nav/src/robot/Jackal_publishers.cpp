#include "Jackal_publishers.hpp"
#include "Utility.hpp"

void RobotVisualizer::publishGoals(const ros::Publisher &global_goal_pub,
                                   const ros::Publisher &local_goal_pub,
                                   const std::vector<double> &goal,
                                   const std::vector<double> &goal1) {
    visualization_msgs::Marker marker;

    marker.header.frame_id = "odom";
    marker.header.stamp = ros::Time::now();

    marker.ns = "point_marker";
    marker.id = 0;

    marker.type = visualization_msgs::Marker::POINTS;
    marker.action = visualization_msgs::Marker::ADD;

    marker.scale.x = 0.2;
    marker.scale.y = 0.2;

    marker.color.r = 0.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1.0;

    geometry_msgs::Point p;
    p.x = goal[0];
    p.y = goal[1];
    p.z = 0.0;

    marker.points.push_back(p);

    global_goal_pub.publish(marker);

    marker.points.clear();
    marker.id = 1;
    marker.color.r = 1.0f;
    marker.color.g = 0.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1.0;
    geometry_msgs::Point p1;
    p1.x = goal1[0];
    p1.y = goal1[1];
    p1.z = 0.0;

    marker.points.push_back(p1);

    local_goal_pub.publish(marker);
}

void RobotVisualizer::publishTrajectoryFromState(const ros::Publisher &traj_pub,
                                                 const Robot_config::PoseState &state,
                                                 std::vector<Robot_config::PoseState> &trajectories,
                                                 int nr_steps_, double theta_offset,
                                                 const std::vector<double> &t) {
    nav_msgs::Path path;
    path.header.stamp = ros::Time::now();
    path.header.frame_id = "odom";

    double x = state.x_;
    double y = state.y_;
    double theta = normalize_angle(state.theta_ + theta_offset);

    for (int i = 0; i < nr_steps_; ++i) {
        x = x + trajectories[i].velocity_ * std::cos(theta) * t[i];
        y = y + trajectories[i].velocity_ * std::sin(theta) * t[i];
        theta = normalize_angle(theta + trajectories[i].angular_velocity_ * t[i]);

        geometry_msgs::PoseStamped pose;
        pose.header.stamp = ros::Time::now();
        pose.header.frame_id = "odom";
        pose.pose.position.x = x;
        pose.pose.position.y = y;
        pose.pose.position.z = 0;
        path.poses.push_back(pose);
    }

    traj_pub.publish(path);
}

void RobotVisualizer::publishTrajectory(const ros::Publisher &traj_pub,
                                        std::vector<Robot_config::PoseState> &trajectories,
                                        int nr_steps_,
                                        const std::vector<double> &t) {
    nav_msgs::Path path;
    path.header.stamp = ros::Time::now();
    path.header.frame_id = "odom";

    for (int i = 0; i < nr_steps_; ++i) {
        double x = trajectories[i].x_;
        double y = trajectories[i].y_;
        double theta = normalize_angle(trajectories[i].angular_velocity_);

        geometry_msgs::PoseStamped pose;

        pose.header.stamp = ros::Time::now();
        pose.header.frame_id = "odom";
        pose.pose.position.x = x;
        pose.pose.position.y = y;
        pose.pose.position.z = 0;

        path.poses.push_back(pose);
    }

    traj_pub.publish(path);
}

void Robot_config::publishRobotState() const {
    std_msgs::String state_msg;

    switch(currentState) {
        case INITIALIZING: state_msg.data = "INITIALIZING"; break;
        case NORMAL_PLANNING: state_msg.data = "NORMAL_PLANNING"; break;
        case LOW_SPEED_PLANNING: state_msg.data = "LOW_SPEED_PLANNING"; break;
        case NO_MAP_PLANNING: state_msg.data = "NO_MAP_PLANNING"; break;
        case BRAKE_PLANNING: state_msg.data = "BRAKE_PLANNING"; break;
        case RECOVERY: state_msg.data = "RECOVERY"; break;
        case ROTATE_PLANNING: state_msg.data = "ROTATE_PLANNING"; break;
        case BACKWARD: state_msg.data = "BACKWARD"; break;
        case FORWARD: state_msg.data = "FORWARD"; break;
        case TEST: state_msg.data = "TEST"; break;
        case IDLE: state_msg.data = "IDLE"; break;
        default: state_msg.data = "UNKNOWN"; break;
    }

    robot_state_pub.publish(state_msg);
}

void Robot_config::view_Goal(std::vector<double> &goal, std::vector<double> &goal1) const {
    RobotVisualizer::publishGoals(global_goal_pub, local_goal_pub, goal, goal1);
}

void Robot_config::viewTrajectories(std::vector<PoseState> &trajectories, int nr_steps_, double theta_,
                                    std::vector<double> &t) const {
    RobotVisualizer::publishTrajectoryFromState(trajectory_pub, robot_state, trajectories, nr_steps_, theta_, t);
}

void Robot_config::viewTrajectories(std::vector<PoseState> &trajectories, int nr_steps_, std::vector<double> &t) const {
    RobotVisualizer::publishTrajectory(trajectory_pub, trajectories, nr_steps_, t);
}

