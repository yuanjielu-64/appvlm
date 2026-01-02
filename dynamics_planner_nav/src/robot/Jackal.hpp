// Jackal.hpp — Robot state/IO manager for Jackal navigation
// Manages ROS communication, state caching, and planner interface

#ifndef DYNAMICS_PLANNER_NAV_JACKAL_HPP
#define DYNAMICS_PLANNER_NAV_JACKAL_HPP

// ROS headers
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/OccupancyGrid.h>
#include <move_base_msgs/MoveBaseActionGoal.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/String.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Twist.h>

// Other dependencies
#include <costmap_2d/costmap_2d_ros.h>
#include <Eigen/Core>

// STL
#include <vector>
#include <memory>
#include <mutex>

// Utils
#include "../utils/AsyncTaskExecutor.hpp"

// Forward declarations
class JackalCallbacks;

class Robot_config {
    // Friend classes for modular access
    friend class JackalCallbacks;

public:
    // Robot pose/velocity snapshot
    class PoseState {
    public:
        PoseState()
            : x_(0.0), y_(0.0), theta_(0.0), velocity_(0.0), angular_velocity_(0.0), valid_(false) {}

        PoseState(double x, double y, double theta, double v, double w, bool valid)
            : x_(x), y_(y), theta_(theta), velocity_(v), angular_velocity_(w), valid_(valid) {
        }

        std::vector<double> pose() const { return {x_, y_}; }

        double x_;
        double y_;
        double theta_;
        double velocity_;
        double angular_velocity_;
        bool valid_;
    };

    // Tuning parameters snapshot
    struct TuningParams {
        double max_vel_x;
        double max_vel_y;
        double max_vel_theta;  // maximum allowed angular velocity
        int vx_sample;
        int vTheta_samples;
        double path_distance_bias;
        double goal_distance_bias;
        int nr_pairs_;
        int nr_steps_;
        double linear_stddev;
        double angular_stddev;
        double lambda;
        double local_goal_distance;
        double distance;
        double robot_radius_;
        double dt;
    };

    // Robot footprint (physical dimensions)
    struct Footprint {
        double length;
        double width;
    };

    // Velocity constraints for planning
    struct VelocityLimits {
        double min_linear;
        double max_linear;
        double min_angular;
        double max_angular;
    };

    // Obstacle placeholder (future use)
    class Obstacle {
    public:
        geometry_msgs::Point center;
        double radius{};
    };

    // Supported local planners
    enum Algorithm {
        DWA,
        DWA_DDP,
        MPPI,
        MPPI_DDP,
        DDP
    };

    // High-level robot operating modes
    enum RobotState {
        INITIALIZING = 0,
        NORMAL_PLANNING = 1,
        LOW_SPEED_PLANNING = 2,
        NO_MAP_PLANNING = 3,
        BRAKE_PLANNING = 4,
        RECOVERY = 5,
        ROTATE_PLANNING = 6,
        BACKWARD = 7,
        FORWARD = 8,
        TEST = 9,
        IDLE = 10
    };

    // Active map source
    enum MapSource {
        ONLY_COSTMAP_RECEIVED = 0,
        ONLY_LASER_RECEIVED = 1,
        NO_ANY_RECEIVED = 2
    };

    Robot_config();

    ~Robot_config() = default;

    TuningParams getTuningParams() const;

    void setTuningParams(const TuningParams &tp);
    void setAlgorithm(Algorithm a) { algorithm = a; }
    void setDt(double t) { dt = t; }
    void setRobotState(RobotState state) { currentState = state; }
    void setLocalGoal(std::vector<double> &lg, double x, double y) {
        local_goal = {lg[0], lg[1]};
        local_goal_odom = {x, y};
    }

    Algorithm getAlgorithm() const { return algorithm; }
    RobotState getRobotState() const { return currentState; }
    PoseState getPoseState() const { return robot_state; }
    double getDt() const { return dt; }
    double getVelocity() const { return robot_state.velocity_; }
    double getAngularVelocity() const { return robot_state.angular_velocity_; }
    bool getMapData();

    std::vector<double> getLocalGoalCfg() { return local_goal; }
    std::vector<double> getGlobalGoalCfg() { return global_goal; }

    const std::vector<Eigen::Vector2f> &getLaserPoints() const { return laserData; }
    std::vector<std::vector<double> > getLaserData();  // Legacy: laser data as double vectors
    std::vector<std::vector<double> > getCostmapDataOdom() const { return costmapDataOdom; }
    std::vector<std::vector<double> > getDataMap() { return map; }

    Footprint getFootprint() const;
    VelocityLimits getVelocityLimits() const;

    costmap_2d::Costmap2DROS *getCostMap() { return costmap; }

    bool setup();  // Prepare map and caches for planning cycle
    bool checkGazeboPaused() const;  // Check if Gazebo simulation is paused

    void triggerRecovery();  // Trigger recovery behavior
    void resetStoppedStatus();  // Reset stopped status flag
    void update_angular_velocity();  // Update angular velocity limits dynamically

    ros::Publisher Control() { return cmd_vel_pub; }
    void publishRobotState() const;
    void publishSmoothedPath(const std::vector<double> &xhat, const std::vector<double> &yhat) const;
    void publishTuningParams() const;

    void view_Goal(std::vector<double> &goal, std::vector<double> &goal1) const;  // Visualize global & local goals
    void viewTrajectories(std::vector<PoseState> &trajectories, int nr_steps_, double theta_, std::vector<double> &t) const;
    void viewTrajectories(std::vector<PoseState> &trajectories, int nr_steps_, std::vector<double> &t) const;

    static double calculateTheta(const PoseState &state, const std::vector<double> &y);

    bool canBeSolved{};
    bool local_goal_received{};
    bool global_goal_received{};
    bool param_received{};

    Algorithm algorithm;
    MapSource currentMap;
    RobotState currentState;

    double dt{};
    double rotating_angle;
    double front_obs{};
    double latter_obs{};

    int recover_times = 0;
    int re = 1;
    int recover_to_low_count = 0;
    double dynamic_recovery_wait_time = 0.5;

    std::vector<std::vector<double> > local_paths;
    std::vector<std::vector<double> > local_paths_odom;
    std::vector<std::vector<double> > local_goals_history;
    std::vector<std::vector<std::vector<double> > > local_paths_history;
    int history_index = 0;

    std::vector<std::vector<double> > actions;
    std::vector<double> local_goal_odom;
    std::vector<std::vector<double> > local_goal_point;
    std::vector<std::vector<double> > local_goal_point_odom;
    std::vector<double> global_goal_odom;

    std::vector<std::vector<geometry_msgs::Point> > polygons;

    std::vector<std::vector<double> > costmapDataOdom;
    std::vector<std::vector<double> > costmapData;
    std::vector<double> laserDataDistance;


    std::vector<double> timeInterval = {
        0.0302, 0.0495, 0.0608, 0.0697, 0.0771, 0.0835, 0.0893, 0.0946, 0.0994, 0.1039,
        0.1082, 0.1122, 0.1160, 0.1196, 0.1231, 0.1264, 0.1296, 0.1327, 0.1357, 0.1386
    };


    // DWA parameters
    double max_vel_x = 2.0;
    double max_vel_y = 0.0;
    double max_vel_theta = 3.0;
    double vx_sample = 10;
    double vTheta_samples = 10;
    double path_distance_bias = 0.7;
    double goal_distance_bias = 0.7;

    // MPPI/DDP parameters
    double nr_pairs_ = 600;
    double nr_steps_ = 20;
    double linear_stddev = 0.1;
    double angular_stddev = 0.05;
    double lambda = 1.0;
    double local_goal_distance = 2.0;
    double distance = 0.3;
    double robot_radius_ = 0.01;

    int num_threads = 8;

    ros::NodeHandle nh;

    ros::Publisher trajectory_pub;
    ros::Publisher global_path_pub;
    ros::Publisher smoothed_global_path_pub;
    ros::Publisher local_goal_pub;
    ros::Publisher global_goal_pub;
    ros::Publisher tuning_params_pub;

    std::shared_ptr<JackalCallbacks> callbacks_;

    std::shared_ptr<AsyncTaskExecutor> async_executor_;

protected:

    // ROS communication
    ros::Subscriber robot_pose_sub;
    ros::Subscriber laser_scan_sub;
    ros::Subscriber dist_to_goal_th_sub_;
    ros::Subscriber costmap_update_sub;
    ros::Subscriber goal_sub;
    ros::Subscriber velocity_sub;
    ros::Subscriber global_path_sub;
    ros::Subscriber array_dt_sub;
    ros::Subscriber params_sub;

    ros::ServiceClient clear_costmaps_clt;
    ros::ServiceClient global_path_clt;
    ros::ServiceClient path_clt;
    ros::Publisher cmd_vel_pub;
    ros::Publisher robot_state_pub;

    costmap_2d::Costmap2DROS *costmap{};

    // Internal state
    // Authoritative tuning parameter snapshot
    TuningParams tuning_params_{};
    std::vector<double> global_goal;
    std::vector<double> local_goal;
    std::vector<double> costmapDataDistance;
    std::vector<double> costmapDataAngle;
    std::vector<Eigen::Vector2f> laserData;
    std::vector<std::vector<double> > map;

    PoseState robot_state;

    // State machine timing
    bool is_stopped = false;
    ros::Time normal_to_low_time;
    bool normal_to_low_active = false;
    ros::Time low_to_normal_time;
    bool low_to_normal_active = false;
    ros::Time low_to_brake_time;
    bool low_to_brake_active = false;

    // Constants
    static constexpr double MIN_SPEED = 0.2;
    static constexpr double STOPPED_TIME_THRESHOLD = 1.0;

    // Robot physical dimensions (meters) - for normal navigation
    static constexpr double ROBOT_LENGTH = 0.508;
    static constexpr double ROBOT_WIDTH = 0.430;

    // Point-mass approximation (nearly zero footprint for tight maneuvers)
    static constexpr double POINT_MASS_LENGTH = 0.02;
    static constexpr double POINT_MASS_WIDTH = 0.02;
};

#endif // DYNAMICS_PLANNER_NAV_JACKAL_HPP
