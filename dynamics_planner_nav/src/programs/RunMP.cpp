#include "../robot/Jackal.hpp"
#include "../localPlanners/DDP.hpp"
#include "../utils/Timer.hpp"
#include <ros/ros.h>

extern "C" int RunMP(int argc, char **argv) {
    ros::init(argc, argv, "dynamics_planner_nav_ddp");

    Robot_config robot;
    robot.setAlgorithm(Robot_config::DDP);

    double n = 20;  // 20 Hz
    robot.setDt(1.0/n);

    ros::Rate rate(n);

    // Create DDP planner
    Antipatrea::DDP planner;
    planner.robot = &robot;

    ROS_INFO("dynamics_planner_nav DDP node started");

    while (ros::ok()) {
        ros::spinOnce();

        if (!robot.setup()) {
            if (robot.getRobotState() == Robot_config::BRAKE_PLANNING)
                rate.sleep();
            continue;
        }

        // Call planner's Solve function
        planner.Solve(1, robot.getDt(), robot.canBeSolved);

    }

    return 0;
}

int main(int argc, char **argv) {
    return RunMP(argc, argv);
}
