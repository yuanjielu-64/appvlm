#include "../robot/Jackal.hpp"
#include "../localPlanners/DDP.hpp"
#include "../localPlanners/DWAPlanner.hpp"
#include "../localPlanners/DWA_DDPPlanner.hpp"
#include "../localPlanners/MPPIPlanner.hpp"
#include "../localPlanners/MPPI_DDPPlanner.hpp"
#include "../utils/Timer.hpp"
#include <ros/ros.h>
#include <unordered_map>
#include <string>
#include <algorithm>

namespace {
    Robot_config::Algorithm parse_algorithm(std::string s) {

        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return static_cast<char>(::toupper(c)); });
        std::replace(s.begin(), s.end(), '-', '_');

        static const std::unordered_map<std::string, Robot_config::Algorithm> kLut = {
            {"DDP",        Robot_config::DDP},
            {"DWA",        Robot_config::DWA},
            {"DWA_DDP",    Robot_config::DWA_DDP},
            {"MPPI",       Robot_config::MPPI},
            {"MPPI_DDP",   Robot_config::MPPI_DDP},
        };
        const auto it = kLut.find(s);
        return it == kLut.end() ? Robot_config::DDP : it->second;
    }
}

extern "C" int RunMP(int argc, char **argv) {
    // Resolve planner selection from CLI or ROS param
    std::string planner_arg;
    for (int i = 1; i + 1 < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--planner" || flag == "-p") { planner_arg = argv[i + 1]; break; }
    }

    // Fallback to ROS private param if not set via CLI
    ros::M_string remappings;
    ros::init(remappings, "dynamics_planner_nav");
    ros::NodeHandle pnh("~");
    if (planner_arg.empty()) { (void)pnh.getParam("planner", planner_arg); }

    Robot_config robot;
    const auto algo = parse_algorithm(planner_arg);
    robot.setAlgorithm(algo);

    double n = 20;  // 20 Hz
    robot.setDt(1.0/n);

    ros::Rate rate(n);

    // Instantiate selected planner and bind solve step
    std::function<void()> solve_step;
    switch (algo) {
        case Robot_config::DWA: {
            static Antipatrea::DWAPlanner planner;
            planner.robot = &robot;
            ROS_INFO("dynamics_planner_nav using DWA planner");
            solve_step = [&](){ (void)planner.Solve(1, robot.getDt(), robot.canBeSolved); };
            break;
        }
        case Robot_config::DWA_DDP: {
            static Antipatrea::DDPDWAPlanner planner;
            planner.robot = &robot;
            ROS_INFO("dynamics_planner_nav using DDPDWA planner");
            solve_step = [&](){ (void)planner.Solve(1, robot.getDt(), robot.canBeSolved); };
            break;
        }
        case Robot_config::MPPI: {
            static Antipatrea::MPPIPlanner planner;
            planner.robot = &robot;
            ROS_INFO("dynamics_planner_nav using MPPI planner");
            solve_step = [&](){ (void)planner.Solve(1, robot.getDt(), robot.canBeSolved); };
            break;
        }
        case Robot_config::MPPI_DDP: {
            static Antipatrea::DDPMPPIPlanner planner;
            planner.robot = &robot;
            ROS_INFO("dynamics_planner_nav using DDPMPPI planner");
            solve_step = [&](){ (void)planner.Solve(1, robot.getDt(), robot.canBeSolved); };
            break;
        }
        case Robot_config::DDP:
        default: {
            static Antipatrea::DDP planner;
            planner.robot = &robot;
            ROS_INFO("dynamics_planner_nav using DDP planner");
            solve_step = [&](){ (void)planner.Solve(1, robot.getDt(), robot.canBeSolved); };
            break;
        }
    }

    while (ros::ok()) {
        ros::spinOnce();

        if (!robot.setup()) {
            if (robot.getRobotState() == Robot_config::BRAKE_PLANNING)
                rate.sleep();
            continue;
        }

    // Call selected planner solve function
    solve_step();

    }

    return 0;
}

int main(int argc, char **argv) {
    return RunMP(argc, argv);
}
