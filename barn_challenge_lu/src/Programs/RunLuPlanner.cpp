#include "Programs/Setup.hpp"
#include "Utils/Timer.hpp"
#include "Utils/Stats.hpp"
#include "Components/TourGeneratorExact.hpp"
#include <fstream>
#include "Programs/GManagerDecomposition.hpp"
#include "Programs/GManagerGoals.hpp"
#include "Programs/GManagerMP.hpp"
#include "Programs/GManagerSimulator.hpp"
#include "Utils/GManager.hpp"
#include "Robot/Jackal.hpp"
#include <iostream>
#include <fstream>
#include <thread>

using namespace Antipatrea;

extern "C" int RunMPPI(int argc, char **argv) {
 Setup setup;
    Params params;

    ros::init(argc, argv, "MPPI");

    params.ReadFromFile(argv[1]);
    params.ProcessArgs(2, argc - 1, argv);

    Robot_config robot;
    robot.setAlgorithm(Robot_config::LuPlanner);

    double n = 20;
    robot.setDt(1.0/n);

    ros::Rate rate(n);

    setup.SetupFromParams(params, robot);
    int state = Robot_config::NORMAL_PLANNING;

    while (ros::ok()) {
        Timer::Clock start_time;
        Timer::Start(start_time);

        ros::spinOnce();

        if (!robot.setup()) {
            if (robot.getRobotState() == Robot_config::BRAKE_PLANNING)
                rate.sleep();
            continue;
        }

        setup.UpdateFromParams(params, robot, state);

        // const auto &pose = robot.getPoseState();
        // Logger::m_out << "  x " << pose.x_
        //               << "  y " << pose.y_
        //               << "  theta " << pose.theta_
        //               << "  v " << pose.velocity_
        //               << "  w " << pose.angular_velocity_ << std::endl;

        // Logger::m_out << "Loading nodes took: " << Timer::Elapsed(start_time) << " seconds" << std::endl;

        static const std::unordered_map<int, std::string> state_descriptions = {
            {Robot_config::INITIALIZING, "initializing"},
            {Robot_config::NORMAL_PLANNING, "normal planning"},
            {Robot_config::ROTATE_PLANNING, "rotate planning"},
            {Robot_config::RECOVERY, "recovery"},
            {Robot_config::LOW_SPEED_PLANNING, "low speed planning"},
            {Robot_config::NO_MAP_PLANNING, "no map"},
            {Robot_config::BACKWARD, "backward"},
            {Robot_config::FORWARD, "forward"},
            {Robot_config::BRAKE_PLANNING, "brake planning"}
        };

        setup.GetMP()->Solve(1, 0.05, robot.canBeSolved);

        // auto state_it = state_descriptions.find(robot.getRobotState());
        // if (state_it != state_descriptions.end()) {
        //     Logger::m_out << "Robot STATE: " << state_it->second << std::endl;
        // } else {
        //     Logger::m_out << "Robot STATE: unknown" << std::endl;
        // }

        // Logger::m_out << "Task execution cost: " << Timer::Elapsed(start_time) << " seconds" << std::endl;
        // Logger::m_out << std::endl;

        // rate.sleep();
    }

    return 0;
}
