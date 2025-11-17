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

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>

// 在main函数或类中添加文件输出流
std::ofstream velocity_log_file;
std::string log_filename;

// 在程序开始时初始化日志文件
void initializeVelocityLog() {
    // 生成带时间戳的文件名
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;

    // 保存到桌面
    std::string desktop_path = std::string(getenv("HOME")) + "/Desktop/";
    ss << desktop_path << "velocity_log_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << ".txt";
    log_filename = ss.str();

    std::cout << "Creating velocity log file: " << log_filename << std::endl;

    velocity_log_file.open(log_filename);
    if (velocity_log_file.is_open()) {
        // 写入文件头
        velocity_log_file << "# Robot Velocity Log - Created: "
                         << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << std::endl;
        velocity_log_file << "# Format: timestamp, x, y, theta, linear_velocity, angular_velocity, gazebo_paused, robot_state" << std::endl;
        velocity_log_file << "# timestamp(s), x(m), y(m), theta(rad), v(m/s), w(rad/s), paused(0/1), state" << std::endl;
        velocity_log_file.flush();  // 立即写入磁盘

        std::cout << "SUCCESS: Velocity log file created at: " << log_filename << std::endl;
        ROS_INFO("Velocity log file created: %s", log_filename.c_str());
    } else {
        std::cout << "ERROR: Failed to create velocity log file: " << log_filename << std::endl;
        ROS_ERROR("Failed to create velocity log file: %s", log_filename.c_str());
    }
}

// 保存速度数据到文件
void logVelocityData(Robot_config& robot) {
    if (!velocity_log_file.is_open()) return;

    const auto &pose = robot.getPoseState();
    ros::Time current_time = ros::Time::now();
    bool is_paused = robot.isPaused();
    int robot_state = robot.getRobotState();


    velocity_log_file << std::fixed << std::setprecision(6)
                     << current_time.toSec() << ", "
                     << pose.x_ << ", "
                     << pose.y_ << ", "
                     << pose.theta_ << ", "
                     << pose.velocity_ << ", "
                     << pose.angular_velocity_ << ", "
                     << (is_paused ? 1 : 0) << ", "
                     << robot_state << std::endl;


    static int log_counter = 0;
    if (++log_counter % 10 == 0) {
        velocity_log_file.flush();
    }
}

extern "C" int RunMP(int argc, char **argv) {
    Setup setup;
    Params params;

    ros::init(argc, argv, "DDP");

    params.ReadFromFile(argv[1]);
    params.ProcessArgs(2, argc - 1, argv);

    Robot_config robot;
    robot.setAlgorithm(Robot_config::DDP);

    double n = 20;
    robot.setDt(1.0/n);

    ros::Rate rate(n);

    //initializeVelocityLog();

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
        
        
        //logVelocityData(robot);

        setup.UpdateFromParams(params, robot, state);

        const auto &pose = robot.getPoseState();

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

        auto state_it = state_descriptions.find(robot.getRobotState());

        // if (state_it != state_descriptions.end()) {
        //     Logger::m_out << "Robot STATE: " << state_it->second << std::endl;
        // } else {
        //     Logger::m_out << "Robot STATE: unknown" << std::endl;
        // }

        //if (Timer::Elapsed(start_time) >= 0.05) {
             // ROS_ERROR_THROTTLE(0.5, "task execution > 0.5!!!");
            // Logger::m_out << "Task execution cost: " << Timer::Elapsed(start_time) << " seconds" << std::endl;
        //}

        //Logger::m_out << "Task execution cost: " << Timer::Elapsed(start_time) << " seconds" << std::endl;

        // Logger::m_out << std::endl;

        rate.sleep();
    }

    return 0;
}

