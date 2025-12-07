//
// Created by yuanjielu on 12/10/24.
//

#include "localPlanners/MPPI_DDPPlanner.hpp"
#include "localPlanners/DWA_DDPPlanner.hpp"
#include "localPlanners/DWAPlanner.hpp"
#include "localPlanners/MPPIPlanner.hpp"
#include "localPlanners/DDP.hpp"

namespace Antipatrea {

    void DDP::commonParameters(Robot_config &robot) {
        const auto tp = robot.setTuningParams();
        dt = tp.dt;

        minAccelerSpeed = -20.0;  // Default acceleration limits
        maxAccelerSpeed = 20.0;
        minAngularAccelerSpeed = -25;
        maxAngularAccelerSpeed = 25;
    }

    void DDP::normalParameters(Robot_config &robot) {
        const auto tp = robot.setTuningParams();
        v_steps_ = 20;
        w_steps_ = 20;

        nr_steps_ = 20;

        nr_pairs_ = tp.nr_pairs_;
        linear_stddev = tp.linear_stddev;
        angular_stddev = tp.angular_stddev;

        global_goal = robot.getGlobalGoalCfg();
        local_goal = robot.getLocalGoalCfg();

        distance = tp.distance;
        robot_radius_ = tp.robot_radius_;

        use_goal_cost_ = true;
        use_angular_cost_ = true;
        use_path_cost_ = false;
        use_speed_cost_ = false;
        use_ori_cost_ = true;
        use_space_cost_ = false;

        to_goal_cost_gain_ = 0.9;
        obs_cost_gain_ = 0.3;
        speed_cost_gain_ = 0.1;
        path_cost_gain_ = 0.6;
        ori_cost_gain_ = 0.2;
        aw_cost_gain_ = 0.8;
    }

    void DDP::lowSpeedParameters(Robot_config &robot) {
        const auto tp = robot.setTuningParams();
        v_steps_ = 20;
        w_steps_ = 20;
        nr_steps_ = 20;

        nr_pairs_ = tp.nr_pairs_;
        linear_stddev = tp.linear_stddev;
        angular_stddev = tp.angular_stddev;

        global_goal = robot.getGlobalGoalCfg();
        local_goal = robot.getLocalGoalCfg();

        distance = tp.distance;
        robot_radius_ = tp.robot_radius_;

        use_goal_cost_ = true;
        use_angular_cost_ = false;
        use_path_cost_ = false;
        use_speed_cost_ = false;
        use_ori_cost_ = true;
        use_space_cost_ = true;

        to_goal_cost_gain_ = 0.8;
        obs_cost_gain_ = 0.2;
        speed_cost_gain_ = 0.2;
        path_cost_gain_ = 0.1;
        ori_cost_gain_ = 0.1;
        aw_cost_gain_ = 0.2;
        space_cost_gain_ = 0.2;
    }

    void DDP::recoverParameters(Robot_config &robot) {
        v_steps_ = 20;
        w_steps_ = 20;
        nr_steps_ = (int) robot.timeInterval.size();
        //nr_pairs_ = 550;
        nr_pairs_ = 800;

        global_goal = robot.getGlobalGoalCfg();
        local_goal = robot.getLocalGoalCfg();

        distance = 0.1;
        robot_radius_ = 0.03;

        use_goal_cost_ = true;
        use_angular_cost_ = false;
        use_path_cost_ = false;
        use_speed_cost_ = false;
        use_ori_cost_ = true;
        use_space_cost_ = true;

        to_goal_cost_gain_ = 0.8;
        obs_cost_gain_ = 0.2;
        speed_cost_gain_ = 0.2;
        path_cost_gain_ = 0.1;
        ori_cost_gain_ = 0.1;
        aw_cost_gain_ = 0.2;
        space_cost_gain_ = 0.2;
    }

    void DDP::frontBackParameters(Robot_config &robot) {
        const auto tp = robot.setTuningParams();
        v_steps_ = 20;
        w_steps_ = 25;
        nr_steps_ = (int) robot.timeInterval.size();

        dt = tp.dt;

        global_goal = robot.getGlobalGoalCfg();
        local_goal = robot.getLocalGoalCfg();

        distance = 0.05;
        robot_radius_ = 0.02;

        use_goal_cost_ = false;
        use_angular_cost_ = false;
        use_path_cost_ = true;
        use_speed_cost_ = false;
        use_ori_cost_ = false;

        to_goal_cost_gain_ = 1.0;
        obs_cost_gain_ = 0.2;
        speed_cost_gain_ = 0.2;
        path_cost_gain_ = 0.4;
        ori_cost_gain_ = 0.5;
        aw_cost_gain_ = 0.5;
    }


    void DDPDWAPlanner::commonParameters(Robot_config &robot) {
        const auto tp = robot.setTuningParams();
        dt = tp.dt;

        minAccelerSpeed = -20.0;  // Default acceleration limits
        maxAccelerSpeed = 20.0;
        minAngularAccelerSpeed = -25;
        maxAngularAccelerSpeed = 25;
    }

    void DDPDWAPlanner::normalParameters(Robot_config &robot) {
        const auto tp = robot.setTuningParams();
        v_steps_ = 20;
        w_steps_ = 18;
        nr_steps_ = (int) robot.timeInterval.size();

        dt = tp.dt;

        global_goal = robot.getGlobalGoalCfg();
        local_goal = robot.getLocalGoalCfg();

        distance = 0.01;
        robot_radius_ = 0.1;

        use_goal_cost_ = true;
        use_angular_cost_ = true;
        use_path_cost_ = true;
        use_speed_cost_ = false;
        use_ori_cost_ = false;

        to_goal_cost_gain_ = 0.7;
        obs_cost_gain_ = 0.7;
        speed_cost_gain_ = 0.1;
        path_cost_gain_ = 0.7;
        ori_cost_gain_ = 0.2;
        aw_cost_gain_ = 0.2;
    }


    void DDPDWAPlanner::lowSpeedParameters(Robot_config &robot) {
        const auto tp = robot.setTuningParams();
        v_steps_ = 20;
        w_steps_ = 25;
        nr_steps_ = (int) robot.timeInterval.size();

        dt = tp.dt;

        global_goal = robot.getGlobalGoalCfg();
        local_goal = robot.getLocalGoalCfg();

        distance = 0.1;
        robot_radius_ = 0.1;

        use_goal_cost_ = true;
        use_angular_cost_ = true;
        use_path_cost_ = true;
        use_speed_cost_ = true;
        use_ori_cost_ = true; // close to the local goal

        to_goal_cost_gain_ = 0.8;
        obs_cost_gain_ = 0.6;
        speed_cost_gain_ = 0.2;
        path_cost_gain_ = 0.4;
        ori_cost_gain_ = 0.2;
        aw_cost_gain_ = 0.8;
    }

    void DDPDWAPlanner::recoverParameters(Robot_config &robot) {
        const auto tp = robot.setTuningParams();
        v_steps_ = 20;
        w_steps_ = 20;
        nr_steps_ = (int) robot.timeInterval.size();

        dt = tp.dt;

        global_goal = robot.getGlobalGoalCfg();
        local_goal = robot.getLocalGoalCfg();

        distance = 0.05;
        robot_radius_ = 0.02;

        use_goal_cost_ = true;
        use_angular_cost_ = false;
        use_path_cost_ = true;
        use_speed_cost_ = false;
        use_ori_cost_ = false;

        to_goal_cost_gain_ = 1.0;
        obs_cost_gain_ = 0.2;
        speed_cost_gain_ = 0.2;
        path_cost_gain_ = 0.4;
        ori_cost_gain_ = 0.5;
        aw_cost_gain_ = 0.5;
    }

    void DDPDWAPlanner::frontBackParameters(Robot_config &robot) {
        const auto tp = robot.setTuningParams();
        v_steps_ = 20;
        w_steps_ = 25;
        nr_steps_ = (int) robot.timeInterval.size();

        dt = tp.dt;

        global_goal = robot.getGlobalGoalCfg();
        local_goal = robot.getLocalGoalCfg();

        distance = 0.05;
        robot_radius_ = 0.02;

        use_goal_cost_ = false;
        use_angular_cost_ = false;
        use_path_cost_ = true;
        use_speed_cost_ = false;
        use_ori_cost_ = false;

        to_goal_cost_gain_ = 1.0;
        obs_cost_gain_ = 0.2;
        speed_cost_gain_ = 0.2;
        path_cost_gain_ = 0.4;
        ori_cost_gain_ = 0.5;
        aw_cost_gain_ = 0.5;
    }

    void DWAPlanner::commonParameters(Robot_config &robot) {
        const auto tp = robot.setTuningParams();
        dt = tp.dt;

        minAccelerSpeed = -20.0;  // Default acceleration limits
        maxAccelerSpeed = 20.0;
        minAngularAccelerSpeed = -25;
        maxAngularAccelerSpeed = 25;
    }

    void DWAPlanner::normalParameters(Robot_config &robot) {
        const auto tp = robot.setTuningParams();
        v_steps_ = tp.vx_sample;
        w_steps_ = tp.vTheta_samples;
        nr_steps_ = 20;

        dt = tp.dt;

        global_goal = robot.getGlobalGoalCfg();
        local_goal = robot.getLocalGoalCfg();

        distance = 0.1;
        robot_radius_ = 0.1;

        use_goal_cost_ = true;
        use_path_cost_ = true;
        use_angular_cost_ = false;
        use_speed_cost_ = false;
        use_ori_cost_ = false;

        to_goal_cost_gain_ = goal_distance_bias;
        obs_cost_gain_ = 0.7;
        speed_cost_gain_ = 0.1;
        path_cost_gain_ = path_distance_bias;
        ori_cost_gain_ = 0.2;
        aw_cost_gain_ = 0.2;

        for (int i = 0; i < nr_steps_; ++i) {
            timeInterval.push_back(dt);
        }

    }

    void DWAPlanner::lowSpeedParameters(Robot_config &robot) {
        const auto tp = robot.setTuningParams();
        v_steps_ = 20;
        w_steps_ = 25;
        nr_steps_ = 20;

        dt = tp.dt;

        global_goal = robot.getGlobalGoalCfg();
        local_goal = robot.getLocalGoalCfg();

        distance = 0.1;
        robot_radius_ = 0.1;

        use_goal_cost_ = true;
        use_angular_cost_ = true;
        use_path_cost_ = true;
        use_speed_cost_ = true;
        use_ori_cost_ = true; // close to the local goal

        to_goal_cost_gain_ = 0.8;
        obs_cost_gain_ = 0.6;
        speed_cost_gain_ = 0.2;
        path_cost_gain_ = 0.4;
        ori_cost_gain_ = 0.2;
        aw_cost_gain_ = 0.8;

        for (int i = 0; i < nr_steps_; ++i) {
            timeInterval.push_back(dt);
        }

    }

    void DWAPlanner::recoverParameters(Robot_config &robot) {
        const auto tp = robot.setTuningParams();
        v_steps_ = 20;
        w_steps_ = 25;
        nr_steps_ = 20;

        dt = tp.dt;

        global_goal = robot.getGlobalGoalCfg();
        local_goal = robot.getLocalGoalCfg();

        distance = 0.05;
        robot_radius_ = 0.02;

        use_goal_cost_ = true;
        use_angular_cost_ = false;
        use_path_cost_ = true;
        use_speed_cost_ = false;
        use_ori_cost_ = false;

        to_goal_cost_gain_ = 1.0;
        obs_cost_gain_ = 0.2;
        speed_cost_gain_ = 0.2;
        path_cost_gain_ = 0.4;
        ori_cost_gain_ = 0.5;
        aw_cost_gain_ = 0.5;

        for (int i = 0; i < nr_steps_; ++i) {
            timeInterval.push_back(dt);
        }

    }

    void DWAPlanner::frontBackParameters(Robot_config &robot) {
        const auto tp = robot.setTuningParams();
        v_steps_ = 20;
        w_steps_ = 25;
        nr_steps_ = 20;

        dt = tp.dt;

        global_goal = robot.getGlobalGoalCfg();
        local_goal = robot.getLocalGoalCfg();

        distance = 0.05;
        robot_radius_ = 0.02;

        use_goal_cost_ = false;
        use_angular_cost_ = false;
        use_path_cost_ = true;
        use_speed_cost_ = false;
        use_ori_cost_ = false;

        to_goal_cost_gain_ = 1.0;
        obs_cost_gain_ = 0.2;
        speed_cost_gain_ = 0.2;
        path_cost_gain_ = 0.4;
        ori_cost_gain_ = 0.5;
        aw_cost_gain_ = 0.5;

        for (int i = 0; i < nr_steps_; ++i) {
            timeInterval.push_back(dt);
        }

    }


    void DDPMPPIPlanner::commonParameters(Robot_config &robot) {
        const auto tp = robot.setTuningParams();
        dt = tp.dt;

        minAccelerSpeed = -20.0;  // Default acceleration limits
        maxAccelerSpeed = 20.0;
        minAngularAccelerSpeed = -25;
        maxAngularAccelerSpeed = 25;
    }

    void DDPMPPIPlanner::normalParameters(Robot_config &robot) {
        v_steps_ = 20;
        w_steps_ = 20;
        nr_steps_ = (int) robot.timeInterval.size();
        nr_pairs_ = 550;

        global_goal = robot.getGlobalGoalCfg();
        local_goal = robot.getLocalGoalCfg();

        distance = 0.1;
        robot_radius_ = 0.1;

        use_goal_cost_ = true;
        use_angular_cost_ = true;
        use_path_cost_ = true;
        use_speed_cost_ = true;
        use_ori_cost_ = false;

        to_goal_cost_gain_ = 0.8;
        obs_cost_gain_ = 0.5;
        speed_cost_gain_ = 0.1;
        path_cost_gain_ = 0.7;
        ori_cost_gain_ = 0.2;
        aw_cost_gain_ = 0.8;
    }

    void DDPMPPIPlanner::lowSpeedParameters(Robot_config &robot) {
        v_steps_ = 20;
        w_steps_ = 20;
        nr_steps_ = (int) robot.timeInterval.size();
        nr_pairs_ = 550;

        global_goal = robot.getGlobalGoalCfg();
        local_goal = robot.getLocalGoalCfg();

        distance = 0.1;
        robot_radius_ = 0.1;

        use_goal_cost_ = true;
        use_angular_cost_ = true;
        use_path_cost_ = true;
        use_speed_cost_ = true;
        use_ori_cost_ = true; // close to the local goal

        to_goal_cost_gain_ = 0.8;
        obs_cost_gain_ = 0.6;
        speed_cost_gain_ = 0.2;
        path_cost_gain_ = 0.4;
        ori_cost_gain_ = 0.2;
        aw_cost_gain_ = 0.8;
    }

    void DDPMPPIPlanner::recoverParameters(Robot_config &robot) {
        v_steps_ = 20;
        w_steps_ = 20;
        nr_steps_ = (int) robot.timeInterval.size();
        nr_pairs_ = 550;

        distance = 0.05;
        robot_radius_ = 0.02;

        global_goal = robot.getGlobalGoalCfg();
        local_goal = robot.getLocalGoalCfg();

        use_goal_cost_ = true;
        use_angular_cost_ = false;
        use_path_cost_ = true;
        use_speed_cost_ = false;
        use_ori_cost_ = false;

        to_goal_cost_gain_ = 1.0;
        obs_cost_gain_ = 0.2;
        speed_cost_gain_ = 0.2;
        path_cost_gain_ = 0.4;
        ori_cost_gain_ = 0.5;
        aw_cost_gain_ = 0.5;
    }

    void DDPMPPIPlanner::frontBackParameters(Robot_config &robot) {
        v_steps_ = 20;
        w_steps_ = 20;
        nr_steps_ = (int) robot.timeInterval.size();
        nr_pairs_ = 550;

        distance = 0.05;
        robot_radius_ = 0.02;

        global_goal = robot.getGlobalGoalCfg();
        local_goal = robot.getLocalGoalCfg();

        use_goal_cost_ = false;
        use_angular_cost_ = false;
        use_path_cost_ = true;
        use_speed_cost_ = false;
        use_ori_cost_ = false;

        to_goal_cost_gain_ = 1.0;
        obs_cost_gain_ = 0.2;
        speed_cost_gain_ = 0.2;
        path_cost_gain_ = 0.4;
        ori_cost_gain_ = 0.5;
        aw_cost_gain_ = 0.5;
    }

    void MPPIPlanner::commonParameters(Robot_config &robot) {
        const auto tp = robot.setTuningParams();
        dt = tp.dt;

        minAccelerSpeed = -20.0;  // Default acceleration limits
        maxAccelerSpeed = 20.0;
        minAngularAccelerSpeed = -25;
        maxAngularAccelerSpeed = 25;
    }

     void MPPIPlanner::normalParameters(Robot_config &robot) {
        const auto tp = robot.setTuningParams();
        v_steps_ = 20;
        w_steps_ = 20;
        nr_steps_ = tp.nr_steps_;
        nr_pairs_ = tp.nr_pairs_;
        linear_stddev = tp.linear_stddev;
        angular_stddev = tp.angular_stddev;
        lambda = tp.lambda;
        exploration_ratio = 0.8;

        global_goal = robot.getGlobalGoalCfg();
        local_goal = robot.getLocalGoalCfg();

        distance = 0.01;
        robot_radius_ = 0.01;

        use_goal_cost_ = true;
        use_angular_cost_ = true;
        use_path_cost_ = true;
        use_speed_cost_ = true;
        use_ori_cost_ = false;

        to_goal_cost_gain_ = 0.8;
        obs_cost_gain_ = 0.5;
        speed_cost_gain_ = 0.1;
        path_cost_gain_ = 0.7;
        ori_cost_gain_ = 0.2;
        aw_cost_gain_ = 0.8;

        for (int i = 0; i < nr_steps_; ++i) {
            timeInterval.push_back(dt);
        }
    }

    void MPPIPlanner::lowSpeedParameters(Robot_config &robot) {
        v_steps_ = 20;
        w_steps_ = 20;
        nr_steps_ = 20;
        nr_pairs_ = 600;

        global_goal = robot.getGlobalGoalCfg();
        local_goal = robot.getLocalGoalCfg();

        distance = 0.01;
        robot_radius_ = 0.01;

        use_goal_cost_ = true;
        use_angular_cost_ = true;
        use_path_cost_ = true;
        use_speed_cost_ = true;
        use_ori_cost_ = true; // close to the local goal

        to_goal_cost_gain_ = 0.8;
        obs_cost_gain_ = 0.6;
        speed_cost_gain_ = 0.2;
        path_cost_gain_ = 0.4;
        ori_cost_gain_ = 0.2;
        aw_cost_gain_ = 0.8;


        for (int i = 0; i < nr_steps_; ++i) {
            timeInterval.push_back(dt);
        }


    }

    void MPPIPlanner::recoverParameters(Robot_config &robot) {
        v_steps_ = 20;
        w_steps_ = 20;
        nr_steps_ = 20;
        nr_pairs_ = 600;

        distance = 0.05;
        robot_radius_ = 0.02;

        global_goal = robot.getGlobalGoalCfg();
        local_goal = robot.getLocalGoalCfg();

        use_goal_cost_ = true;
        use_angular_cost_ = false;
        use_path_cost_ = true;
        use_speed_cost_ = false;
        use_ori_cost_ = false;

        to_goal_cost_gain_ = 1.0;
        obs_cost_gain_ = 0.2;
        speed_cost_gain_ = 0.2;
        path_cost_gain_ = 0.4;
        ori_cost_gain_ = 0.5;
        aw_cost_gain_ = 0.5;

        for (int i = 0; i < nr_steps_; ++i) {
            timeInterval.push_back(dt);
        }

    }

    void MPPIPlanner::frontBackParameters(Robot_config &robot) {
        v_steps_ = 20;
        w_steps_ = 20;
        nr_steps_ = 20;
        nr_pairs_ = 600;

        distance = 0.05;
        robot_radius_ = 0.02;

        global_goal = robot.getGlobalGoalCfg();
        local_goal = robot.getLocalGoalCfg();

        use_goal_cost_ = false;
        use_angular_cost_ = false;
        use_path_cost_ = true;
        use_speed_cost_ = false;
        use_ori_cost_ = false;

        to_goal_cost_gain_ = 1.0;
        obs_cost_gain_ = 0.2;
        speed_cost_gain_ = 0.2;
        path_cost_gain_ = 0.4;
        ori_cost_gain_ = 0.5;
        aw_cost_gain_ = 0.5;

        for (int i = 0; i < nr_steps_; ++i) {
            timeInterval.push_back(dt);
        }

    }
}
