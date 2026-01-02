// Jackal_recovery.cpp — Recovery behavior implementations
// Member functions for Robot_config class related to recovery

#include "Jackal.hpp"
#include "std_srvs/Empty.h"

//==============================================================================
// RECOVERY BEHAVIORS
//==============================================================================

// Clear costmaps and reset stopped state
void Robot_config::triggerRecovery() {
    std_srvs::Empty srv;
    if (clear_costmaps_clt.call(srv)) {
        ROS_INFO("Recovery behavior triggered.");
        local_goal_odom.clear();
    } else {
        ROS_ERROR("Failed to call clear_costmaps service.");
    }

    resetStoppedStatus();
}

// Reset stopped flag and log resume
void Robot_config::resetStoppedStatus() {
    if (is_stopped) {
        is_stopped = false;
        ROS_INFO("Robot resumed moving.");
    }
}
