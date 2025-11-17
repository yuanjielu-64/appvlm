#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/gazebo_client.hh>
#include <gazebo/gazebo_config.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <geometry_msgs/Vector3.h>
#include <std_msgs/Bool.h>
#include <iostream>
#include <vector>

ros::Publisher pub;
bool airborne;
bool is_colliding = false;
const std::string DELIMITER = "::";

// Forces callback function
void forcesCb(ConstContactsPtr &_msg){
    bool collision_detected = false;

    for (int i = 0; i < _msg->contact_size(); ++i) {
        std::string entity1 = _msg->contact(i).collision1();
        entity1 = entity1.substr(0, entity1.find(DELIMITER));

        std::string entity2 = _msg->contact(i).collision2();
        entity2 = entity2.substr(0, entity2.find(DELIMITER));

        if(entity1 != "ground_plane" && entity2 != "ground_plane"){
            if (entity1 == "jackal" || entity2 == "jackal"){
                collision_detected = true;
                ROS_INFO_STREAM("Collision: " << entity1 << " <-> " << entity2);
                break;  
            }
        }
    }
    
    is_colliding = collision_detected;
}

// Position callback function
void positionCb(const nav_msgs::Odometry::ConstPtr& msg2){
    if (msg2->pose.pose.position.z > 0.3) {
        airborne = true;
    } else {
        airborne = false;
    }
}

int main(int _argc, char **_argv){
    // Set variables
    airborne = false;
    is_colliding = false;

    // Load Gazebo & ROS
    gazebo::client::setup(_argc, _argv);
    ros::init(_argc, _argv, "collision_publisher");

    // Create Gazebo node and init
    gazebo::transport::NodePtr node(new gazebo::transport::Node());
    node->Init();

    // Create ROS node and init
    ros::NodeHandle n;
    pub = n.advertise<std_msgs::Bool>("collision", 1000);

    // Listen to Gazebo contacts topic
    gazebo::transport::SubscriberPtr sub = node->Subscribe("/gazebo/default/physics/contacts", forcesCb);

    // Listen to ROS for position
    ros::Subscriber sub2 = n.subscribe("ground_truth/state", 1000, positionCb);

    ros::Rate loop_rate(50);

    ROS_INFO("Collision publisher started, publishing at 50Hz");

    // Busy wait loop...replace with your own code as needed.
    // Busy wait loop...replace with your own code as needed.
    while (ros::ok()) {

        std_msgs::Bool collision_msg;
        collision_msg.data = is_colliding;
        pub.publish(collision_msg);

        ros::spinOnce();
        
        loop_rate.sleep();
    }

    gazebo::client::shutdown();

    return 0;
}