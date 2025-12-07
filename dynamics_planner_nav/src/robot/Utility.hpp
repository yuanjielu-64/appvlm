
#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <cmath>
#include <Eigen/Dense>
#include <geometry_msgs/PoseStamped.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

double l2_distance(double x1, double y1, double x2, double y2);

std::vector<double> savgolFilter(const std::vector<double>& data, int windowSize, int polyOrder);

geometry_msgs::PoseStamped getPose(double x, double y, double theta);

std::vector<double> transform_lg(double x, double y, double X, double Y, double PSI);

// Normalize angle to [-pi, pi]
inline double normalize_angle(double a) {
    a = std::fmod(a + M_PI, 2 * M_PI);
    if (a < 0)
        a += 2 * M_PI;
    return a - M_PI;
}

#endif //UTILITY_HPP
