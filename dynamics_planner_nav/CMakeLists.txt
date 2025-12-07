cmake_minimum_required(VERSION 3.10...3.26)
project(dynamics_planner_nav)

# Suppress new CMake dev warning about deprecated FindPython* in catkin on
# newer CMake versions, without affecting older versions.
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    sensor_msgs
    move_base_msgs
    visualization_msgs
    tf2
    tf2_ros
    tf2_geometry_msgs
    costmap_2d
)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

catkin_package(
    CATKIN_DEPENDS
        roscpp
        std_msgs
        geometry_msgs
        nav_msgs
        sensor_msgs
        move_base_msgs
        visualization_msgs
        tf2
        tf2_ros
        tf2_geometry_msgs
        costmap_2d
)

include_directories(
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    src
)

## Collect source files
file(GLOB ROBOT_SOURCES src/robot/*.cpp)
file(GLOB LOCAL_PLANNER_SOURCES src/localPlanners/*.cpp)
file(GLOB GLOBAL_PLANNER_SOURCES src/globalPlanners/*.cpp)

# Ensure new modular units are compiled even if CMake glob cache lags
list(APPEND ROBOT_SOURCES
        src/robot/Jackal_publishers.cpp
    src/robot/Jackal_callbacks.cpp
)

# Only include essential Utils files needed by planners
set(UTILS_SOURCES
    src/utils/Algebra.cpp
    src/utils/PseudoRandom.cpp
)

## Build main executable
add_executable(${PROJECT_NAME}_node
    src/programs/RunMP.cpp
    ${ROBOT_SOURCES}
    ${LOCAL_PLANNER_SOURCES}
    ${GLOBAL_PLANNER_SOURCES}
    ${UTILS_SOURCES}
)

target_link_libraries(${PROJECT_NAME}_node
    ${catkin_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    ${OpenCV_LIBS}
    pthread
)

add_dependencies(${PROJECT_NAME}_node
    ${${PROJECT_NAME}_EXPORTED_TARGETS}
    ${catkin_EXPORTED_TARGETS}
)
