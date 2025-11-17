FROM ros:noetic

# Install dependencies
RUN apt update && apt install -y \
    python3-pip \
    git \
    libprotobuf-dev \
    protobuf-compiler \
    gazebo11 \
    libgazebo11-dev \
    ros-noetic-gazebo-ros \
    ros-noetic-gazebo-ros-pkgs \
    ros-noetic-gazebo-ros-control \
    ros-noetic-tf2 \
    ros-noetic-tf2-ros \
    ros-noetic-tf2-geometry-msgs \
    ros-noetic-tf2-msgs \
    ros-noetic-tf2-eigen \
    ros-noetic-tf2-kdl \
    ros-noetic-tf2-sensor-msgs \
    ros-noetic-nmea-msgs \
    ros-noetic-roslint \
    ros-noetic-rosserial-server \
    ros-noetic-twist-mux \
    ros-noetic-hector-gazebo-plugins \
    ros-noetic-xacro \
    ros-noetic-rosdoc-lite \
    ros-noetic-teleop-twist-joy \
    ros-noetic-costmap-2d \
    ros-noetic-rviz \
    ros-noetic-robot-upstart \
    ros-noetic-jackal-simulator \
    ros-noetic-jackal-desktop \
    ros-noetic-jackal-navigation \
    ros-noetic-rosserial-python \
    ros-noetic-nmea-navsat-driver \
    && apt clean

# Upgrade pip
RUN pip3 install --upgrade pip

# Set working directory
WORKDIR /jackal_ws/src

# Clone repository
RUN git clone https://github.com/yuanjielu-64/adaptive_dynamics_learning.git .

# Install Python dependencies
WORKDIR /jackal_ws/src/ros_jackal
RUN pip3 install -r requirements.txt
RUN pip3 install scipy gymnasium psutil

# Set up ROS workspace
WORKDIR /jackal_ws
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash; catkin_make"

# Set environment
ENV ROS_WORKSPACE=/jackal_ws
WORKDIR /jackal_ws/src

# Default command
CMD ["/bin/bash"]