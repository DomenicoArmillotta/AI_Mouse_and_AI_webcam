#!/bin/bash


if [ -e "/dev/video0" ]; then
    echo "Webcam connected."


    if [ -z "$ROS_DISTRO" ]; then
        echo "Errore ROS"
        exit 1
    fi

    # new Terminal with node 1
    gnome-terminal -- ros2 run my_package topic_webcam_pub

    # new Terminal with node 2
    gnome-terminal -- ros2 run my_package topic_webcam_sub

else
    echo "Errore: Webcam not connected"
fi

