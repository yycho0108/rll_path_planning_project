#!/bin/bash

case $1 in
    1*)
        roslaunch rll_planning_project moveit_planning_execution.launch gui:=false
        ;;
    2*)
        roslaunch rll_planning_project planning_iface.launch
        ;;
    3*)
        roslaunch rll_planning_project path_planner.launch
        ;;
    4*)
        roslaunch rll_project_runner run_project.launch
        ;;
    *)
        echo 'only steps 1-4 are valid'
        ;;
esac
