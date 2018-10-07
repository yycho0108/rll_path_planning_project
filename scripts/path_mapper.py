#! /usr/bin/env python

import rospy
import actionlib
from rll_planning_project.srv import *
from rll_planning_project.msg import *
from rll_planning_project import utils as U
#from rll_planning_project.mc_mapper import monte_carlo_checker
#from rll_planning_project.grid_mapper import GridMapper
from rll_planning_project.skel_mapper import SkelMapper

from geometry_msgs.msg import Pose2D
from heapq import heappush, heappop # for priority queue
import math

import numpy as np
import cv2
from std_srvs.srv import Empty, EmptyResponse
from scipy.spatial import ConvexHull

#def main():
#    rospy.init_node('cast')
#    srv = rospy.ServiceProxy('check_path', CheckPath, persistent=True)
#
#    # map params
#    map_width  = rospy.get_param('~map_width' , default=1.2)
#    map_length = rospy.get_param('~map_length', default=1.6)
#    map_res    = rospy.get_param('~map_res'   , default=0.005)
#    n, m = int(np.round(map_width / map_res)), int(np.round((map_length / map_res)))
#    map = np.zeros(shape=(n,m), dtype=np.uint8)
#
#    # footprint params
#    fw = float(rospy.get_param('~fw', default=0.06)) # footprint width
#    fh = float(rospy.get_param('~fh', default=0.07)) # footprint height
#    fpt = [[-fw/2,-fh/2],[-fw/2,fh/2],[fw/2,fh/2],[fw/2,-fh/2]] # 4x2
#    fpt = np.asarray(fpt, dtype=np.float32).T #2x4
#
#    grid_checker = GridMapper(map_width, map_length)
#
#    # begin cycle
#    def step_cb(_):
#        #monte_carlo_checker(srv,map,fpt,map_width,map_length)
#        grid_checker(srv, map, fpt)
#        return EmptyResponse()
#
#    call = rospy.Service('step', Empty, step_cb)
#
#    rate = rospy.Rate(50)
#    while not rospy.is_shutdown():
#        grid_checker(srv, map, fpt)
#        cv2.imshow('map', map)
#        cv2.waitKey(10)
#        #rate.sleep()
#    #monte_carlo_checker(srv, map, fpt, map_width, map_height)
#    #print cast_ray(srv,[0.38,-0.5], [0.2,0.5], h=0, min_d=0.005)
#
#if __name__ == "__main__":
#    main()

def plan_to_goal(req):
    """ Plan a path from Start to Goal """
    pose_start = Pose2D()
    pose_goal = Pose2D()
    pose_check_start = Pose2D()
    pose_check_goal = Pose2D()
    pose_move = Pose2D()

    rospy.loginfo("Got a planning request")

    pose_start = req.start
    pose_goal = req.goal

    move_srv = rospy.ServiceProxy('move', Move)
    check_srv = rospy.ServiceProxy('check_path', CheckPath, persistent=True)

    ###############################################
    # Implement your path planning algorithm here #
    ###############################################

    # Input: map dimensions, start pose, and goal pose
    # retrieving input values  
    map_width = rospy.get_param('~map_width')
    map_length = rospy.get_param('~map_length')
    xStart, yStart, tStart = pose_start.x, pose_start.y, pose_start.theta
    xGoal, yGoal, tGoal = pose_goal.x, pose_goal.y, pose_goal.theta
    # printing input values
    rospy.loginfo("map dimensions: width=%1.2fm, length=%1.2fm", map_width, map_length)
    rospy.loginfo("start pose: x %f, y %f, theta %f", xStart, yStart, tStart)
    rospy.loginfo("goal pose: x %f, y %f, theta %f", xGoal, yGoal, tGoal)

    # Output: movement commands
    pose_check_start.x, pose_check_start.y, pose_check_start.theta= xStart, yStart, tStart
    pose_check_goal.x, pose_check_goal.y, pose_check_goal.theta= xGoal, yGoal, tGoal
    resp = check_srv(pose_check_start, pose_check_goal) # checking if the arm can move to the goal pose
    if resp.valid:
        rospy.loginfo("Valid pose")
        pose_move.x, pose_move.y, pose_move.theta = xGoal, yGoal, tGoal 
        # executing a move command towards the goal pose
        resp = move_srv(pose_move)
    else:
        rospy.loginfo("Invalid pose")

class PathMapper:
    def __init__(self):
        # map params
        self._mw = mw = float(rospy.get_param('~map_width', default=1.2))
        self._mh = mh = float(rospy.get_param('~map_length', default=1.6))
        self._mr = mr = float(rospy.get_param('~map_res', default=0.005))
        n, m = int(np.round(mw/mr)), int(np.round((mh/mr)))
        self._map = np.zeros(shape=(n,m), dtype=np.uint8)

        # footprint params
        self._fw = fw = float(rospy.get_param('~fw', default=0.06)) # footprint width
        self._fh = fh = float(rospy.get_param('~fh', default=0.07)) # footprint height
        fpt = [[-fw/2,-fh/2],[-fw/2,fh/2],[fw/2,fh/2],[fw/2,-fh/2]] # 4x2
        self._fpt = np.asarray(fpt, dtype=np.float32).T #2x4

        # misc params
        self._rate = float(rospy.get_param('~rate', default=50.0))

        # Flag
        self._init_flag = False
        self._stop_flag = False

        # mapper handle
        #self.mapper_ = monte_carlo_mapper
        #self.mapper_ = GridMapper(mw, mh, fw, fh, r=0.02)
        self.mapper_ = SkelMapper(mw, mh, fw, fh, r=0.02)

        # Provide Services
        self._step = rospy.Service('step', Empty, self.step_cb)
        self._stop = rospy.Service('stop', Empty, self.stop_cb)
        self.server = actionlib.SimpleActionServer("plan_to_goal", PlanToGoalAction, self.execute, False)
        self.server.start()

        rospy.on_shutdown(self.mapper_.save)

    def step_cb(self, _):
        if not self._init_flag:
            return EmptyResponse()
        try:
            self.mapper_(self._check_srv, self._map, self._fpt, log=rospy.loginfo)
            #monte_carlo_checker(self._check_srv,
            #        self._map,self._fpt,
            #        self._mw, self._mh, a=0)#np.pi/2)
        except Exception as e:
            rospy.logerr_throttle(1.0, 'Mapper Failed : {}'.format(e))
        return EmptyResponse()

    def stop_cb(self, _):
        self._stop_flag = True
        return EmptyResponse()

    def execute(self, req):
        self._init_flag = True
        rate = rospy.Rate(self._rate)

        # Create Service handles
        self._stop_flag = False
        self._check_srv = rospy.ServiceProxy('check_path', CheckPath, persistent=True)
        self._move_srv  = rospy.ServiceProxy('move', Move)

        delay_t = (1000 / self._rate) if (self._rate > 0) else 10
        delay_t = np.round(delay_t).astype(np.int32)

        cv_flag = False
        ros_flag = False

        src = [req.start.x, req.start.y, req.start.theta]
        dst = [req.goal.x, req.goal.y, req.goal.theta]
        self.mapper_.reset(self._check_srv, src, dst)

        while True:
            #rate.sleep()
            #self.step_cb(None)
            cv2.imshow('map', self._map)
            k = cv2.waitKey(max(1, delay_t))

            cv_flag  = (k == 27)
            ros_flag = rospy.is_shutdown()
            if cv_flag | ros_flag | self._stop_flag:
                break

        # after everything is done, pretend to be path_planner.py
        if not (cv_flag or ros_flag):
            plan_to_goal(req)
        self.server.set_succeeded()

if __name__ == '__main__':
    rospy.init_node('path_planner')
    server = PathMapper()
    rospy.spin()
