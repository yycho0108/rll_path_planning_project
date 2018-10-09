#!/usr/bin/env python2
import os
import cv2
import numpy as np
import time
from rll_planning_project.srv import *
from rll_planning_project.msg import *
from rll_planning_project.map_utils import fpt_hull
from rll_planning_project import utils as U

import rospkg
rospack = rospkg.RosPack()

import rospy
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import Pose2D

import actionlib

class MinSim(object):
    """ Minified Simulation of Planning Project """
    def __init__(self):
        # default path setup
        map_file = os.path.join(
                rospack.get_path('rll_planning_project'),
                'figs', 'rll_map.png')

        # get params
        self._map_file = rospy.get_param('~map_file', default=map_file)
        self._mw = float(rospy.get_param('~mw', default=1.2)) # map width
        self._mh = float(rospy.get_param('~mh', default=1.6)) # map height
        self._fw = fw=float(rospy.get_param('~fw', default=0.06)) # footprint width
        self._fh = fh=float(rospy.get_param('~fh', default=0.07)) # footprint height
        self._r0 = [[-fw/2,-fh/2],[-fw/2,fh/2],[fw/2,fh/2],[fw/2,-fh/2]]
        self._r0 = np.asarray(self._r0, dtype=np.float32).T #2x4

        map_img = cv2.imread(self._map_file, cv2.IMREAD_GRAYSCALE)
        self._viz = map_img
        _, self._map = cv2.threshold(map_img, 127, 255, cv2.THRESH_BINARY_INV)
        self._check_srv = rospy.Service('check_path', CheckPath, self.check_path_cb)
        self._move_srv = rospy.Service('move', Move, self.move_cb)
        self._clk_pub = rospy.Publisher('/clock', Clock, queue_size=1)

        # goal testing logic
        cv2.namedWindow('world')
        cv2.setMouseCallback('world', self.mouse_cb)

        self.cli_ = actionlib.SimpleActionClient('plan_to_goal', PlanToGoalAction)
        self.mode_ = None
        self.ang_  = None
        self.src_  = None
        self.dst_  = None
        self.loc_  = None

        self.t0_ = time.time()

    def mouse_cb(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONUP:
            if self.ang_ is None:
                print('Set Angle First!')
                return

            if self.mode_ is None:
                print('Set Mode First!')
                return

            if self.mode_ == 'src':
                x,y = self.uv2xy([x,y])
                self.src_ = [x,y,self.ang_]
                print('source : {}'.format(self.src_))
            elif self.mode_ == 'dst':
                x,y = self.uv2xy([x,y])
                self.dst_ = [x,y,self.ang_]
                print('target : {}'.format(self.dst_))

    def send_goal(self):
        if (self.src_ is None) or (self.dst_ is None):
            print('no goal to send : {} -> {}'.format(self.src_,self.dst_))

        self.cli_.wait_for_server()
        print('sending : {} -> {}'.format(self.src_, self.dst_))

        sx,sy,sh = self.src_
        gx,gy,gh = self.dst_
        goal = PlanToGoalGoal(
                start = Pose2D(sx,sy,sh),
                goal  = Pose2D(gx,gy,gh))
        self.cli_.send_goal(goal)
        self.loc_ = self.src_ # current location

        # reset data
        # self.ang_  = None
        # self.mode_ = None
        # self.src_  = None
        # self.dst_  = None

    def run(self):
        while not rospy.is_shutdown():
            # publish time
            now = time.time() - self.t0_
            t = rospy.Time(now)
            self._clk_pub.publish(t)

            cv2.imshow('world', self._viz)
            k = cv2.waitKey(10)
            if k == 27: break

            # handle point mode
            if k == ord('s'):
                print('src mode')
                self.mode_ = 'src'
            if k == ord('d'):
                print('dst mode')
                self.mode_ = 'dst'

            # handle point angular orientation
            if k == ord('y'): # al
                print('y-direction')
                self.ang_ = 0 
            if k == ord('x'):
                print('x-direction')
                self.ang_ = np.pi/2

            # send goal
            if k == ord('p'):
                self.send_goal()

            if cv2.waitKey(10) == 27:
                break
            time.sleep(0.01)

    def xy2uv(self, ps):
        # convert from physical coordinates to map coordinates
        # x -> v, y -> u
        x, y = ps
        n, m = self._map.shape[:2]
        mv = (n/2.0) + x*(float(n) / self._mw)
        mu = (m/2.0) + y*(float(m) / self._mh)
        mv = np.round(np.clip(mv, 0, n)).astype(np.int32)
        mu = np.round(np.clip(mu, 0, m)).astype(np.int32)
        return np.stack([mu,mv], axis=-1) # TODO : mu-mv? check order

    def uv2xy(self, ps):
        u, v = ps
        n, m = self._map.shape[:2]
        y = (u - (m/2.0)) * (self._mh / float(m))
        x = (v - (n/2.0)) * (self._mw / float(n))
        return np.stack([x,y], axis=-1)

    def move_cb(self, req):
        p = req.pose
        dst = [p.x, p.y, p.theta]
        src = self.loc_
        self.loc_ = dst

        hull_xy = fpt_hull(self._r0, src[:2], dst[:2],
                a0=src[2], a1=dst[2])
        hull_uv = [self.xy2uv(hull_xy)]
        cv2.drawContours(self._viz, hull_uv, 0, color=128, thickness=-1)
        return MoveResponse(success=True)

    def check_path_cb(self, req):
        p0 = req.pose_start
        p1 = req.pose_goal

        hull_xy = fpt_hull(self._r0, [p0.x,p0.y], [p1.x,p1.y],
                a0=p0.theta, a1=p1.theta)
        hull_uv = [self.xy2uv(hull_xy)]

        sweep_map = np.zeros_like(self._map)
        #print 'huv', hull_uv
        cv2.drawContours(sweep_map, hull_uv, 0, color=255, thickness=-1)
        #cv2.drawContours(self._viz, hull_uv, 0, color=128, thickness=-1)

        valid = (not np.any(np.logical_and(sweep_map, self._map)))
        return CheckPathResponse(valid=valid)

def main():
    rospy.init_node('min_sim')
    app = MinSim()
    app.run()

if __name__ == "__main__":
    main()

