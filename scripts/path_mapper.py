#! /usr/bin/env python

import rospy
import actionlib
from rll_planning_project.srv import *
from rll_planning_project.msg import *
from geometry_msgs.msg import Pose2D
from heapq import heappush, heappop # for priority queue
import math

import numpy as np
import cv2
from std_srvs.srv import Empty, EmptyResponse
from scipy.spatial import ConvexHull

def R(x):
    c,s = np.cos(x), np.sin(x)
    return np.asarray([[c,-s],[s,c]], dtype=np.float32) #2x2

def cast_ray(srv, p0, p1, a=0, min_d=0.005):
    """ Search ray collision point via simple binary search""" 
    p0p = Pose2D(x=p0[0], y=p0[1], theta=a)
    p1p = Pose2D(x=p1[0], y=p1[1], theta=a)

    # check p0 -> p1
    resp = srv(p0p, p1p)
    if resp.valid:
        # open
        return p1
    else:
        pm = np.add(p0, p1) / 2.0
        if np.linalg.norm(pm-p0) < min_d:
            # reached resolution limit
            return p0 # gives "conservative" estimate for free space
        # check p0 -> pm
        p0m = cast_ray(srv, p0, pm, a, min_d)
        
        if np.allclose(p0m, pm):
            # p0 - pm open, check pm -> p1
            return cast_ray(srv, pm, p1, a, min_d)
        else:
            return p0m

def xy2uv(xy, w, h, map):
    # convert from physical coordinates to map coordinates
    # x -> u, y -> v
    x, y = xy
    n, m = map.shape[:2]
    mv = (n/2.0) + x*(float(n) / w)
    mu = (m/2.0) + y*(float(m) / h)
    mv = np.round(np.clip(mv, 0, n)).astype(np.int32)
    mu = np.round(np.clip(mu, 0, m)).astype(np.int32)
    return np.stack([mu,mv], axis=0) # TODO : mu-mv? check order

def fpt_hull(fpt, p0, p1, a=0):
    r0 = R(a).dot(fpt) + np.reshape(p0, [2,1])
    r1 = R(a).dot(fpt) + np.reshape(p1, [2,1])
    ps = np.concatenate([r0,r1], axis=-1) # 2x8
    hull = ps.T[ConvexHull(ps.T).vertices].T# TODO: ps.T??
    return hull

def monte_carlo_checker(srv, map, fpt, w, h, a=0):
    px, py = np.random.uniform(low=[-w/2,-h/2], high=[w/2,h/2])
    p0 = [px,py]
    pms = []

    # check cross
    for p1 in [[px,-h/2], [px,h/2], [-w/2,py], [w/2,py]]:
        pm = cast_ray(srv, p0, p1,a=a)
        if not np.allclose(p0, pm):
            pms.append(pm)

    if len(pms) > 0: # == start pose was valid
        # fill trace
        print ':)'
        for pm in pms:
            trace = fpt_hull(fpt, p0, pm, a=a)
            print trace
            trace = xy2uv(trace, w, h, map)
            print 'drawing ... '
            print trace, trace.shape
            cv2.drawContours(map, [trace.T], 0, color=255, thickness=-1)

class GridMapper(object):
    def __init__(self, w, h, fw, fh, r=0.02):
        self._w = w
        self._h = h
        self._fw = fw
        self._fh = fh
        self._r = r

        #self._xs = np.linspace(-(w/2.0), w/2.0, num=np.ceil(w/r).astype(np.int32), endpoint=True)
        #self._ys = np.linspace(-(h/2.0), h/2.0, num=np.ceil(h/r).astype(np.int32), endpoint=True)

        # set to initial corner
        self._wlim = wlim = (w/2 - fw/2)
        self._hlim = hlim = (h/2 - fh/2)
        self._xpt = [-wlim + 0.02, -hlim + 0.02]
        self._ypt = [-wlim + 0.02, -hlim + 0.02]
        self._xfin = self._yfin = False

        # segment data
        self._xseg = []
        self._yseg = []

    def __call__(self, srv, map, fpt):
        rospy.loginfo_throttle(1.0,
                '{} : {}'.format(self._xpt, self._ypt))
        # unroll parameters / data
        xpt, ypt = self._xpt, self._ypt
        wlim, hlim = self._wlim, self._hlim
        w, h, r  = self._w, self._h, self._r

        # check x
        if(xpt[0] >= wlim):
            xpt[1] += r
            xpt[0] = -wlim
        if(xpt[1] >= hlim):
            self._xfin = True
        if not self._xfin:
            xm = cast_ray(srv, xpt, [wlim, xpt[1]], a=np.pi/2, min_d=0.005)

            if not np.allclose(xpt, xm):
                # mark ...
                trace = fpt_hull(fpt, xpt, xm, a=np.pi/2)
                trace = xy2uv(trace, w, h, map)
                cv2.drawContours(map, [trace.T], 0, color=255, thickness=-1)

                #save
                self._xseg.append(np.copy([xpt, xm]))

            # set to next point
            xpt[0] = max(xpt[0]+r, xm[0]+r)

        # check y
        if(ypt[1] >= hlim):
            ypt[0] += r
            ypt[1] = -hlim
        if(ypt[0] >= wlim):
            self._yfin = True
        if not self._yfin:
            ym = cast_ray(srv, ypt, [ypt[0], hlim], a=0., min_d=0.005)

            if not np.allclose(ypt, ym):
                # mark ...
                trace = fpt_hull(fpt, ypt, ym, a=0.)
                trace = xy2uv(trace, w, h, map)
                cv2.drawContours(map, [trace.T], 0, color=255, thickness=-1)

                # save
                self._yseg.append(np.copy([ypt, ym]))

            # set to next point
            ypt[1] = max(ypt[1]+r, ym[1]+r)

    def save(self):
        np.save('/tmp/segments.npy', [self._xseg, self._yseg])

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

        # grid checker handle
        self._grid_checker = GridMapper(mw, mh, fw, fh, r=0.02)

        # Provide Services
        self._step = rospy.Service('step', Empty, self.step_cb)
        self._stop = rospy.Service('stop', Empty, self.stop_cb)
        self.server = actionlib.SimpleActionServer("plan_to_goal", PlanToGoalAction, self.execute, False)
        self.server.start()

        rospy.on_shutdown(self._grid_checker.save)

    def step_cb(self, _):
        if not self._init_flag:
            return EmptyResponse()
        try:
            self._grid_checker(self._check_srv, self._map, self._fpt)
            #monte_carlo_checker(self._check_srv,
            #        self._map,self._fpt,
            #        self._mw, self._mh, a=0)#np.pi/2)
        except Exception as e:
            #rospy.logerr_throttle(1.0, 'Monte Carlo Mapper Failed : {}'.format(e))
            rospy.logerr_throttle(1.0, 'Grid Mapper Failed : {}'.format(e))
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

        while not self._stop_flag:
            #rate.sleep()
            self.step_cb(None)
            cv2.imshow('map', self._map)
            cv2.waitKey(max(1, delay_t))

        # after everything is done, pretend to be path_planner.py
        plan_to_goal(req)
        self.server.set_succeeded()

if __name__ == '__main__':
    rospy.init_node('path_planner')
    server = PathMapper()
    rospy.spin()
