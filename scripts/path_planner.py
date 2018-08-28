#! /usr/bin/env python

import rospy
import actionlib
from rll_planning_project.srv import *
from rll_planning_project.msg import *
from geometry_msgs.msg import Pose2D
from heapq import heappush, heappop # for priority queue
import math

import numpy as np
import os
from path_mapper import GridMapper
from seg_proc import SegProc
import cv2

def p2l(p):
    return [p.x,p.y,p.theta]

def angspace(a0, a1, num=5):
    if np.abs(a1 - a0) > np.pi:
        a1 += 2*np.pi*np.sign(a0-a1)
    # angular linspace, shortest path
    res = np.linspace(a0, a1, num=num)
    return ((res + np.pi) % (2*np.pi)) - np.pi

def angnorm(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

class LocalPathPlanner(object):
    def __init__(self):
        pass
    def __call__(self, srv, wpts, t0, t1):
        # stitch wpts = waypoints
        # and ensure that all intermediate paths are valid
        pose0 = Pose2D()
        pose1 = Pose2D()

        new_wpts = [] # include rotation transitions

        start = wpts[0] # immutable
        theta = t0

        for wi, wpt in enumerate(wpts):
            if wi==0: continue # avoid confusing wi index

            pose0.x, pose0.y = start
            pose0.theta = theta

            print 'planning from : {}'.format(p2l(pose0))
            delta = np.subtract(wpt, start)
            delta_dir = np.divide(delta, np.linalg.norm(delta))

            # if delta is in x direction, theta should be pi/2
            # else theta = 0
            if np.abs(delta[0]) > np.abs(delta[1]):
                # TODO: not a super robust check
                path_theta = np.pi/2 # +- np.pi/2
            else:
                path_theta = 0.0 # 0.0 or np.pi

            #if wi == (len(wpts) - 1):
            #    path_theta = t1 # try?

            success = False

            # method 1 : if simple path is permissible
            if not success:
                pose1.x, pose1.y = wpt
                pose1.theta = path_theta
                if srv(pose0, pose1).valid:
                    new_wpts.append( Pose2D(pose1.x,pose1.y,pose1.theta) )
                    success = True
                    print '[{}] : {} ; {} -> {}'.format(wi, 1, p2l(pose0), p2l(pose1))

            # method 1, but try again 180'
            if not success:
                path_theta = angnorm(path_theta + np.pi)
                pose1.theta = path_theta
                if srv(pose0, pose1).valid:
                    new_wpts.append( Pose2D(pose1.x,pose1.y,pose1.theta) )
                    success = True
                    print '[{}] : {} ; {} -> {}'.format(wi, 1, p2l(pose0), p2l(pose1))

            # method 2 : handle rotation first
            if not success:
                ad1 = np.abs(angnorm(pose0.theta - path_theta))
                ad2 = np.abs(angnorm(pose0.theta - (path_theta + np.pi)))
                wpts_m2 = []

                for _ in range(10): # number of total twiddle tries
                    if success: break

                    tw_suc = True # set to true in case rotation is not required
                    if ad1 > np.deg2rad(5.0) and ad2 > np.deg2rad(5.0): # rotation is required
                        rospy.loginfo('Attempting Twiddle')
                        rospy.loginfo('Twiddle Source : {}'.format(p2l(pose0)))
                        # seq : {twiddle - rotate} - {twiddle - move}
                        tw_suc = False

                        # options:
                        # 0.0 -> np.pi/2
                        # 0.0 -> -np.pi/2
                        # np.pi -> np.pi/2
                        # np.pi -> -np.pi/2

                        if not tw_suc:
                            tw_suc = True
                            wpts_m2 = []
                            thetas = angspace(pose0.theta, path_theta, num=5)[1:] #22.5,45.67.5,90

                            pose_r = Pose2D(pose0.x, pose0.y, pose0.theta)
                            for theta_r in thetas:
                                for i in range(100):
                                    sign = np.random.choice([-1,1], size=2, replace=True)
                                    zx, zy = sign * np.random.uniform(0.005, 0.02, size=2)
                                    posem_r = Pose2D(pose_r.x+zx, pose_r.y+zy, theta_r) # in-place rot
                                    if srv(pose_r, posem_r).valid:
                                        pose_r = posem_r
                                        wpts_m2.append( Pose2D(posem_r.x, posem_r.y, posem_r.theta) )
                                        break
                                else:
                                    tw_suc = False
                                    break # NOTE:breaks out of OUTER for loop!

                        if not tw_suc:
                            path_theta = angnorm(path_theta + np.pi)
                            tw_suc = True
                            wpts_m2 = []
                            thetas = angspace(pose0.theta, path_theta, num=5)[1:] #22.5,45.67.5,90

                            pose_r = Pose2D(pose0.x, pose0.y, pose0.theta)
                            for theta_r in thetas:
                                for i in range(100):
                                    sign = np.random.choice([-1,1], size=2, replace=True)
                                    zx, zy = sign * np.random.uniform(0.005, 0.02, size=2)
                                    posem_r = Pose2D(pose_r.x+zx, pose_r.y+zy, theta_r) # in-place rot
                                    if srv(pose_r, posem_r).valid:
                                        pose_r = posem_r
                                        wpts_m2.append( Pose2D(posem_r.x, posem_r.y, posem_r.theta) )
                                        break
                                else:
                                    tw_suc = False
                                    break # NOTE: breaks out of OUTER for loop!

                        print 'tw_suc', tw_suc
                    else:
                        # rotation is not required
                        posem_r = pose0

                    #pose_r = posem_r

                    #if srv(pose0, pose0_z).valid and srv(pose0_z, posem_r).valid:
                    #    wpts_m2.append( Pose2D(pose0_z.x, pose0_z.y, pose0_z.theta) )
                    #    wpts_m2.append( Pose2D(posem_r.x, posem_r.y, posem_r.theta) )
                    #    tw_suc = True
                    #    print 'Twiddle Part Success!'
                    #    break

                    if tw_suc:
                        print 'twiddled to ...', p2l(posem_r)

                        pose1.theta = path_theta

                        if srv(posem_r, pose1).valid:
                            wpts_m2.append( Pose2D(pose1.x, pose1.y, path_theta) )
                            new_wpts.extend(wpts_m2)
                            success = True
                        else:
                            for i in range(200):
                                #sign = np.random.choice([-1,1], size=2, replace=True)
                                #zx0, zy0 = sign * np.random.uniform(0.005, 0.03, size=2)
                                #posem_t = Pose2D(posem_r.x+zx0, posem_r.y+zx0, path_theta)
                                #if not srv(posem_r, posem_t).valid:
                                #    continue

                                sign = np.random.choice([-1,1], size=2, replace=True)
                                zx1, zy1 = sign * np.random.uniform(0.005, 0.02, size=2)
                                pose1_z = Pose2D(pose1.x+zx1, pose1.y+zy1, path_theta)

                                if srv(posem_r, pose1_z).valid:
                                    pose1 = pose1_z
                                    wpts_m2.append( Pose2D(pose1.x, pose1.y, path_theta) )
                                    new_wpts.extend(wpts_m2)
                                    success = True
                                    break
                    #if not success:
                    #    new_wpts.extend(wpts_m2) # debugging

                # seq : static_twiddle - rotate - move
                #dx, dy = delta_dir * 0.01
                #posem = Pose2D(pose0.x+dx, pose1.y+dy, path_theta)
                #posem2 = Pose2D(pose0.x, pose1.y, path_theta)
                #if srv(pose0, posem).valid and srv(posem, posem2).valid\
                #        and srv(posem2, pose1).valid:
                #    new_wpts.append( Pose2D(posem.x, posem.y, posem.theta) )
                #    new_wpts.append( Pose2D(posem2.x, posem2.y, posem2.theta) )
                #    new_wpts.append( Pose2D(pose1.x, pose1.y, pose1.theta) )
                #    success = True
                #    print '[{}] : {} ; {} -> {} -> {}'.format(wi, 2, p2l(pose0), p2l(posem), p2l(pose1))

            # method 3: sample random intermediate points until valid
            # NOTE : also samples endpoints at random offsets within +-2cm

            #if not success:
            #    rospy.loginfo('Attempting Random Intermediate Points')
            #    for i in range(100): # 100 tries [TODO : make configurable]
            #        dx, dy = delta * np.random.uniform()
            #        zxm, zym = np.random.uniform(-0.02, 0.02, size=2)
            #        zx1, zy1 = np.random.uniform(-0.02, 0.02, size=2)
            #        posem  = Pose2D(pose0.x+dx+zxm, pose1.y+dy+zym, path_theta)
            #        pose1z = Pose2D(pose1.x+zx1, pose1.y+zy1, path_theta)

            #        if srv(pose0, posem).valid and srv(posem, pose1z).valid:
            #            pose1 = pose1z
            #            new_wpts.append( Pose2D(posem.x, posem.y, posem.theta) )
            #            new_wpts.append( Pose2D(pose1.x, pose1.y, pose1.theta) )
            #            success = True
            #            print '[{}] : {} ; {} -> {} -> {}'.format(wi, 3, p2l(pose0), p2l(posem), p2l(pose1))
            #            break

            if not success:
                print 'failed to compute local path! \n\t src : \n {} \n\t dst \n {}'.format(pose0, pose1)
                print '{}/{}'.format(wi, len(wpts))
                # abort
                print new_wpts
                return new_wpts

            # reset starting point
            start = [pose1.x, pose1.y]
            theta = pose1.theta

        # handle endpoint theta
        pose0.x, pose0.y = start
        pose0.theta = theta
        d_theta = angnorm(theta - t1)
        if np.abs(d_theta) > np.deg2rad(5):
            # NOTE : somewhat arbitrary angle tolerance assignment
            for i in range(100):
                zx, zy = np.random.uniform(-0.02, 0.02, size=2)
                posem = Pose2D(pose0.x+zx, pose0.y+zy, t1)
                pose1 = Pose2D(pose0.x, pose0.y, t1)

                if srv(pose0, posem).valid and srv(posem, pose1).valid:
                    new_wpts.append( Pose2D(posem.x, posem.y, posem.theta) )
                    new_wpts.append( Pose2D(pose1.x, pose1.y, pose1.theta) )
                    break

        # even if the final endpoint-angle handling code fails,
        # still attempt the path to see if the result will be acceptable
        # (i.e. the code will not return [] as in previous failure cases)
        print 'new_wpts'
        print '==='
        for wpt in new_wpts:
            print p2l(wpt)
        print '==='

        return new_wpts

class PathManager(object):
    def __init__(self):
        # map params
        self._mw = mw = float(rospy.get_param('~map_width', default=1.2))
        self._mh = mh = float(rospy.get_param('~map_length', default=1.6))
        self._mr = mr = float(rospy.get_param('~map_res', default=0.005))
        n, m = int(np.round(mw/mr)), int(np.round((mh/mr)))
        self._map = np.zeros(shape=(n,m), dtype=np.uint8)

        # footprint params (not really used during runtime; only for testing)
        self._fw = fw = float(rospy.get_param('~fw', default=0.06)) # footprint width
        self._fh = fh = float(rospy.get_param('~fh', default=0.07)) # footprint height
        fpt = [[-fw/2,-fh/2],[-fw/2,fh/2],[fw/2,fh/2],[fw/2,-fh/2]] # 4x2
        self._fpt = np.asarray(fpt, dtype=np.float32).T #2x4

        # misc params
        self._rate = float(rospy.get_param('~rate', default=50.0))

        # Flag
        self._init_flag = False
        self._stop_flag = False

        # ROS Handle
        self.server = actionlib.SimpleActionServer("plan_to_goal", PlanToGoalAction, self.execute, False)
        self.server.start()


    def execute(self, req):
        # Testing Only!
        viz = False

        # create service handles
        check_srv = rospy.ServiceProxy('check_path', CheckPath, persistent=True)
        move_srv  = rospy.ServiceProxy('move', Move)

        # phase 1 : mapping
        # grid_mapper = GridMapper(self._mw, self._mh, self._fw, self._fh, r=0.02)
        # if viz:
        #     cv2.namedWindow('map', cv2.WINDOW_NORMAL)
        # while not (grid_mapper.done()):
        #     try:
        #         grid_mapper(check_srv, self._map, self._fpt)
        #     except Exception as e:
        #         rospy.logerr_throttle(1.0, 'Grid Mapper Failed : {}'.format(e))
        #     if viz:
        #         cv2.imshow('map', self._map)
        #         cv2.waitKey(1)
        ## grid_mapper.save()

        #xseg, yseg = grid_mapper._xseg, grid_mapper._yseg
        #xseg = np.asarray(xseg, dtype=np.float32)
        #yseg = np.asarray(yseg, dtype=np.float32)
        #if len(xseg) <= 0 or len(yseg) <= 0:
        #    rospy.logerr_throttle(1.0, 'Grid Mapper Failed!')
        #    return

        # alt 1: use cached segments, NOTE : only for testing!!
        data_path = os.path.expanduser('/tmp/segments.npy')
        xseg, yseg = np.load(data_path)

        # phase 2 : process path segments --> graph + plan
        pose0 = req.start
        pose1 = req.goal
        t0,t1 = pose0.theta, pose1.theta
        extra_pts = [[pose0.x, pose0.y], [pose1.x, pose1.y]]
        extra_pts = np.asarray(extra_pts, dtype=np.float32)
        proc = SegProc(xseg, yseg, pts=extra_pts)
        path = np.copy(proc._path)
        if len(path) <= 0:
            rospy.logerr_throttle(1.0, 'Global Planner Failed!')
            return

        #print 'Validate Initial Points : ', path[0], req.start

        # phase 3 : construct executable local paths from global path
        local_planner = LocalPathPlanner()
        full_path = local_planner(check_srv, path, t0, t1)
        if len(full_path) <= 0:
            rospy.logerr_throttle(1.0, 'Local Planner Failed!')
            return

        # phase 4 : move
        for wi, wpt in enumerate(full_path):
            print 'wpt', p2l(wpt)
            if wi>0:
                print 'validity : ', check_srv(full_path[wi-1], wpt)
            move_srv(wpt)

        self.server.set_succeeded()


#def plan_to_goal(req):
#    """ Plan a path from Start to Goal """
#    pose_start = Pose2D()
#    pose_goal = Pose2D()
#    pose_check_start = Pose2D()
#    pose_check_goal = Pose2D()
#    pose_move = Pose2D()
#
#    rospy.loginfo("Got a planning request")
#
#    pose_start = req.start
#    pose_goal = req.goal
#
#    move_srv = rospy.ServiceProxy('move', Move)
#    check_srv = rospy.ServiceProxy('check_path', CheckPath, persistent=True)
#
#    ###############################################
#    # Implement your path planning algorithm here #
#    ###############################################
#
#    # Input: map dimensions, start pose, and goal pose
#    # retrieving input values  
#    map_width = rospy.get_param('~map_width')
#    map_length = rospy.get_param('~map_length')
#    xStart, yStart, tStart = pose_start.x, pose_start.y, pose_start.theta
#    xGoal, yGoal, tGoal = pose_goal.x, pose_goal.y, pose_goal.theta
#    # printing input values
#    rospy.loginfo("map dimensions: width=%1.2fm, length=%1.2fm", map_width, map_length)
#    rospy.loginfo("start pose: x %f, y %f, theta %f", xStart, yStart, tStart)
#    rospy.loginfo("goal pose: x %f, y %f, theta %f", xGoal, yGoal, tGoal)
#
#    # Output: movement commands
#    pose_check_start.x, pose_check_start.y, pose_check_start.theta= xStart, yStart, tStart
#    pose_check_goal.x, pose_check_goal.y, pose_check_goal.theta= xGoal, yGoal, tGoal
#    resp = check_srv(pose_check_start, pose_check_goal) # checking if the arm can move to the goal pose
#    if resp.valid:
#        rospy.loginfo("Valid pose")
#        pose_move.x, pose_move.y, pose_move.theta = xGoal, yGoal, tGoal 
#        # executing a move command towards the goal pose
#        resp = move_srv(pose_move)
#    else:
#        rospy.loginfo("Invalid pose")
#        
#    ###############################################
#    # End of Algorithm #
#    ###############################################
#
#
#class PathPlanner:
#    def __init__(self):
#        self.server = actionlib.SimpleActionServer("plan_to_goal", PlanToGoalAction, self.execute, False)
#        self.server.start()
#
#    def execute(self, req):
#        plan_to_goal(req)
#        self.server.set_succeeded()
#
#

if __name__ == '__main__':
    rospy.init_node('path_planner')
    #server = PathPlanner()
    server = PathManager()
    rospy.spin()
