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
        self._srv = rospy.Service('check_path', CheckPath, self.check_path_cb)
        self._clk_pub = rospy.Publisher('/clock', Clock, queue_size=1)

        self.t0_ = time.time()

    def run(self):
        while not rospy.is_shutdown():
            # publish time
            now = time.time() - self.t0_
            t = rospy.Time(now)
            self._clk_pub.publish(t)

            cv2.imshow('viz', self._viz)
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

    def check_path_cb(self, req):
        p0 = req.pose_start
        p1 = req.pose_goal

        hull_xy = fpt_hull(self._r0, [p0.x,p0.y], [p1.x,p1.y],
                a0=p0.theta, a1=p1.theta)
        hull_uv = [self.xy2uv(hull_xy)]

        sweep_map = np.zeros_like(self._map)
        print 'huv', hull_uv
        cv2.drawContours(sweep_map, hull_uv, 0, color=255, thickness=-1)
        cv2.drawContours(self._viz, hull_uv, 0, color=128, thickness=-1)

        valid = (not np.any(np.logical_and(sweep_map, self._map)))
        return CheckPathResponse(valid=valid)

def main():
    rospy.init_node('min_sim')
    app = MinSim()
    app.run()

if __name__ == "__main__":
    main()

