#!/usr/bin/env python2
import cv2
import numpy as np

import rospy
from rll_planning_project.srv import *
from rll_planning_project.msg import *
from scipy.spatial import ConvexHull

def R(x):
    c,s = np.cos(x), np.sin(x)
    return np.asarray([[c,-s],[s,c]], dtype=np.float32) #2x2

class MinSim(object):
    """ Minified Simulation of Planning Project """
    def __init__(self):
        # get params
        self._map_file = rospy.get_param('~map_file', default='rll_map.png')
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

    def run(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            cv2.imshow('viz', self._viz)
            cv2.waitKey(10)
            rate.sleep()

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

        r0 = R(p0.theta).dot(self._r0) + np.reshape([p0.x,p0.y], [2,1])
        r1 = R(p1.theta).dot(self._r0) + np.reshape([p1.x,p1.y], [2,1])
        ps = np.concatenate([r0,r1], axis=-1) # 2x8

        #print 'ps', ps
        hull_xy = ps.T[ConvexHull(ps.T).vertices].T# TODO: ps.T??
        #print 'hxy', hull_xy
        hull_uv = [self.xy2uv(hull_xy)]
        #print 'huv', hull_uv

        sweep_map = np.zeros_like(self._map)
        print 'huv', hull_uv
        cv2.drawContours(sweep_map, hull_uv, 0, color=255, thickness=-1)
        cv2.drawContours(self._viz, hull_uv, 0, color=128, thickness=-1)

        valid = (not np.any(np.logical_and(sweep_map, self._map)))
        #print 'valid', valid
        return CheckPathResponse(valid=valid)

def main():
    rospy.init_node('min_sim')
    app = MinSim()
    app.run()

if __name__ == "__main__":
    main()

