#!/usr/bin/env python

viz_enabled = True
import numpy as np
import utils as U
from map_utils import cast_ray
from seg_utils import segment_intersect

try:
    import cv2
    from scipy.spatial import ConvexHull
except ImportError as e:
    print('cv2/scipy import failed : {}'.format(e))
    print('visualization disabled.')
    viz_enabled = False

class SkelMapper(object):
    """
    Perform specialized skeletal mapping from start to goal.
    Requires source and destination points for search initialization.
    """
    def __init__(self, w, h, fw, fh, r=0.02):
        self._w = w
        self._h = h
        self._fw = fw
        self._fh = fh
        self._r = r

        # set to initial corner
        self._wlim = wlim = (w/2 - fw/2)
        self._hlim = hlim = (h/2 - fh/2)

        # set all data fields to None
        self.done_ = False
        self.path_ = None
        self.seg_s_ = None
        self.seg_g_ = None
        self.s_tree_ = None
        self.g_tree_ = None
        self.si0_ = None
        self.si1_ = None
        self.gi0_ = None
        self.gi1_ = None

    def lim_dir(self, src, ang):
        """
        find extreme endpoints from source position and orientation.

        Args:
            src(array): source point, [2] array formatted (x,y)
            ang(float): point orientation for expansion
        """
        wlim, hlim = self._wlim, self._hlim

        x, y = src
        # TODO : check xy-segment orientation
        if np.allclose(np.cos(ang), 0): # x   
            return [0, y], [wlim, y]
        else:
            return [x, 0], [x, hlim]

    def expand(self, srv, src, ang):
        """ find actual endpoints from source position and orientation.

        Args:
            srv(function): see cast_ray argument for reference.
            src(array): source point, [2] array formatted (x,y)
            ang(float): point orientation for expansion
        """
        x,y = src
        p0, p1 = self.lim_dir(src, ang)
        pm0 = cast_ray(srv, src, p0, a=ang, min_d=0.005)
        pm1 = cast_ray(srv, src, p1, a=ang, min_d=0.005)
        return [pm0, pm1]

    def search(self, srv, seg, skip=None, pad=0.0, min_l=0.02):
        """ search along segment for orthogonal segments.

        Note:
            search does not exhaustively exclude already discovered segments.
            take care not to process segments twice, which would result in a loop.

        Args:
            srv(function): see cast_ray argument for reference.
            seg(np.ndarray): source segment, [2,2] array formatted (x,y)
            skip(np.ndarray): skip point, [2] array formatted (x,y) (default:None)
            pad(float): distance to pad segment for search (default:0)
            min_l(float): minimum segment length to return (default:0.02)

        Returns:
            segs(list): list of discovered segments
        """
        segs = []

        d = seg[1] - seg[0]
        l = np.linalg.norm(d)
        d /= l

        # set search angle to be  perpendicular to source segment
        ang = U.anorm(np.arctan2(d[1], d[0]) + np.pi/2)

        # define search steps from seg[0] to seg[1] with resolution _r
        steps = d.reshape(1,2) * np.arange(-pad,l+pad,step=self._r).reshape(-1,1)

        # seg[:1] = (1,2)
        # steps = (N,2)
        srcs = seg[:1] + steps # = (N,2)

        if not np.allclose(srcs[-1], seg[1]):
            # account for endpoints since they tend to be important
            srcs = np.concatenate(srcs, [seg[1]], axis=0)

        for src in srcs:
            if (skip is not None) and np.allclose(src, skip):
                # skip source
                continue
            new_seg = self.expand(srv, src, ang)
            l_seg = np.linalg.norm(new_seg[1] - new_seg[0])
            if l_seg > min_l:
                segs.append(new_seg)

        return segs

    @staticmethod
    def check_suc(seg0s, seg1s):
        """ Check if latest source segments intersect with any of target segments.

        Args:
            seg0s(np.ndarray): [N,2,2] list of segments formatted (x,y)
            seg1s(list): [M,2,2] list of list of segments formatted (x,y).

        Returns:
            (suc, data)
            suc : True if done, else False
            data : None if not done, else (i,j),
                where seg0s[i] and seg1s[j] intersected.
        """
        for i, seg0 in enumerate(seg0s):
            for j, seg1s in enumerate(seg1s):
                if segment_intersect(seg0, seg1):
                    return True, (i,j)

        return False, None

    @staticmethod
    def trace_path(s_tree, g_tree, si, gi):
        # TODO : implement
        raise NotImplementedError("Tree based Path Tracing not supported yet")

    def reset(self, srv, src, dst):
        """ Initialize mapping.

        Args:
            srv(function): see cast_ray argument for reference.
            src(array): source point, [3] array formatted (x,y,h)
            dst(array): target point, [3] array formatted (x,y,h)
        """

        # TODO : current mapper probably doesn't handle when src==dst
        x0, y0, h0 = src
        x1, y1, h1 = dst

        self.seg_s_ = [self.expand(srv, src, h0)]
        self.seg_g_ = [self.expand(srv, dst, h1)]

        self.s_tree_ = {} # (i) -> [j,k,l], 
        self.g_tree_ = {}

        self.si0_, self.si1_ = 0, 1
        self.gi0_, self.gi1_ = 0, 1

    def __call__(self, srv, map, fpt, viz=False, log=None):
        """ Single step """
        viz = (viz and viz_enabled) # check global viz flag

        if log is not None:
            log('{} : {}'.format(self._xpt, self._ypt))

        self.reset(srv, src, dst)

        # unroll parameters
        seg_s, seg_g = self.seg_s_, self.seg_g_
        s_tree, g_tree = self.s_tree_, self.g_tree_
        si0, si1 = self.si0_, self.si1_
        gi0, gi1 = self.gi0_, self.gi1_

        # check success first
        suc, res = self.check_suc(seg_s[si0:si1], seg_g)
        if suc:
            si, gi = res
            self.path_ = self.trace_path(s_tree_, g_tree_, si, gi)
            self.done_ = True
            return

        # search for new segments, branching from old ones
        for seg in seg_s[si0:si1]:
            new_segs = self.search(seg, skip=None, pad=0.0, min_l=0.02)
            seg_s.extend(new_segs) # in-place update

        for seg in seg_g[gi0:gi1]:
            new_segs = self.search(seg, skip=None, pad=0.0, min_l=0.02)
            seg_g.extend(new_segs) # in-place update

        # update segment indexing range
        si0, gi0 = si1, gi1
        si1, gi1 = len(seg_s), len(seg_g)

        # update index cache variables
        self.si0_, self.si1_ = si0, si1
        self.gi0_, self.gi1_ = gi0, gi1

        if (si0 == si1) and (gi0 == gi1):
            # did not find any new segments,
            # meaning no further progress can be made
            self.path_ = []
            self.done_ = True

    def done(self):
        return self.done_

    def save(self, path='/tmp/skel_map.npy'):
        np.save(path, [self.path_, self.s_tree_, self.g_tree_])

def main():
    mapper = SkelMapper(1.2,1.6,0.06,0.07,r=0.02)
    mapper.reset(srv, src, dst)

if __name__ == "__main__":
    main()
