#!/usr/bin/env python

viz_enabled = True
import numpy as np
import utils as U
from map_utils import cast_ray, fpt_hull
from seg_utils import segment_intersect, seg_join, seg_str, pt_str, s2s_ixt

try:
    import cv2
    from scipy.spatial import ConvexHull
except ImportError as e:
    print('cv2/scipy import failed : {}'.format(e))
    print('visualization disabled.')
    viz_enabled = False

def seg_h(seg):
    pa,pb = seg
    delta = pb-pa
    return np.arctan2(delta[1],delta[0])

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
        self.path_ = []
        self.wpts_ = []
        self.src_ = None
        self.dst_ = None
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
        if np.allclose(np.cos(ang), 1.0): # == along y
            # fix y, x-directional segment
            return np.float32([[x, -hlim], [x, hlim]])
        else:
            # fix x, y-directional segment
            return np.float32([[-wlim, y], [wlim, y]])

    def expand(self, srv, src, ang):
        """ find actual endpoints from source position and orientation.

        Args:
            srv(function): see cast_ray argument for reference.
            src(array): source point, [2] array formatted (x,y)
            ang(float): point orientation for expansion
        """
        x,y = src
        p0, p1 = self.lim_dir(src, ang)
        #print('src|p0-p1 : {}|{}-{}'.format(src,p0,p1))
        pm0 = cast_ray(srv, src, p0, a=ang, min_d=0.005)
        pm1 = cast_ray(srv, src, p1, a=ang, min_d=0.005)
        #print('src|pm0-pm1 : {}|{}-{}'.format(src,pm0,pm1))
        return np.asarray([pm0, pm1], dtype=np.float32)

    def search(self, srv, seg, skip=None, pad=0.05, min_l=0.08, min_d=0.02):
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
        # TODO : maybe min_l is why it fails?
        print('currently considering {}'.format(seg_str(seg)))
        segs = []

        d = seg[1] - seg[0]
        l = np.linalg.norm(d)
        if np.isclose(l, 0):
            # seg[1] == seg[0], single-point
            return []
        d /= l

        # set search angle to be  perpendicular to source segment
        ang = U.anorm(np.arctan2(d[1], d[0]))

        # define search steps from seg[0] to seg[1] with resolution _r
        steps = np.arange(-pad,l+pad,step=self._r).reshape(-1,1)
        diffs = d.reshape(1,2) * steps

        # seg[:1] = (1,2)
        # steps = (N,2)
        srcs = seg[:1] + diffs # = (N,2)

        #for src in srcs:
        #    print('src : {}'.format(pt_str(src)))

        #if not np.allclose(srcs[-1], seg[1]):
        #    # account for endpoints since they tend to be important
        #    srcs = np.concatenate( (srcs, [seg[1]]), axis=0)
        #    steps = np.concatenate( (srcs, 

        s_0 = None
        s_1 = None
        new_seg = None

        for src in srcs:
            #if (skip is not None) and np.allclose(src, skip):
            #    # skip source
            #    continue
            new_seg = self.expand(srv, src, ang)
            # print(seg_str(new_seg))

            l_seg = np.linalg.norm(new_seg[1] - new_seg[0])

            # filter by length
            if l_seg > min_l:
                # filter by previous seg.
                if s_0 is None:
                    s_0 = new_seg
                    s_1 = new_seg
                else:
                    s_j = seg_join(s_1, new_seg)
                    if s_j is not None:
                        s_1 = s_j
                    else:
                        prv_seg = s_1 #np.mean([s_0,s_1], axis=0)
                        segs.append(prv_seg)
                        # form new segment
                        s_0 = new_seg
                        s_1 = new_seg

        if s_1 is not None:
            segs.append(s_1)

        print('len(segs) : {}'.format(len(segs)))
        print('segs : {}'.format(segs))

        l = len(segs)
        mask = [False for _ in range(l)]
        new_segs = []

        for i in range(l):
            s0 = segs[i]
            if mask[i]: continue
            mask[i] = True
            for j in range(i+1,l):
                if mask[j]: continue
                s1 = segs[j]
                s2 = seg_join(s0,s1)
                if s2 is not None:
                    s0 = s2
                    mask[j] = True # j already used
            new_segs.append(s0)
        return new_segs

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
            for j, seg1 in enumerate(seg1s):
                if segment_intersect(seg0, seg1):
                    return True, (i, j)

        return False, None

    @staticmethod
    def trace_path(s_tree, g_tree, si, gi, seg_s, seg_g):
        # TODO : implement

        s_path = []
        while not (si == 0): #backtracking
            s_path.append(seg_s[si])
            si = s_tree[si]
        s_path.append(seg_s[si])
        s_path = s_path[::-1] #reverse

        g_path = []
        while not (gi == 0): #backtracking
            g_path.append(seg_g[gi])
            gi = g_tree[gi]
        g_path.append(seg_g[gi])
        # no need to reverse the goal segment

        s = seg_join(s_path[-1], g_path[0])

        if seg_join(s_path[-1], g_path[0]) is not None:
            # last src and first goal path are the same
            # path = np.concatenate([s_path[:-1], [s], g_path[1:]], axis=0)
            path = np.concatenate([s_path, g_path], axis=0)
        else:
            path = np.concatenate([s_path, g_path], axis=0)
        return path

    def refine_path(self, path, src, dst, log):
        # mostly, trim path and produce waypoints
        new_path = [src]
        prv = src
        for s0, s1 in zip(path[:-1], path[1:]):
            # requires tolerance here since footprint is not a square.
            wpt = s2s_ixt(s0,s1,as_pt=True,tol=3e-2)
            if wpt is None:
                log("Path Reconstruction Failed : {}-{}".format(s0,s1))
                return []

            # figure out orientation ...
            new_path.append(wpt)
        new_path.append(dst)

        return np.asarray(new_path, dtype=np.float32)

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

        self.src_ = [x0,y0]
        self.dst_ = [x1,y1]

        self.seg_s_ = [self.expand(srv, [x0,y0], h0)]
        self.seg_g_ = [self.expand(srv, [x1,y1], h1)]

        self.s_tree_ = {} # s_tree[child_idx] = parent_idx
        self.g_tree_ = {}

        self.si0_, self.si1_ = 0, 1
        self.gi0_, self.gi1_ = 0, 1

    @staticmethod
    def null_log(*args, **kwargs):
        return

    def __call__(self, srv, map, fpt, viz=False, log=None):
        """ Single step """
        viz = (viz and viz_enabled) # check global viz flag
        if log is None:
            log=self.null_log

        # unroll parameters
        seg_s, seg_g = self.seg_s_, self.seg_g_
        s_tree, g_tree = self.s_tree_, self.g_tree_
        si0, si1 = self.si0_, self.si1_
        gi0, gi1 = self.gi0_, self.gi1_

        # check success first
        suc, res = self.check_suc(seg_s[si0:si1], seg_g)
        if suc:
            si, gi = res
            si += si0 # account for offset!!
            log('Intersect indices : {},{}'.format(si, gi))
            self.path_ = self.trace_path(
                    s_tree, g_tree,
                    si, gi,
                    seg_s, seg_g
                    )
            self.wpts_ = self.refine_path(self.path_,
                    self.src_, self.dst_, log)
            self.done_ = True
            log("success!")
            return

        log('{}'.format(len(seg_s)))

        # search for new segments, branching from old ones
        for si in range(si0, si1):
            seg = seg_s[si]
            new_segs_raw = self.search(srv, seg, skip=None)
            len_new = len(new_segs_raw)

            # filter by previously found segments
            new_segs = []
            for s0 in new_segs_raw:
                for s1 in seg_s[:si0]:
                    if seg_join(s0,s1) is not None:
                        break
                else:
                    new_segs.append(s0)

            log('new_s: {}'.format(new_segs))
            i0 = len(seg_s)
            seg_s.extend(new_segs) # in-place update
            i1 = len(seg_s)
            for i in range(i0, i1):
                s_tree[i] = si

        for gi in range(gi0, gi1):
            seg = seg_g[gi]
            new_segs_raw = self.search(srv, seg, skip=None)
            len_new = len(new_segs_raw)

            # filter by previously found segments
            new_segs = []
            for s0 in new_segs_raw:
                for s1 in seg_g[:gi0]:
                    if seg_join(s0,s1) is not None:
                        break
                else:
                    new_segs.append(s0)

            log('new_g: {}'.format(new_segs))
            i0 = len(seg_g)
            seg_g.extend(new_segs) # in-place update
            i1 = len(seg_g)
            for i in range(i0, i1):
                g_tree[i] = gi

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

        if viz:
            w,h = self._w, self._h
            n,m = map.shape[:2]
            i = 0
            for segs in [seg_s, seg_g]:
                for seg in segs:
                    i += 1
                    print i
                    delta = (seg[1] - seg[0])
                    theta = np.arctan2(delta[1], delta[0])
                    trace = fpt_hull(fpt, seg[0], seg[1], a0=theta)
                    trace = U.xy2uv(trace, w, h, n, m)
                    cv2.drawContours(map, [trace.T], 0, color=255, thickness=-1)
        print('step')

    def done(self):
        return self.done_

    def save(self, path='/tmp/skel_map.npy'):
        print self.seg_s_
        print self.seg_g_
        np.save(path, [self.path_, self.wpts_, self.s_tree_, self.g_tree_, self.seg_s_, self.seg_g_])

def main():
    mapper = SkelMapper(1.2,1.6,0.06,0.07,r=0.02)
    mapper.reset(srv, src, dst)

if __name__ == "__main__":
    main()
