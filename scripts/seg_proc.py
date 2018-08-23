#!/usr/bin/env python2

import os
import numpy as np
import cv2

def dist_p2l(p, l):
    # p = (x,y)
    # l = [(x0,y0), (x1,y1)]
    p1, p2 = l
    d = np.abs(np.cross(p2-p1, p1-p))/np.linalg.norm(p2-p1)
    return d

class Graph(object):
    def __init__(self, nodes={}, edges={}):
        self._nodes = nodes # == [(x,y)]
        self._edges = edges # == [(i0,i1)]

    @staticmethod
    def from_segments(xseg, yseg):
        pass

    def build(self):
        # add distance information to edges
        pass

    def match_node(self, pt, tol=5e-2):
        for i, n in enumerate(self._nodes):
            if np.linalg.norm(np.subtract(n,pt)) < tol:
                return i
        return None

    def match_edge(self, pt):
        # warning : match_edge WILL modify edges to accomodate nodes.
        for (ep0,ep1) in self._edges:
            pass
        # TODO : impl.

    def add_node(self, pt):
        # searches matching nodes if possible,
        # splits edges if necessary
        # returns : (success, node_index)

        # match node first
        n_i = self.match_node(pt)
        if n_i is not None:
            return True, n_i

        # match edge
        n_i = self.match_edge(pt)

    def dijkstra(self, i0, i1):
        pass

    def plan(self, start, goal):
        # uses dijkstra algorithm to plan path
        i0 = self.add_node(start)
        if i0 is None:
            return []

        i1 = self.add_node(goal)
        if i1 is None:
            return []

        ix = self.dijkstra(i0, i1) # astar?
        if ix is None:
            return []

        # successful plan!
        return [i0, ix, i1]

class SegProc(object):
    def __init__(self, xseg, yseg, w=1.2, h=1.6, r=0.005, raw=False):
        if raw:
            self._xseg = xseg
            self._yseg = yseg
        else:
            self._xseg = self.prune(xseg) #self.prune(xseg)
            self._xseg = self.merge_x(self._xseg, dtol=8e-2, gtol=5e-2)
            self._yseg = self.prune(yseg)
            #self._yseg = self._yseg[np.logical_and(0.5<self._yseg[:,0,0], self._yseg[:,0,0]<1.0)]
            self._yseg = self.merge_y(self._yseg, dtol=7e-2, gtol=5e-2)#yseg #selg.prune(yseg)

        self._joints = self.join_xy(self._xseg, self._yseg)

        self._seg = np.concatenate([self._xseg, self._yseg],axis=0)
        self._seg_i = 0
        self._seg_n = len(self._seg)

        # map parameters
        self._mw = w
        self._mh = h
        self._r = r

        # create map for visualization
        n = np.ceil(w/r).astype(np.int32)
        m = np.ceil(h/r).astype(np.int32)
        self._map = np.zeros(shape=(n,m), dtype=np.float32)

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

    def prune(self, seg, tol=5e-2):
        seg = np.asarray(seg, dtype=np.float32)
        l   = np.linalg.norm(np.subtract(seg[:,0], seg[:,1]), axis=-1)
        return seg[l>tol]

    def merge_x(self, seg, dtol=1e-2, gtol=5e-2):
        n = len(seg)
        seg = np.copy(seg) # make a copy
        select = np.full(n, False, dtype=np.bool)
        merged = np.full(n, False, dtype=np.bool)
        change = False

        for i in range(n):
            if merged[i]:
                continue
            merged[i] = True
            select[i] = True
            for j in range(i+1, n):
                if merged[j]:
                    continue
                (ix0, iy), (ix1, _) = seg[i]
                (jx0, jy), (jx1, _) = seg[j]

                nd = np.abs(iy - jy)
                x0 = min(ix0, jx0)
                x1 = max(ix1, jx1)
                gd = (x1-x0) - (ix1-ix0) - (jx1-jx0)

                if nd<dtol and gd<gtol:
                    y = (iy + jy) / 2.0
                    seg[i][:,:] = [[x0,y],[x1,y]]
                    merged[j] = True
                    change = True

        if change:
            return self.merge_x(seg[select], dtol, gtol)
        else:
            return seg[select]
    
    def merge_y(self, seg, dtol=1e-1, gtol=5e-2):
        # TODO : avoid duplicate code
        n = len(seg)
        seg = np.copy(seg) # make a copy
        select = np.full(n, False, dtype=np.bool)
        merged = np.full(n, False, dtype=np.bool)
        change = False

        for i in range(n):
            if merged[i]:
                continue
            merged[i] = True
            select[i] = True
            for j in range(i+1, n):
                if merged[j]:
                    continue
                (ix, iy0), (_, iy1) = seg[i]
                (jx, jy0), (_, jy1) = seg[j]

                nd = np.abs(ix - jx)
                y0 = min(iy0, jy0)
                y1 = max(iy1, jy1)
                gd = (y1-y0) - (iy1-iy0) - (jy1-jy0)

                if nd<dtol and gd<gtol:
                    x = (ix + jx) / 2.0
                    seg[i][:,:] = [[x,y0],[x,y1]]
                    merged[j] = True
                    change = True

        if change:
            return self.merge_y(seg[select], dtol, gtol)
        else:
            return seg[select]

    def join_xy(self, xseg, yseg, dtol=5e-2):

        raw_joints = []

        # join x-y segments
        for xi, xx in enumerate(xseg):
            for yi, yy in enumerate(yseg):
                px0, px1 = xx
                py0, py1 = yy
                x, y = [py0[0], px0[1]] # point of intersection

                in_x = (px0[0] < x < px1[0])
                in_y = (py0[1] < y < py1[1])
                dx = np.min(np.abs( [px0[0]-x, x-px1[0]] ))
                dy = np.min(np.abs( [py0[1]-y, y-py1[1]] ))

                x_good = (in_x or (dx < dtol))
                y_good = (in_y or (dy < dtol))

                #print dx, dy

                if x_good and y_good:
                    raw_joints.append([x,y,xi,yi])

        # process additional points
        #nodes = [{'x':e[0], 'y':e[1], 'c':[]} for e in enumerate(raw_joints)]

        #j_n = len(raw_joints)
        #for i0 in range(j_n):
        #    j0 = raw_joints[i0]
        #    for i1 in range(i0+1, j_n):
        #        j1 = raw_joints[i1]
        #        if j0[2] == j1[2] or j0[3] == j1[3]: # on same segment


        #for j0 in raw_joints:
        #    for j1 in raw_joints:

        # add segment endpoints
        joints = np.asarray(raw_joints)[:, :2]
        joints = np.asarray(joints, dtype=np.float32) # == (N_J, 2)
        d_xs = np.reshape(joints, [-1,1,2]) - np.reshape(xseg, [1,-1,2]) # == (N_J, 2*N_X, 2)
        d_xs = np.linalg.norm(d_xs, axis=-1) # == (N_J, 2*N_X)
        xsel  = np.all(d_xs > dtol, axis=0)
        j_x  = np.reshape(xseg, [-1,2])[xsel]

        d_ys = np.reshape(joints, [-1,1,2]) - np.reshape(yseg, [1,-1,2]) # == (N_J, 2*N_X, 2)
        d_ys = np.linalg.norm(d_ys, axis=-1) # == (N_J, 2*N_X)
        ysel  = np.all(d_ys > dtol, axis=0)
        j_y  = np.reshape(yseg, [-1,2])[ysel]

        joints = np.concatenate([joints, j_x, j_y], axis=0)
        return joints

        # kind of inefficient (but probably doesn't matter)
        # new_xseg = []
        # new_yseg = []
        # for j in joints:
        #     x, y = j
        #     for xx in xseg:
        #         px0, px1 = xx

        #         in_x = (px0[0] < x < px1[0])
        #         dx = np.min(np.abs( [px0[0]-x, x-px1[0]] ))
        #         x_good = (in_x or (dx < dtol))

        #         if in_x:
        #             new_xseg.append([[px0[0],y], [x,y]])
        #             new_xseg.append([[x,y], [px1[0],y]])
        #         else:
        #             new_xseg.append([[px0[0],y], [px1[0],y]])

        #     for yy in yseg:
        #         py0, py1 = yy

        #         in_y = (py0[1] < y < py1[1])
        #         dy = np.min(np.abs( [py0[1]-y, y-py1[1]] ))
        #         y_good = (in_y or (dy < dtol))

        #         if in_y:
        #             new_yseg.append([[x,py0[1]], [x,y]])
        #             new_yseg.append([[x,y], [x,py1[1]]]) 
        #         else:
        #             new_yseg.append([[x,py0[1]], [x,py1[1]]])

        # new_xseg = np.asarray(new_xseg, dtype=np.float32)
        # new_yseg = np.asarray(new_yseg, dtype=np.float32)
        # return joints, new_xseg, new_yseg

    #def merge_hv(self, seg, tol=1e-2):
    #    # merge horizontal or vertical segments
    #    i = 0
    #    n = len(seg)
    #    new_seg = []
    #    merged = []

    #    for i in range(n):
    #        if i in merged:
    #            continue
    #        for j in range(i+1, n):
    #            if j in merged:
    #                continue
    #            s0a, s0b = seg[i] # 2 points
    #            s1a, s1b = seg[j] # 2 points
    #            d0a1a = np.linalg.norm(s0a-s1a)
    #            d0a1b = np.linalg.norm(s0a-s1b)
    #            d0b1a = np.linalg.norm(s0b-s1a)
    #            d0b1b = np.linalg.norm(s0b-s1b)
    #            dist_p2l(s0a
    #            d = np.min([d0a1a, d0a1b, d0b1a, d0b1b])
    #            if d <= tol:
    #                merged.append(j)

    #    return new_seg

    def show(self):
        print self._seg_n

        cv2.namedWindow('segments', cv2.WINDOW_NORMAL)
        while self._seg_i < self._seg_n:
            #self._map.fill(0.)
            p0, p1 = self._seg[self._seg_i] # == (2,2)
            self._seg_i += 1
            #if np.allclose(p0, p1):
            #    continue
            p0 = tuple(self.xy2uv(p0))
            p1 = tuple(self.xy2uv(p1))
            cv2.line(self._map, p0, p1, color=1.0, thickness=1)

            cv2.imshow('segments', self._map)
            cv2.waitKey(50)
            #k = cv2.waitKey(0)
            #if k == 27:
            #    break

        for j in self._joints:
            pj = tuple(self.xy2uv(j))
            cv2.circle(self._map, pj, radius=10, color=1.0)
            cv2.imshow('segments', self._map)
            cv2.waitKey(20)

        while True:
            k = cv2.waitKey(0)
            if k == 27:
                break

def main():
    data_path = os.path.expanduser('~/segments.npy')
    xseg, yseg = np.load(data_path)
    proc = SegProc(xseg, yseg)
    proc.show()

if __name__ == "__main__":
    main()
