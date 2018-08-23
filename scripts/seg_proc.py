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

class SegProc(object):
    def __init__(self, xseg, yseg, w=1.2, h=1.6, r=0.005, raw=False):
        if raw:
            self._xseg = xseg
            self._yseg = yseg
        else:
            self._xseg = self.prune(xseg) #self.prune(xseg)
            self._xseg = self.merge_x(xseg, dtol=4e-2, gtol=5e-2)
            self._yseg = self.prune(yseg)
            self._yseg = self.merge_y(yseg, dtol=4e-2, gtol=5e-2)#yseg #selg.prune(yseg)

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
                gd = np.min([jx0-ix1, ix0-jx1]) # handles overlaps=negative
                x0 = min(ix0, jx0)
                x1 = max(ix1, jx1)
                gd = (x1-x0) - (ix1-ix0) - (jx1-jx0)

                if nd<dtol and gd<gtol:
                    y = (iy + jy) / 2.0
                    seg[i][:,:] = [[x0,y],[x1,y]]
                    merged[j] = True
        return seg[select]
    
    def merge_y(self, seg, dtol=1e-1, gtol=5e-2):
        # TODO : avoid duplicate code
        n = len(seg)
        seg = np.copy(seg) # make a copy
        select = np.full(n, False, dtype=np.bool)
        merged = np.full(n, False, dtype=np.bool)
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
        return seg[select]

    def join_xy(self, xseg, yseg, dtol=5e-2):
        for xx in xseg:
            for yy in yseg:
                
        pass

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
            cv2.waitKey(200)
            #k = cv2.waitKey(0)
            #if k == 27:
            #    break

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
