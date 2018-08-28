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

def dist_p2s(p, s):
    p      = np.asarray(p, dtype=np.float32)
    p1, p2 = np.asarray(s, dtype=np.float32)

    sd = np.linalg.norm(p2 - p1)
    if np.isclose(sd, 0):
        return np.linalg.norm(p - p1)
    t = np.dot(p - p1, p2 - p1) / (sd*sd)
    if t < 0:
        return np.linalg.norm(p - p1)
    elif t > 1:
        return np.linalg.norm(p - p2)
    proj = p1 + t * (p2 - p1)
    return np.linalg.norm(p - proj)

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

    def match_edge(self, pt, tol=2e-2):
        # warning : match_edge WILL modify edges to accomodate nodes.
        for ei, (ep0,ep1) in enumerate(self._edges):
            dist = dist_p2s(pt, [ep0,ep1])
            if np.abs(dist) < tol:
                return ei
        return None

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
    def __init__(self, xseg, yseg, pts=[], w=1.2, h=1.6, r=0.005, raw=False):
        if raw:
            self._xseg = xseg
            self._yseg = yseg
        else:
            self._xseg = self.prune(xseg) #self.prune(xseg)
            self._xseg = self.merge_x(self._xseg, dtol=8e-2, gtol=5e-2)
            self._yseg = self.prune(yseg)
            #self._yseg = self._yseg[np.logical_and(0.5<self._yseg[:,0,0], self._yseg[:,0,0]<1.0)]
            self._yseg = self.merge_y(self._yseg, dtol=7e-2, gtol=5e-2)#yseg #selg.prune(yseg)

        self._xseg, self._yseg, self._joints, self._con, self._dist = self.join_xy(self._xseg, self._yseg, pts)

        self._path = self.plan(self._joints, self._con)

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
        self._map = np.zeros(shape=(n,m,3), dtype=np.float32)

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

    def add_points(self, xseg, yseg, joints,
            pts, tol=3e-2):

        joints = list(joints)
        xseg   = list(xseg)
        yseg   = list(yseg)
        indices= []

        for pidx, pt in enumerate(pts):
            nx = len(xseg)
            ny = len(yseg)
            jds = []
            sds = []

            for jpt in joints:
                jds.append(np.linalg.norm(np.subtract(pt,jpt)))
            jdi = np.argmin(jds)
            if jds[jdi] < tol:
                # pre-existing point match
                indices.append(jdi)
                continue

            for s in xseg:
                sds.append(dist_p2s(pt, s))
            for s in yseg:
                sds.append(dist_p2s(pt, s))

            i = np.argmin(sds)
            d = sds[i]
            if d > tol: # fail to match segment
                print 'failed tolerance'
                return False
            if i < nx: # split x
                x0, x1 = xseg[i]
                xseg[i] = [x0, [pt[0], x0[1]]]
                xseg.append([[pt[0], x0[1]], x1])
            else: # split y
                y0, y1 = yseg[i - nx]
                yseg[i - nx] = [y0, [y1[0], pt[1]]]
                yseg.append([[y1[0], pt[1]], y1])
            joints.append(pt)
            indices.append(len(joints) - 1)

        xseg = np.asarray(xseg, dtype=np.float32)
        yseg = np.asarray(yseg, dtype=np.float32)
        joints = np.asarray(joints, dtype=np.float32)
        return xseg, yseg, joints, indices

    def join_xy(self, xseg, yseg, pts=[], dtol=5e-2):

        joints = []

        # join x-y segments
        xseg = list(np.copy(xseg))
        yseg = list(np.copy(yseg))
        xi,yi = 0,0

        xn0 = len(xseg)
        yn0 = len(yseg)

        while xi < len(xseg):
            px0, px1 = xseg[xi]
            yi = 0
            slices = []
            new_yseg = []

            while yi < len(yseg):
                py0, py1 = yseg[yi]

                x, y = [py0[0], px0[1]] # point of intersection

                in_x = (px0[0] < x < px1[0])
                in_y = (py0[1] < y < py1[1])
                dx = np.min(np.abs( [px0[0]-x, x-px1[0]] ))
                dy = np.min(np.abs( [py0[1]-y, y-py1[1]] ))

                x_good = (in_x or (dx < dtol))
                y_good = (in_y or (dy < dtol))

                if x_good and y_good:
                    if xi<xn0:
                        joints.append([x,y])

                    if in_y and dy > dtol:
                        # split + append
                        yseg[yi] =  [[x, py0[1]], [x, y]] # lesser half
                        new_yseg.append([[x, y], [x, py1[1]]]) # greater half

                    if in_x and dx > dtol:
                        slices.append(x)
                yi += 1
            yseg.extend(new_yseg)

            if len(slices) > 0:
                slices.append(px0[0])
                slices.append(px1[0])
                slices = sorted(slices)

                y = px0[1]
                xseg[xi] = [[slices[0],y], [slices[1],y]]
                sn = len(slices)
                for i in range(1, sn-1):
                    xseg.append([[slices[i],y], [slices[i+1],y]])

                #for (x0, x1) in zip(slices[:-1], slices[1:]):
                #    # split + append
                #    xseg[xi] =  [[x0, y], [x, y]] # lesser half
                #    xseg.append([[x, y], [x1, y]]) # greater half

            xi += 1

        #for xi, xx in enumerate(xseg):
        #    for yi, yy in enumerate(yseg):
        #        px0, px1 = xx
        #        py0, py1 = yy
        #        x, y = [py0[0], px0[1]] # point of intersection

        #        in_x = (px0[0] < x < px1[0])
        #        in_y = (py0[1] < y < py1[1])
        #        dx = np.min(np.abs( [px0[0]-x, x-px1[0]] ))
        #        dy = np.min(np.abs( [py0[1]-y, y-py1[1]] ))

        #        x_good = (in_x or (dx < dtol))
        #        y_good = (in_y or (dy < dtol))

        #        #print dx, dy

        #        if x_good and y_good:
        #            joints.append([x,y])

        joints = np.asarray(joints, dtype=np.float32) # == (N_J, 2)
        #joints = self.add_points(xseg, yseg, joints,
        #    [[-0.37, -0.5], [0.45, -0.26]], tol=3e-2)

        # add segment endpoints
        d_xs = np.reshape(joints, [-1,1,2]) - np.reshape(xseg, [1,-1,2]) # == (N_J, 2*N_X, 2)
        d_xs = np.linalg.norm(d_xs, axis=-1) # == (N_J, 2*N_X)
        xsel  = np.all(d_xs > dtol, axis=0)
        j_x  = np.reshape(xseg, [-1,2])[xsel]

        d_ys = np.reshape(joints, [-1,1,2]) - np.reshape(yseg, [1,-1,2]) # == (N_J, 2*N_X, 2)
        d_ys = np.linalg.norm(d_ys, axis=-1) # == (N_J, 2*N_X)
        ysel  = np.all(d_ys > dtol, axis=0)
        j_y  = np.reshape(yseg, [-1,2])[ysel]

        joints = np.concatenate([j_x, j_y, joints], axis=0)

        # add start / finish ( TODO : current code is not flexible)
        xseg, yseg, joints, pidx = self.add_points(xseg, yseg, joints, pts, tol=3e-2)
        self._pidx = pidx

        n_j = len(joints)
        print 'Number of Joints : ', len(joints)

        # construct joint connectivity assignments from segments
        con  = np.zeros(shape=(n_j,n_j), dtype=np.bool) # connectivity matrix
        dist = np.full(shape=(n_j,n_j), fill_value=np.inf, dtype=np.float32) # edge distance matrix

        seg = np.concatenate([xseg, yseg], axis=0)

        print 'Number of Segments : ', len(seg)
        for p0, p1 in seg:
            print p0, p1
            src = np.argmin(np.linalg.norm(joints - np.reshape(p0, [1,2]), axis=-1))
            dst = np.argmin(np.linalg.norm(joints - np.reshape(p1, [1,2]), axis=-1))
            con[src,dst] = con[dst,src] = True
            dist[src,dst] = dist[dst,src] = np.linalg.norm((p0 - p1), axis=-1)

        # visualize matrix
        #cv2.namedWindow('con', cv2.WINDOW_NORMAL)
        #cv2.imshow('con', con.astype(np.float32))

        return xseg, yseg, joints, con, dist

    def plan(self, joints, con):
        # WARNING : call plan after join_xy(), add_points() and related calls.
        src, dst = self._pidx
        dist = np.full(len(joints), 9999.0,  dtype=np.float32)
        dist[src] = 0.0
        prv = {}
        n_j = len(joints)
        Q = range(n_j)

        while len(Q) > 0:
            i = np.argmin([dist[e] for e in Q])
            u = Q.pop(i)
            p0 = joints[u]
            for v in np.where(con[u])[0]: # works? or not?
                p1 = joints[v]
                alt = dist[u] + np.linalg.norm(p0 - p1, axis=-1)
                if alt < dist[v]:
                    dist[v] = alt
                    prv[v] = u

            if u == dst:
                break
        else:
            return [] # no valid plan found!!

        u = dst
        res = [dst]
        print 'Src-Dist', src, dst
        print 'Optimal Path', prv
        while True:
            res.insert(0, prv[u])
            u = prv[u]
            if u == src:
                break
        return [joints[i] for i in res]

    def show(self):
        print self._seg_n

        cv2.namedWindow('segments', cv2.WINDOW_NORMAL)

        # draw joints
        n_j = len(self._joints)
        for ji, jpt in enumerate(self._joints):
            pj = tuple(self.xy2uv(jpt))
            col = ((1.0,1.0,1.0) if (ji not in self._pidx) else np.random.uniform(size=3))
            cv2.circle(self._map, pj, radius=10, color=col, thickness=1)
            #cv2.circle(self._map, pj, radius=10, color=np.random.uniform(size=3), thickness=1)

            cv2.imshow('segments', self._map)
            k = cv2.waitKey(10)
            if k == 27:
                break

        # draw connectivity
        for (i, j) in np.transpose(np.triu_indices(self._con.shape[0])):
            if not self._con[i,j]: continue
            p0 = tuple(self.xy2uv(self._joints[i]))
            p1 = tuple(self.xy2uv(self._joints[j]))
            cv2.line(self._map, p0, p1, color=(1.0,1.0,1.0), thickness=1)
            #cv2.line(self._map, p0, p1, color=np.random.uniform(size=3), thickness=2)
            cv2.imshow('segments', self._map)
            k = cv2.waitKey(10)
            if k == 27:
                break

        # draw path

        for p0, p1 in zip(self._path[:-1], self._path[1:]):
            p0 = tuple(self.xy2uv(p0))
            p1 = tuple(self.xy2uv(p1))

            cv2.line(self._map, p0, p1, color=(1.0,0.0,0.0), thickness=2)
            #cv2.line(self._map, p0, p1, color=np.random.uniform(size=3), thickness=2)
            cv2.imshow('segments', self._map)
            k = cv2.waitKey(0)
            if k == 27:
                break

        ## draw segments (orig)
        #while self._seg_i < self._seg_n:
        #    #self._map.fill(0.)
        #    p0, p1 = self._seg[self._seg_i] # == (2,2)
        #    self._seg_i += 1
        #    #if np.allclose(p0, p1):
        #    #    continue
        #    p0 = tuple(self.xy2uv(p0))
        #    p1 = tuple(self.xy2uv(p1))
        #    cv2.line(self._map, p0, p1, color=np.random.uniform(size=3), thickness=1)

        #    cv2.imshow('segments', self._map)
        #    cv2.waitKey(10)
        #    #k = cv2.waitKey(0)
        #    #if k == 27:
        #    #    break


        while True:
            k = cv2.waitKey(0)
            if k == 27:
                break

def main():
    data_path = os.path.expanduser('~/segments.npy')
    xseg, yseg = np.load(data_path)
    #proc = SegProc(xseg, yseg, pts=[[0.38,0], [-0.37,0.5]])
    proc = SegProc(xseg, yseg, pts=[[-0.37, -0.5],[0.45,-0.26]])
    proc.show()

if __name__ == "__main__":
    main()
