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

import os
import numpy as np
import cv2

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

    def __call__(self, srv, map, fpt, viz=False):
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
                if viz:
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
                if viz:
                    # mark ...
                    trace = fpt_hull(fpt, ypt, ym, a=0.)
                    trace = xy2uv(trace, w, h, map)
                    cv2.drawContours(map, [trace.T], 0, color=255, thickness=-1)

                # save
                self._yseg.append(np.copy([ypt, ym]))

            # set to next point
            ypt[1] = max(ypt[1]+r, ym[1]+r)

    def done(self):
        return (self._xfin and self._yfin)

    def save(self, path='/tmp/segments.npy'):
        np.save(path, [self._xseg, self._yseg])

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
        # TODO : add visualization argument, separate phases into functions, etc.
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
        grid_mapper = GridMapper(self._mw, self._mh, self._fw, self._fh, r=0.02)
        if viz:
            cv2.namedWindow('map', cv2.WINDOW_NORMAL)
        while not (grid_mapper.done()):
            try:
                grid_mapper(check_srv, self._map, self._fpt)
            except Exception as e:
                rospy.logerr_throttle(1.0, 'Grid Mapper Failed : {}'.format(e))
            if viz:
                cv2.imshow('map', self._map)
                cv2.waitKey(1)
        ## grid_mapper.save()
        xseg, yseg = grid_mapper._xseg, grid_mapper._yseg
        xseg = np.asarray(xseg, dtype=np.float32)
        yseg = np.asarray(yseg, dtype=np.float32)
        if len(xseg) <= 0 or len(yseg) <= 0:
            rospy.logerr_throttle(1.0, 'Grid Mapper Failed!')
            return

        # alt 1: use cached segments, NOTE : only for testing!!
        # data_path = os.path.expanduser('/tmp/segments.npy')
        # xseg, yseg = np.load(data_path)

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

if __name__ == '__main__':
    rospy.init_node('path_planner')
    server = PathManager()
    rospy.spin()
