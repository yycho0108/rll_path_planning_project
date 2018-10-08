#!/usr/bin/env python

import numpy as np
import utils as U

def on_segment(p,q,r):
    px,py = p
    qx,qy = q
    rx,ry = r
    return (qx <= max(px,rx) and
        qx >= min(px,rx) and
        qy <= max(py,ry) and
        py >= min(py,ry))

def triplet_orientation(p,q,r):
    px,py = p
    qx,qy = q
    rx,ry = r
    val = (qy-py) * (rx-qx) - (qx-px) * (ry-qy)  
    if np.isclose(val, 0):
        return 0
    if val>0:
        return 1
    else:
        return 2

def segment_intersect(s0, s1):
    p1, q1 = s0
    p2, q2 = s1

    o1 = triplet_orientation(p1,q1,p2)
    o2 = triplet_orientation(p1,q1,q2)
    o3 = triplet_orientation(p2,q2,p1)
    o4 = triplet_orientation(p2,q2,q1)

    if (o1 != o2 and o3 != o4):
        return True

    if (o1 == 0 and on_segment(p1, p2, q1)): return True
    if (o2 == 0 and on_segment(p1, q2, q1)): return True
    if (o3 == 0 and on_segment(p2, p1, q2)): return True
    if (o4 == 0 and on_segment(p2, q1, q2)): return True

    return False

def d_s2p(seg,pt):
    v = seg[1] - seg[0]
    u = (pt - seg[0]).dot(v) / (v.dot(v))
    u = np.clip(u,0,1)
    c = seg[0] + u * v
    return np.linalg.norm(c - pt)

def d_s2s(s0,s1):
    d00=d_s2p(s0,s1[0])
    d01=d_s2p(s0,s1[1])
    d10=d_s2p(s1,s0[0])
    d11=d_s2p(s1,s0[1])
    return np.min([d00,d01,d10,d11])

def seg_h(seg):
    pa,pb = seg
    delta = pb-pa
    return np.arctan2(delta[1],delta[0])

def seg_l(seg):
    pa,pb = seg
    delta = pb-pa
    return np.linalg.norm(delta)

def seg_joinable(s0, s1,
        max_dh=0.17, # ~10 deg
        max_dl=0.05, # 5 cm
        max_dr=0.05
        ):

    h0 = seg_h(s0)
    h1 = seg_h(s1)
    dh = np.abs(U.adiff(h1,h0))
    if dh > max_dh:
        return False

    pa = np.min([s0,s1], axis=0)
    pb = np.max([s0,s1], axis=0)

    l0 = seg_l(s0)
    l1 = seg_l(s1)
    l2 = seg_l([pa,pb])
    dl = np.min([np.abs(l2-l0), np.abs(l2-l1)])
    if dl > max_dl:
        return False

    dr = d_s2s(s0,s1)
    if dr > max_dr:
        return False

    return True

def main():
    n_test = 10
    np.random.seed(0)
    from matplotlib import pyplot as plt
    seg_1, seg_2 = np.random.uniform(size=(2,n_test,2,2))
    for s1, s2 in zip(seg_1, seg_2):
        # test colinear cases
        # ds = (s1[1] - s1[0])
        # ds /= np.linalg.norm(ds)
        # s2 = np.mean(s1, axis=0, keepdims=True) + [2.0 * ds, 3.0 * ds]
        ix = segment_intersect(s1,s2)
        plt.clf()
        plt.plot(s1[:,0], s1[:,1], 'r-')
        plt.plot(s2[:,0], s2[:,1], 'b-')
        plt.title('Intersection' if ix else 'No Intersection')
        plt.show()

if __name__ == "__main__":
    main()
