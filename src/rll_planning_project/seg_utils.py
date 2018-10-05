#!/usr/bin/env python

import numpy as np

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
