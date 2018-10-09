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
    pa, pb = seg
    delta = pb-pa
    return np.linalg.norm(delta)

def seg_join(s0, s1,
        max_dh=0.17, # ~10 deg
        max_dr=0.08,
        max_dl=0.05 # 5 cm
        ):

    h0 = seg_h(s0)
    h1 = seg_h(s1)
    dh = np.abs(U.adiff(h1,h0))
    if dh > max_dh:
        #print('violate h')
        return None

    dr = d_s2s(s0,s1)
    if dr > max_dr:
        #print('violate r')
        return None

    s = np.concatenate([s0,s1], axis=0)

    is_h = np.abs(np.sin(h0)) < np.sqrt(1.0/2)
    if is_h:
        pa = s[np.argmin(s[:,0])]
        pb = s[np.argmax(s[:,0])]
    else:
        pa = s[np.argmin(s[:,1])]
        pb = s[np.argmax(s[:,1])]

    l0 = seg_l(s0)
    l1 = seg_l(s1)
    l2 = np.dot(pb-pa, U.uvec(s0[1]-s0[0]))

    delta = np.abs([l2-l0, l2-l1])
    dl = np.min(delta)
    if dl > max_dl:
        return None
    else:
        return [s0,s1][np.argmin(delta)]

def ixtest():
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

def ds2ptest():
    seg = np.random.uniform(size=(2,2))
    dv  = seg[1] - seg[0]
    l   = np.linalg.norm(dv)
    tv  = dv / l
    nv  = tv.dot(U.R(np.pi/2).T)

    dy  = np.random.normal(scale=1.0)

    # case 1 : dx falls within segment
    dx  = np.random.uniform(low=0.0,high=l)
    pt = seg[0] + (dx * tv) + (dy * nv)
    print (d_s2p(seg, pt), np.abs(dy))

    # case 2 : point falls left of segment
    dx  = np.random.uniform(low=-5.0,high=0.0)
    pt = seg[0] + (dx * tv) + (dy * nv)
    print (d_s2p(seg, pt), np.linalg.norm([dx,dy]))

    # case 3 : point falls right of segment
    dx  = np.random.uniform(low=l, high=l+5.0)
    pt = seg[0] + (dx * tv) + (dy * nv)
    print (d_s2p(seg, pt), np.linalg.norm([dx-l,dy]))

    # case 4 : more randomized tests
    for _ in range(100):
        dx, dy = np.random.normal(scale=5.0, size=2)
        pt = seg[0] + (dx * tv) + (dy * nv)
        if (0 <= dx <= l):
            dx = 0
        else:
            dx = np.min(np.abs([0-dx,l-dx]))
        d_p = d_s2p(seg, pt)
        d_y = np.linalg.norm([dx,dy])
        if not (np.isclose(d_p,d_y)):
            raise AssertionError("Wrong Distance!")

def plot_segments(segs):
    from matplotlib import pyplot as plt
    for seg in segs:
        plt.plot(seg[:,0], seg[:,1])
    plt.show()

def sjtest():
    s0 = np.float32([[-0.39908934, -0.57437503],
       [ 0.53790772, -0.57437503]])
    #print('s0', s0)
    s1 = np.float32([[-0.39440185, -0.57437503],
       [ 0.5394702 , -0.57437503]])
    #print('s1', s1)
    print seg_join(s0,s1)

def pt_str(p):
    p = np.around(p, 3)
    return '({},{})'.format(p[0],p[1])

def seg_str(s):
    return '{}->{}'.format(pt_str(s[0]), pt_str(s[1]))


def pstest():
    segs = [
            [[-0.37200001,  0.21539062], [-0.37200001,  0.5909375 ]],
            [[-0.39925   ,  0.21539062], [ 0.04887498,  0.21539062]],
            [[-0.39925   ,  0.41539061], [ 0.39459372,  0.41539061]],
            [[-0.5423125 ,  0.57539064], [ 0.39459372,  0.57539064]],
            [[ 0.04075   , -0.25354248], [ 0.04075   ,  0.44679445]],
            [[-0.17925   ,  0.41539061], [-0.17925   ,  0.59442019]],
            [[ 0.02075   , -0.25045106], [ 0.02075   ,  0.4500415 ]],
            [[ 0.22075   , -0.44739017], [ 0.22075   ,  0.4500415 ]],
            [[ 0.38075   , -0.59743899], [ 0.38075   ,  0.73879886]],
            [[-0.5423125 , -0.74248779], [-0.5423125 ,  0.73914796]],
            [[-0.1823125 ,  0.41065583], [-0.1823125 ,  0.59504151]],
            [[ 0.1376875 ,  0.55944854], [ 0.1376875 ,  0.73914796]],
            [[ 0.37768748, -0.5990091 ], [ 0.37768748,  0.73914796]],
            ]
    #        [[ 0.        , -0.59250003], [ 0.        , -0.41429687]],
    #        [[-0.39640623, -0.59250003], [ 0.54390621, -0.59250003]],
    #        [[-0.39640623, -0.43250003], [ 0.39640623, -0.43250003]],
    #        [[-0.39640623, -0.74289066], [-0.39640623, -0.22122072]],
    #        [[ 0.22359377, -0.74289066], [ 0.22359377, -0.56021488]],
    #        [[ 0.36359376, -0.59851563], [ 0.36359376,  0.74195313]],
    #        [[ 0.54359376, -0.74289066], [ 0.54359376,  0.74195313]],
    #        [[-0.39640623, -0.74093753], [-0.39640623, -0.22324221]],
    #        [[ 0.22359377, -0.44902349], [ 0.22359377,  0.44257814]],
    #        [[ 0.38359377, -0.59773439], [ 0.38359377,  0.73744142]],
    #        ]
    segs = np.asarray(segs)
    plot_segments(segs)

def main():
    #ds2ptest()
    #sjtest()
    pstest()

if __name__ == "__main__":
    main()







