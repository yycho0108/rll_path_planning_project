import numpy as np
from geometry_msgs.msg import Pose2D
from scipy.spatial import ConvexHull
import utils as U

def cast_ray(srv, p0, p1, a=0, min_d=0.005):
    """ Search ray collision point via simple binary search.

    Args:
        srv(function(p0,p1)->bool): check if path from p0 to p1 is valid.
            p0(geometry_msgs/Pose2D): initial point
            p1(geometry_msgs/Pose2D): final point
        p0(np.ndarray): Ray start, [2] array formatted (x,y)
        p1(np.ndarray): Ray end, [2] array formatted (x,y)
        a(float): Object orientation during path, radians
        min_d(float): raycasting collision search resolution

    Returns:
        pm(np.ndarray): Ray colliison endpoint [2] array formatted (x,y)
    
    """ 
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

def fpt_hull(fpt, p0, p1, a0=0, a1=None):
    """ Convex Hull from footprint trajectory.

    Args:
        fpt(np.ndarray): [N,2] polygon formatted (x,y)
        p0(np.ndarray): segment start, [2] array formatted (x,y)
        p1(np.ndarray): segment end, [2] array formatted (x,y)
        a0(float): segment start angle, radians (default: 0)
        a1(float): segment end angle, radians (default: a0)

    """
    if a1 is None:
        a1 = a0
    r0 = U.R(a0).dot(fpt) + np.reshape(p0, [2,1])
    r1 = U.R(a1).dot(fpt) + np.reshape(p1, [2,1])
    ps = np.concatenate([r0,r1], axis=1) # 2x8
    hull = ps.T[ConvexHull(ps.T).vertices].T# TODO: ps.T??
    return hull

