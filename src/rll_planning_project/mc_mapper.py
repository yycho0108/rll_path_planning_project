import numpy as np
from map_utils import cast_ray
import utils as U

# check random points.
def monte_carlo_checker(srv, map, fpt, w, h, a=0):
    n,m = map.shape[:2]
    px, py = np.random.uniform(low=[-w/2,-h/2], high=[w/2,h/2])
    p0 = [px,py]
    pms = []

    # check cross
    for p1 in [[px,-h/2], [px,h/2], [-w/2,py], [w/2,py]]:
        pm = cast_ray(srv, p0, p1,a=a)
        if not np.allclose(p0, pm):
            pms.append(pm)

    if len(pms) > 0: # == start pose was valid
        # fill trace
        print ':)'
        for pm in pms:
            trace = fpt_hull(fpt, p0, pm, a=a)
            print trace
            trace = U.xy2uv(trace, w, h, n, m)
            print 'drawing ... '
            print trace, trace.shape
            cv2.drawContours(map, [trace.T], 0, color=255, thickness=-1)
