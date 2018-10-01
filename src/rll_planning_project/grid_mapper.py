#!/usr/bin/env python

viz_enabled = True
import numpy as np
import utils as U
from map_utils import cast_ray

try:
    import cv2
    from scipy.spatial import ConvexHull
except ImportError as e:
    print('cv2/scipy import failed : {}'.format(e))
    print('visualization disabled.')
    viz_enabled = False

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

    def __call__(self, srv, map, fpt, viz=False, log=None):
        viz = (viz and viz_enabled) # check global viz flag

        if log is not None:
            log('{} : {}'.format(self._xpt, self._ypt))

        # unroll parameters / data
        n, m = map.shape[:2]
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
                    trace = U.xy2uv(trace, w, h, n, m)
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
                    trace = U.xy2uv(trace, w, h, map)
                    cv2.drawContours(map, [trace.T], 0, color=255, thickness=-1)

                # save
                self._yseg.append(np.copy([ypt, ym]))

            # set to next point
            ypt[1] = max(ypt[1]+r, ym[1]+r)

    def done(self):
        return (self._xfin and self._yfin)

    def save(self, path='/tmp/segments.npy'):
        np.save(path, [self._xseg, self._yseg])
