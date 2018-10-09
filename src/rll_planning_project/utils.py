import numpy as np
import time

def xy2uv(xy, w, h, n, m):
    # convert from physical coordinates to map coordinates
    # x -> u, y -> v
    x, y = xy
    mv = (n/2.0) + x*(float(n) / w)
    mu = (m/2.0) + y*(float(m) / h)
    mv = np.round(np.clip(mv, 0, n)).astype(np.int32)
    mu = np.round(np.clip(mu, 0, m)).astype(np.int32)
    return np.stack([mu,mv], axis=0) # TODO : mu-mv? check order

def R(x):
    c,s = np.cos(x), np.sin(x)
    return np.asarray([[c,-s],[s,c]], dtype=np.float32) #2x2

def anorm(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def adiff(a,b):
    return anorm(a-b)

def uvec(x):
    return x / np.linalg.norm(x)

class Benchmark(object):
    """ from https://stackoverflow.com/a/41408510 """
    def __init__(self, msg, fmt="%0.3g"):
        self.msg = msg
        self.fmt = fmt
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        t = time.time() - self.start
        print(("%s : " + self.fmt + " seconds") % (self.msg, t))
        self.time = t
