import numpy as np

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
