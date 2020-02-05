"""Miscellaneous routines for manipulating data."""

import matplotlib.pylab as plt
import numpy as np
import matplotlib as mpl
import matplotlib.ticker
import matplotlib.transforms
import matplotlib.gridspec

from matplotlib.collections import LineCollection

from skimage.morphology import label

__author__ = "David Andrews"
__copyright__ = "Copyright 2015, David Andrews"
__license__ = "MIT"
__version__ = "1.0"
__email__ = "david.andrews@irfu.se"

paper_sizes = dict(A4=(8.27,11.69), A3=(11.69,16.54), A5=(5.83, 8.27))

datastore = dict()

def clear_datastore():
    datastore = dict()

def code_interact(local):
    import code
    print('Stopping...')
    code.interact(local=local)
    print('Resuming...')

def find_period(data):
    """Directional statistics up ins"""
    n = data.shape[0]
    best_r = -0.0
    for i in range(2, n/2):
        r = np.sqrt()

def deg_unwrap(data, discont=180.):
    non_nan_inx = np.isfinite(data)
    if not np.all(non_nan_inx):
        out = np.empty_like(data) + np.nan
        out[non_nan_inx] = np.rad2deg( np.unwrap(np.deg2rad(data[non_nan_inx]),
                                    np.deg2rad(discont)))
        return out
    return np.rad2deg( np.unwrap(np.deg2rad(data), np.deg2rad(discont)))

def remove_none_edge_intersecting(img, edge=0, width=1):

    mask = np.zeros(img.shape,dtype=int)
    out = np.zeros(img.shape,dtype=int)
    # print '--->', img.sum()

    if edge == 0:
        mask[:,0:0+width] = 1
    elif edge == 1:
        mask[:,-1-width:-1] = 1
    elif edge == 2:
        mask[0:width,:] = 1
    elif edge == 3:
        mask[-1-width:-1,:] = 1
    else:
        raise ValueError('Edge is duff')

    s = label(img.astype(int))
    s_set = np.unique(s * mask)
    if s_set.sum() > 0:
        for v in s_set:
            q = (s == v)
            if np.all(img[q]):
                out[s == v] = 1

    return out

def interp_within_dx(xout, x, y, dx=1., fill=np.nan, left=None, right=None):
    """interp, but then any interpolates that were greater than dx from a point in x are filled"""
    if left is None:
        left = fill
    if right is None:
        right = fill

    out = np.interp(xout, x, y, left=left, right=right)

    xt = np.interp(xout, x, x)
    print(xt)
    print(xout)
    print(np.abs(xt - xout))
    out[np.abs(xt - xout) > dx] = fill

    raise NotImplementedError()
    return out

def interp_safe(x_new, x_old, y_old, max_step=1.,
                    left=np.nan, right=np.nan, missing=np.nan):
    """Interpolate, defaulting to using nans for unknowns.  Also fill result with nans if the gap between interpolation points and the input is bigger than max_step"""

    y_new = np.interp(x_new, x_old, y_old, left=left, right=right)

    dx_min = np.min(np.abs(x_old - x_new[:, np.newaxis]), 1)

    y_new[dx_min > max_step] = missing

    if np.all(~np.isfinite(y_new)):
        print(np.sum(dx_min > max_step), y_new.size)
        raise ValueError()

    return y_new

def lat_lon_distance(p1, p2, radius):
    """p1, p2 are points (lat, lon) in degrees.  Distance returned"""
    lat_1, lon_1 = np.deg2rad(np.array(p1))
    lat_2, lon_2 = np.deg2rad(np.array(p2))
    q = np.sin((lat_2 - lat_1)/2.)**2. + np.cos(lat_1) * np.cos(lat_2) * np.sin((lon_2 - lon_1)/2.)**2.
    return 2. * radius * np.arcsin(q**0.5)

def weighted_avg_std(values, weights):
    """
    Returns the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.dot(weights, (values-average)**2)/weights.sum()  # Fast and numerically precise
    return (average, np.sqrt(variance))




def running_operation(data, w, operation=None, edge_nan=False):
    """Apply a function along a 1-D array using a window of half-width `w`"""
    result = []
    n = len(data)
    for i in range(w + 1, 2 * w + 1):
        if edge_nan:
            result.append(np.nan)
            continue
        window = data[:i]
        result.append(operation(window))
        # print '[ ', window[0], window[-1], (window[-1] - window[0]) == (2 * w)
    for i in range(w, data.shape[0] - w):
        window = data[i-w:i+w+1]
        result.append(operation(window))
        # print '> ', window[0], window[-1], (window[-1] - window[0]) == (2 * w)

    for i in range(n - 2 * w, n-w):
        if edge_nan:
            result.append(np.nan)
            continue
        window = data[i:]
        result.append(operation(window))
        # print '] ', window[0], window[-1], (window[-1] - window[0]) == (2 *w)

    # print len(data), len(result)

    return np.array(result)


def running_mean(data, w):
    """Running np.mean"""
    return running_operation(data, w, operation=np.mean)

def running_median(data, w):
    """Running np.median"""
    return running_operation(data, w, operation=np.median)

def running_std(data, w):
    """Running np.std"""
    return running_operation(data, w, operation=np.std)


class InfiniteIterator(object):
    """There is always a next() time"""
    def __init__(self, data):
        super(InfiniteIterator, self).__init__()
        self._data = data
        self._i = 0
        self._len = len(self._data)

    def __next__(self):
        val = self._data[self._i]
        self._i += 1
        if self._i >= self._len:
            self._i = 0
        return val

def angle_difference(x, y, degrees=False):
    """Signed smallest distance between two angles x and y.  Input in radians, or degrees if specified."""
    if degrees:
        conv = np.pi / 180.
        return np.arctan2(np.sin((x - y) * conv), np.cos((x-y) * conv)) / conv
    return np.arctan2(np.sin(x - y), np.cos(x-y))

def circular_mean(x, degrees=False, std=False, r=False):
    """Return circular / directional mean of x.  Also std, r if requested. Degrees if specified, otherwise radians assumed"""

    conv = 1.
    if degrees: conv = np.pi/180.

    cm = np.mean(np.cos(x * conv))
    sm = np.mean(np.sin(x * conv))

    r_v = np.sqrt(cm * cm + sm*sm)
    ma = np.arctan2(sm,cm) / conv
    std_v = np.sqrt(-2. * np.log(r_v)) / conv

    if r:
        if std:
            return ma, std_v, r_v
        else:
            return ma, r_v
    if std:
        return ma, std_v
    return ma

def nice_range(x, p=5.):
    """Range of data from p'th to 100-p'th percentile (p=5 default)"""
    i = np.isfinite(x)
    return np.percentile(x[i],p), np.percentile(x[i],100.-p)

def print_call(f):
    def fin(*args, **kwargs):
        print('Calling ' + f.__name__)
        return f(*args, **kwargs)
    return functools.wraps(f)(fin)




def bin_2d(x, y, z=None, xbins=None, ybins=None, func=None, background=None):
    """2-D processing (binning) of data, using np.digitize.
    bins from (x0, ..., xM), (y0, ..., yN), returned image has dimension (M-1, N-1).  Bins therefore specify a closed interval.

    Nans are not handled, so that they can be dealt with in the func.
    """

    if background is None:
        background = np.nan

    if z is None:
        z = np.empty_like(x)
        if func is None:
            func = np.size

    if func is None:
        func = np.sum

    if len(x.shape) != 1: raise ValueError('1D-only')
    if z.shape != x.shape: raise ValueError("Shape Mis-match")
    if z.shape != y.shape: raise ValueError("Shape Mis-match")


    if (xbins[1] - xbins[0]) < 0. :
        raise ValueError("Bins should be increasing")
    if (ybins[1] - ybins[0]) < 0.:
        raise ValueError("Bins should be increasing")

    test_call = func(z[0:2])
    img = np.empty((ybins.shape[0] - 1, xbins.shape[0] - 1)) + background

    if isinstance(test_call, np.ndarray):
        print('XX', test_call.shape)
        if test_call.shape[0] != 1:
            shape = test_call.shape[0]
            img = np.empty((ybins.shape[0] - 1, xbins.shape[0] - 1,
                        test_call.shape[0])) + background

    dx = np.digitize(x, xbins, right=False) - 1
    dy = np.digitize(y, ybins, right=False) - 1

    # print 'Binning...'
    empty_bins = 0
    for i in range(xbins.shape[0] - 1):
        tmp = (dx == i)
        # print i
        for j in range(ybins.shape[0] - 1):
            tmp2 = tmp & (dy == j)
            try:
                img[j,i,...] = func(z[tmp2])


            except ValueError as e:
                img[j,i,...] = np.nan
                continue

    return img

def center_phase_data(x, center=0., interval=360.):
    t = interval/2. - center
    return (((x + t) % interval) - t)


def polar_to_cartesian(pos, vec):
    """Coordinate conversion, input position in (radial dist, latitude, longitude ) [deg]."""
    clat = np.pi/2 - pos[1] * np.pi/180.
    lon = pos[2] * np.pi/180.

    out = np.array((
      np.sin(clat) * np.cos(lon) * vec[0] + np.cos(clat) * np.cos(lon) * vec[1] - np.sin(lon) * vec[2],
      np.sin(clat) * np.sin(lon) * vec[0] + np.cos(clat) * np.sin(lon) * vec[1] + np.cos(lon) * vec[2],
      np.cos(clat) * vec[0] - np.sin(clat) * vec[1]
    ))

    return out

def cartesian_to_polar(pos, vec):
    """Coordinate conversion, input position in (radial dist, latitude, longitude ) [deg]."""
    clat = np.pi/2 - pos[1] * np.pi/180.
    lon = pos[2] * np.pi/180.
    out = np.array((
      np.sin(clat) * np.cos(lon) * vec[0] + np.sin(clat) * np.sin(lon) * vec[1] + np.cos(clat) * vec[2],
      np.cos(clat) * np.cos(lon) * vec[0] + np.cos(clat) * np.sin(lon) * vec[1] - np.sin(clat) * vec[2],
      -np.sin(lon) * vec[0] + np.cos(lon) * vec[1]
    ))
    return out
