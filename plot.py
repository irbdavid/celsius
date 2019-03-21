"""Routines for extending Matplotlib.
"""

import matplotlib.pylab as plt
import numpy as np
import matplotlib as mpl
import matplotlib.ticker
import matplotlib.transforms
import matplotlib.gridspec

from matplotlib.collections import LineCollection

from skimage.morphology import label

from celsius import interp_safe

__author__ = "David Andrews"
__copyright__ = "Copyright 2015, David Andrews"
__license__ = "MIT"
__version__ = "1.0"
__email__ = "david.andrews@irfu.se"

paper_sizes = dict(A4=(8.27,11.69), A3=(11.69,16.54), A5=(5.83, 8.27))

def make_colorbar_cax(ax=None, width=0.01, offset=0.02, height=0.9,
                        left=False, half=False, upper=True):
    """Define a new vertical colorbar axis relative to :ax:.  Avoids 'stealing' space from the axis, required for stack plots.

Args:
    ax: Axes instance to work with, defaults to current.
    width: width of colorbar relative to axes width
    height: height relative to axes height
    offset: offset relative to appropriate axes dimension
    left: place on left side (default False)
    half: only occupy half the vertical extent, for two colorbars?
    upper: upper of two colorbars?

Returns:
    new axes instance for passing to colorbar(cax=...)
"""
    if ax is None:
        ax = plt.gca()
    else:
        plt.sca(ax)

    bbox = ax.get_position()
    x0 = bbox.x1
    if left:
        offset = abs(offset) * -1.0
        x0 = bbox.x0

    y0 = bbox.y0
    y1 = bbox.y1
    # ys = (1.0 - height) * (y1 - y0)
    ys = (1.0 - height) * (y1 - y0) / 2.
    # new_ax_coords = [x0 + offset, y0 + ys, width, height * (y1 - y0) - ys]
    new_ax_coords = [x0 + offset, y0 + ys, width, height * (y1 - y0)]

    # |==|---------|==|----------|==|
    if half:
        new_ax_coords[3] = new_ax_coords[3] / 2.
        if upper:
            new_ax_coords[1] = y0 + height * (y1 - y0) * 0.5 + ys/2.
        else:
            new_ax_coords[1] -= ys/2.
    new_ax = ax.figure.add_axes(new_ax_coords)
    plt.sca(ax)
    return new_ax

def make_colorbar_cax_horizontal(ax=None, width=0.9, offset=0.02, height=0.02,
                                upper=True, x0=None, x1=None, dw=None):
    """Define a new horizontal colorbar axis relative to :ax:.  Avoids 'stealing' space from the axis, required for stack plots.

Args:
    ax: Axes instance to work with, defaults to current.
    width: width of colorbar relative to axes width
    height: height relative to axes height
    offset: offset relative to appropriate axes dimension
    upper: place on upper side (default True)
    x0, x1: coordinates of the colorbar, for shifting the axes.
    dw: additional tweak

Returns:
    new axes instance for passing to colorbar(cax=...)
"""
    if ax is None:
        ax = plt.gca()
    else:
        plt.sca(ax)

    bbox = ax.get_position()
    if dw is None:
        dw = (1. - width) * (bbox.x1 - bbox.x0) / 2.
    actual_width = width * (bbox.x1 - bbox.x0)

    if x0 is None:
        x0 = bbox.x0 + dw
    if x1 is None:
        x1 = bbox.x1 - dw

    if upper:
        y0 = bbox.y1 + offset
        y1 = y0 + height
    else:
        y1 = bbox.y0 - offset
        y0 = y1 - height

    new_ax_coords = [x0, y0, actual_width, height]

    new_ax = ax.figure.add_axes(new_ax_coords)
    plt.sca(ax)
    return new_ax

def n_random_colors(n, spacing=None, max_breaks=None):
    """Generates a sequence of random colors, with somewhat optimized spacing.

    Args:
       n (int):  Number of colors.

    Kwargs:
       spacing (float): minimum spacing between colors on the color wheel
       max_breaks (int): max number of computations to perform

    Returns:
        list of colors."""
    colors = [np.random.rand(3)]

    if spacing is None:
        spacing = 5 * 1. / float(n)

    if max_breaks is None:
        max_breaks = 2 * n

    break_count = 0
    while len(colors) < n:
        new_color = np.random.rand(3)
        for c in colors:
            if np.sqrt(np.sum((c - new_color)**2., axis=0)) < spacing:
                if break_count < max_breaks:
                    break_count+=1
                    break
        else:
            colors.append(new_color)

    return [tuple(c) for c in colors]


def n_colors(n):
    """Return :n: evenly spaced colors"""
    c = np.linspace(0., 1., n)
    return [tuple((x, x, x)) for x in c]

# def ylabel(s, ax=None, offset=0.07, right=False, *args, **kwargs):
def ylabel(s, ax=None, offset=0.07, right=False, *args, **kwargs):
    """Y axis label at fixed offset, useful for stack plots"""
    if ax is None:
        ax = plt.gca()
    plt.sca(ax)
    plt.ylabel(s)
    #
    # xy = (-offset, 0.5)
    # if right:
    #     xy = (1. + offset, 0.5)
    # ax.annotate(s, xy, xycoords='axes fraction', rotation='vertical',
    #     ha='center', va='center', *args, **kwargs)


def corner_label(n, ax=None, location='top left', color='black', **kwargs):
    """Label the corner of an axis.  n can be an iterator, or a string"""
    if hasattr(n, 'next'):
        val = next(n)
    else:
        val = n

    if ax is None:
        ax = plt.gca()
    else:
        plt.sca(ax)

    if location.lower() == 'top left':
        plt.annotate(val, (0., 1.), xycoords='axes fraction', ha='left', va='top', color=color, **kwargs)

    if location.lower() == 'top right':
        plt.annotate(val, (1., 1.), xycoords='axes fraction', ha='right', va='top', color=color, **kwargs)

    if location.lower() == 'bottom left':
        plt.annotate(val, (0., 0.), xycoords='axes fraction', ha='left', va='bottom', color=color, **kwargs)

    if location.lower() == 'bottom right':
        plt.annotate(val, (1., 0.), xycoords='axes fraction', ha='right', va='bottom', color=color, **kwargs)


class CircularLocator(mpl.ticker.Locator):
    """Nice multiples for trigonometric plots"""
    def __init__(self, units='degrees', modulo=True, prune=None, nmax=5):

        self._units = units
        self._prune = prune
        self._nmax = nmax

        if self._units == 'degrees':
            self._nice = [0.01, 0.05, 1.0, 2.0, 5.0, 10.0, 30.0, 45.0, 90.0, 180.0, 360.0]
        elif self._units == 'radians':
            self._nice = [np.pi/32, np.pi/16, np.pi/8, np.pi/4, np.pi/2, np.pi, 2. * np.pi]
        elif self._units == 'natural':
            self._nice = [0.001, 0.005, 0.01,0.05, 0.1, 0.2, 0.5, 1.0]

    def bin_boundaries(self, vmin, vmax):
        dv = vmax - vmin
        best_s = None
        for s in self._nice:
            if int(dv / s) > self._nmax:
                continue
            best_s = s
            break

        if best_s is None:
            return np.linspace(vmin, vmax, self._nmax)

        offset = int(vmin / best_s) + 1
        return np.arange(offset * best_s, vmax, best_s)

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        vmin, vmax = matplotlib.transforms.nonsingular(vmin, vmax, expander = 1.)

        locs = self.bin_boundaries(vmin, vmax)

        prune = self._prune
        if prune=='lower':
            locs = locs[1:]
        elif prune=='upper':
            locs = locs[:-1]
        elif prune=='both':
            locs = locs[1:-1]
        return self.raise_if_exceeds(locs)

def boxplot(x,y, color='k', width=None, width_scale=1.,
                    lw=None, linewidth=None, *args, **kwargs):
    "Box and whisker plot."
    import scipy.stats

    if linewidth is not None:
        _lw = linewidth
    elif lw is not None:
        _lw = lw
    else:
        _lw = 1.

    if isinstance(x, np.ndarray):
        x0 = np.mean(x[np.isfinite(x)])
        if width is None:
            width = 0.5 * (np.nanmax(x) - np.nanmin(x)) * width_scale
    else:
        x0 = x
        if width is None:
            width = 1.
    yy = y[np.isfinite(y)]

    ym = np.median(yy)
    ylow = scipy.stats.scoreatpercentile(yy, 25)
    yhigh = scipy.stats.scoreatpercentile(yy, 75)
    ymin = np.amin(yy)
    ymax = np.amax(yy)

    lines = []
    lines.extend( plt.plot((x0, x0), (ylow, ymin), color=color, lw=_lw,
            *args, **kwargs))
    lines.extend( plt.plot((x0, x0), (yhigh, ymax), color=color, lw=_lw,
            *args, **kwargs))

    lines.extend( plt.plot(x0 + np.array((-1,-1,1,1,-1)) * width,
        (ylow, yhigh, yhigh, ylow, ylow), color=color, lw=_lw, *args, **kwargs))

    lines.extend( plt.plot((x0 + width, x0 - width), (ym, ym), color=color,
                lw=2.*_lw, *args, **kwargs))

    return lines

def map_along_line(x, y, q, ax=None, cmap=None, norm=None,
            time=None, max_step=1., missing=np.nan,
            new_timebase=None,
            **kwargs):
    """Map some quantity q along x,y as a coloured line.
With time set, perform linear interpolation of x,y,q onto new_timebase
filling with missing, and with max_step."""

    if ax is None:
        ax = plt.gca()

    if x.shape != y.shape:
        raise ValueError('Shape mismatch')
    if x.shape != q.shape:
        raise ValueError('Shape mismatch')

    if time is not None:
        if new_timebase is None:
            new_timebase = np.arange(time[0], time[-1], np.min(np.diff(time)))

        # Bit redundant
        x = interp_safe(new_timebase, time, x, max_step=max_step, missing=missing)
        y = interp_safe(new_timebase, time, y, max_step=max_step, missing=missing)
        q = interp_safe(new_timebase, time, q, max_step=max_step, missing=missing)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm, **kwargs)

    lc.set_array(q)
    plt.gca().add_collection(lc)

    return lc

def add_labelled_bar(time, q, values_to_mark=None, ax=None,
            offset=0.05,  height=0.0001,
            formatter=None, top=False, label=''):
    """Plot a line-axis below main axes, indicating quantity q.

Args:

Example:


    """

    if ax is None:
        ax = plt.gca()
    else:
        plt.sca(ax)
    fig = ax.figure

    if not isinstance(time, np.ndarray):
        _time = np.array(time)
    else:
        _time = time

    if not isinstance(q, np.ndarray):
        _q = np.array(q)
    else:
        _q = q

    p = ax.get_position()
    if top:
        new_ax = fig.add_axes([p.x0, p.y1 + offset, p.x1 - p.x0, height],
            yticks=[])
    else:
        new_ax = fig.add_axes([p.x0, p.y0 - offset, p.x1 - p.x0, -height],
            yticks=[])

    plt.sca(new_ax)
    plt.xlim(ax.get_xlim())
    if formatter is None:
        formatter = lambda x: "%.1f" % x # a guess

    if values_to_mark is None:
        # Print at the same locations as the parent axis
        xticks = ax.xaxis.get_ticklocs()
        xnames = [formatter(x) for x in np.interp(xticks, _time, _q)]
        new_ax.set_xticks(xticks)
        new_ax.set_xticklabels(xnames)

    else:
        new_ticklocs = []
        new_ticknames = []
        for v in values_to_mark:
            comp = _q > v
            for i in range(0, _q.shape[0]-1):
                if comp[i] != comp[i+1]:
                    new_ticklocs.append(np.interp(v, _q[[i,i+1]],
                            _time[[i,i+1]]))
                    new_ticknames.append(formatter(v))
        new_ax.set_xticks(new_ticklocs)
        new_ax.set_xticklabels(new_ticknames)


    plt.annotate(label + '  ', (0., -1), xycoords='axes fraction', va='center', ha='right', clip_on=False)

    plt.sca(ax)
    return new_ax

def plot_planet(radius=1., orientation='dawn', ax=None, origin=(0.,0.), scale=0.96,
                    edgecolor='black', facecolor='white', resolution=256, zorder=None, **kwargs):
    """`orientation' specifies view direction, one of 'noon', 'midnight',
anything else being used as the terminator.  `scale' is used to adjust the 'white' polygon
size in relation to the radius for aesthetics """
    if zorder is None:
        zorder = -9999

    if ax is None:
        ax = plt.gca()

    o = orientation.lower()
    if o == 'noon':
        ax.add_patch(plt.Circle(origin, radius, fill=True, zorder=zorder,
                        facecolor=facecolor, edgecolor=edgecolor, **kwargs))
        return
    elif o == 'midnight':
        ax.add_patch(plt.Circle(origin, radius, fill=True, color=edgecolor, zorder=zorder, **kwargs))
        return

    theta = np.linspace(0., np.pi, resolution) - np.pi/2.
    xy = np.empty((resolution, 2))
    xy[:,0] = scale * radius * np.cos(theta)
    xy[:,1] = scale * radius * np.sin(theta)
    if o == 'dusk':
        xy[:,0] = -1. * xy[:,0]

    semi = plt.Polygon(xy, closed=True, color=facecolor, fill=True)
    ax.add_patch(plt.Circle(origin, radius, fill=True, color=edgecolor, zorder=zorder+1, **kwargs))
    ax.add_patch(semi)

class DJAPage(object):
    """
Provides a set of stacked panels, with callbacks to link the xlimits.
Additional items such as orbit bars can be registered after creation.
    """
    def __init__(self, npanels=1, ratios=None, figure_kwds=None, vspace=0.1,
                        bounds=None, label_all=False):
        super(DJAPage, self).__init__()
        self.npanels = npanels
        if ratios is None:
            ratios = [1. for n in range(self.npanels)]
        else:
            self.npanels = len(ratios)

        self.ratios = ratios
        self.vspace = vspace
        if bounds is None:
            bounds = [0.15, 0.9, 0.15, 0.85]

        self.bounds = bounds
        if figure_kwds is None:
            figure_kwds = {}

        self.fig = plt.figure(**figure_kwds)

        self.axs = []
        self.cids = []
        if self.npanels > 1:
            dh = self.vspace / (self.npanels - 1)
        else:
            dh = 0.

        height = bounds[3] - bounds[2] - dh * (self.npanels - 1) # page fracs
        width  = bounds[1] - bounds[0] #

        total = float(sum(ratios))
        scale = height / total
        top = bounds[3]

        for i in range(self.npanels):
            tmp = ratios[i] * scale
            new_pos = [
                bounds[0], top - tmp , bounds[1] - bounds[0], tmp
            ]
            print(new_pos)
            top = new_pos[1] - dh
            ax = self.fig.add_axes(new_pos)
            cid = ax.callbacks.connect('xlim_changed', self.xlim_changed)
            self.axs.append(ax)
            self.cids.append(cid)

            if not label_all:
                if i != (self.npanels - 1):
                    ax.xaxis.set_major_formatter(plt.NullFormatter())

            if i == 0:
                self.top_axes = ax
            if i == (self.npanels - 1):
                self.bottom_axes = ax

    def xlim_changed(self, a):
        """Called when xaxis changes."""

        new_xlim = a.get_xlim()

        # existing callbacks are cleared, and then reinstated
        # to avoid recursion
        for ax in self.axs:
            for c in self.cids:
                ax.callbacks.disconnect(c)

        self.cids = []
        for ax in self.axs:
            ax.set_xlim(*new_xlim)
            cid = ax.callbacks.connect('xlim_changed', self.xlim_changed)
            self.cids.append(cid)

    def register_new_axis(self, ax):
        if not ax in self.axs:
            cid = ax.callbacks.connect('xlim_changed', self.xlim_changed)
            self.cids.append(cid)
            self.axs.append(ax)

def terminator(sslat, sslon, N=1024, repeat=True):
    """Compute a terminator line for a given sub-solar latitude and longitude (radians) over N points.
    Returns latitude and longitude in radians.
    """

    # Parameter
    t = np.arange(-np.pi, np.pi - 0.0001, 2. * np.pi/N)

    # Circle about the pole
    x = np.cos(t)
    y = np.sin(t)
    z = np.zeros_like(t)

    # Rotate about y-axis by angle
    rot = np.pi/2. - sslat
    xx = x * np.cos(rot) + z * np.sin(rot)
    yy = y
    zz = x * -np.sin(rot) + z * np.cos(rot)

    # Convert to polar
    r = np.sqrt(xx**2 + yy**2 + zz**2)
    theta = np.arccos(zz / r)
    phi = np.arctan2(yy,xx)

    # Spin to correct longitude and cleanup
    phi = (phi + sslon) % (2. * np.pi)
    phi = (phi + (2 * np.pi)) % (2. * np.pi)
    phi = np.unwrap(phi)

    inx = np.argsort(phi)
    phi = phi[inx]
    lat = np.pi/2. - theta
    lat = lat[inx]

    if repeat:
        return np.hstack((lat, lat, lat)), np.hstack((phi - 2.*np.pi, phi, phi + 2.*np.pi))

    return lat, phi

def test_axes_stuff():
    fig = DJAPage(ratios=[2.,1., 0.63])
    plt.sca(fig.top_axes)
    x = np.arange(360.)
    y = np.sin(x * np.pi/180. + 2.)
    plt.plot(x, y, 'r-')
    plt.sca(fig.bottom_axes)
    mb = add_labelled_bar(x,x)
    fig.register_new_axis(mb)
    plt.xlim(0., 360.)
    plt.show()

if __name__ == '__main__':
    test_axes_stuff()
