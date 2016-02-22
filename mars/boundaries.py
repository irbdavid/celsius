import numpy as np
import pylab as plt
from . import constants

def hold_xylim(func):
    def wrapped(*args, **kwargs):
        xlim = plt.xlim()
        ylim = plt.ylim()
        out = func(*args, **kwargs)
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        return out
    return wrapped

@hold_xylim
def plot_mpb_model_sza(fmt='k-', model="TROTIGNON06", shadow=True,
            return_values=False, **kwargs):
    if model == "VIGNES00":
        t = np.arange(0., np.pi, 0.01)
        x = 0.78 + 0.96 * np.cos(t) / (1 + 0.9 * np.cos(t))
        y =       0.96 * np.sin(t) / (1 + 0.92 * np.cos(t))
    elif model == 'EDBERG08':
        t = np.arange(0., np.pi, 0.01)
        x = 0.86 + 0.9 * np.cos(t) / (1 + 0.92 * np.cos(t))
        y =       0.9 * np.sin(t) / (1 + 0.92 * np.cos(t))
    elif model == "DUBININ06":
        t = np.arange(0., np.pi, 0.01)
        x = 0.70 + 0.96 * np.cos(t) / (1 + 0.9 * np.cos(t))
        y =       0.96 * np.sin(t) / (1 + 0.9 * np.cos(t))
    elif model == "TROTIGNON06":
        t = np.arange(0., np.pi, 0.01)
        # x > 0:
        x1 = 0.64 + 1.08 * np.cos(t) / (1 + 0.77 * np.cos(t))
        y1 =        1.08 * np.sin(t) / (1 + 0.77 * np.cos(t))
        # x < 0:
        x2 = 1.60 + 0.528 * np.cos(t) / (1 + 1.009 * np.cos(t))
        y2 =        0.528 * np.sin(t) / (1 + 1.009 * np.cos(t))

        inx1 = x1 > 0.0
        inx2 = x2 < 0.0

        x = np.hstack((x1[inx1], x2[inx2]))
        y = np.hstack((y1[inx1], y2[inx2]))

    alt = np.sqrt(x*x + y*y) - 1.
    sza = np.arctan2(y, x) / np.pi * 180.

    if shadow:
        y = np.arange(0., 2500., 10.)
        x = 90. + 180. /np.pi * np.arccos(3376. / (3376. + y))

        if return_values:
            return ((sza, alt * constants.mars_mean_radius_km), (x, y))
        plt.plot(x, y, fmt[0] + '--', **kwargs)

    if return_values:
        return (sza, alt * constants.mars_mean_radius_km)

    plt.plot(sza, alt * constants.mars_mean_radius_km, fmt, **kwargs)
