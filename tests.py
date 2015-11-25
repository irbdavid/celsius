
from celsius import *

import pylab as plt
import numpy as np

def test():
    pass

if __name__ == '__main__':
    plt.close('all')

    e = 86400 * 365. * 4.

    for i in range(100):
        t0 = spiceet("2000-%02d-01T00:00" % i)
        plt.plot( (t0, t0), (0., 1.), 'r-')

    for i in range(2000,2010):
        t0 = spiceet("%04d-01-01T00:00" % i)
        plt.plot( (t0, t0), (0., 1.), 'k-')

    for i in range(200):
        t0 = spiceet("2000-%03dT00:00" % i)
        plt.plot( (t0, t0), (0., 1.), 'g-')

    for i in range(24):
        t0 = spiceet("2000-001T%02d:00" % i)
        plt.plot( (t0, t0), (0., 1.), 'b-')

    for i in range(60):
        t0 = spiceet("2000-002T00:%02d" % i)
        plt.plot( (t0, t0), (0., 1.), 'c-')

    plt.xlim(0.01, e)

    l1 = SpiceetLocator(calendar=False, nmax=6)

    # l1.bin_boundaries

    f1 = SpiceetFormatter(locator=l1, label=True)
    plt.gca().xaxis.set_major_locator(l1)
    plt.gca().xaxis.set_major_formatter(f1)


    plt.show()
