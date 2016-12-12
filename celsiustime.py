"""Time conversion routines, in particular wrappers around spice routines.

The internal time type is SPICEET, i.e. seconds since J2000. All conversions are defined against this. For all realistic uses, precision should be easily around the ms level. More precise calculations need to be performed some other way.

"""

import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
import matplotlib.transforms
import matplotlib.ticker
from scipy.interpolate import griddata
import datetime
import os
import sys
import functools

import time as py_time

import pickle
import spiceypy
import glob

VALID_FORMATS  = ['SPICEET', 'AUTO', 'DATETIME', 'CELSIUSTIME',
                'MPLNUM', 'UTCSTR', 'DOY2004', 'AISSCET', 'MATLABTIME']

MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# The SPICEET reference epoch for each class of date
spiceet_epoch  = 0.0
#2000-001T11:58:55.816
datetime_epoch = datetime.datetime(2000, 1, 1, 11, 58, 55, 816000)
mplnum_epoch   = mpl.dates.date2num(datetime_epoch)
utcstr_epoch   = "2000-001T11:58:55.816"
doy2004_epoch  = -126187264.184/86400.0

## note, need to set spiceypy.tsetyr(0000) to compute,
# changed sign to stick with conventions below
matlabtime_epoch = 730486.4995233236

#1958-001T00:00:00Z
# et = mex_time.spiceet('1958-001T00:00:00')
# = -1325419158.816061 s =  -15340.49952333404 days
aisscet_epoch = 15340.49952333404

# Some other constants
max_orbits = 20000 # A guess for now

LSK_FURNSH_KERNELS_DONE = False

def time_convert(time, output_format='SPICEET',
        input_format='AUTO', output_kwargs={}, input_kwargs={}):
    """Convert input time from input_format to output_format, by calling the appropriate :input_to_ouput: function. If neccessary, conversion proceeds via intermediate step of converting input to SPICEET.

Args:
    time: Input time in some format.  Float interpreted as SPICEET, string as UTCSTR, datetime object.
    input_format: one of VALID_FORMATS, default is to guess based on :time:.
    output_format: default SPICEET.
    output_kwargs, input_kwargs: optional arguments to conversion routines."""
    # global VALID_FORMATS

    if time is None:
        return None

    if not input_format.upper() in VALID_FORMATS:
        raise ValueError('Invalid input format: ' + str(input_format))

    if not output_format.upper() in VALID_FORMATS:
        raise ValueError('Invalid output format: ' + str(output_format))

    # Passthrough
    if input_format == output_format:
        return time

    if input_format == 'AUTO':
        if isinstance(time, float):
            input_format = 'SPICEET'
        elif isinstance(time, datetime.datetime):
            input_format = 'DATETIME'
        elif isinstance(time, str):
            input_format = 'UTCSTR'
        elif isinstance(time, bytes):
            input_format = 'UTCSTR'
        elif isinstance(time, CelsiusTime):
            input_format = 'CELSIUSTIME'

    if input_format == 'AUTO':
        raise ValueError('Could not detrmine input format automatically')

    if input_format != 'SPICEET':
        fname = input_format.lower() + '_to_spiceet'
        as_et = globals()[fname](time, **input_kwargs)
    else:
        as_et = time

    if output_format == 'SPICEET':
        return as_et

    fname = 'spiceet_to_' + output_format.lower()

    return globals()[fname](as_et, **output_kwargs)

class CelsiusTime(object):
    """An object that represents a Spiceet and calculates its various representations.
    """
    def __init__(self, t):
        """Create a CelsiusTime object.

Args:
    t: anything that can be converted to a spiceet representation.

Returns:
    CelsiusTime instance.

Example:
    o = CelsiusTime("2015-01-01T00:37")
    print(o.year, o.hour)
    o.spiceet = now() # redefine, automatically recompute
    print(o.hour, o.minute)

"""
        super(CelsiusTime, self).__init__()
        if not isinstance(t, float):
            et = spiceet(t)
        else:
            et = t

        self.spiceet = et

    @property
    def spiceet(self):
        return self._et

    @spiceet.setter
    def spiceet(self, et):
        self._et = et
        self._compute()

    def _compute(self):
        if self._et is None:
            raise ValueError("Time not set")

        self.isoc = utcstr(self._et, 'ISOC')
        self.isod = utcstr(self._et, 'ISOD')

        self.year = int(self.isoc[0:4])
        self.month = int(self.isoc[5:7])
        self.day = int(self.isoc[8:10])
        self.hour = int(self.isoc[11:13])
        self.minute = int(self.isoc[14:16])
        self.second = float(self.isoc[17:])

        self.doy = int(self.isod[5:8])
        self.month_name = MONTHS[self.month - 1]
        self.month_name_short = self.month_name[:3].upper()

    def __repr__(self):
        return "CelsiusTime(%s)" % self.isod

    def __str__(self):
        return self.isod

    def __lt__(self, o):
        if isinstance(o, CelsiusTime):
            return self._et < o._et
        return self._et < o

    def __gt__(self, o):
        if isinstance(o, CelsiusTime):
            return self._et > o._et
        return self._et > o

    def __le__(self, o):
        if isinstance(o, CelsiusTime):
            return self._et <= o._et
        return self._et <= o

    def __ge__(self, o):
        if isinstance(o, CelsiusTime):
            return self._et >= o._et
        return self._et >= o

    def __eq__(self, o):
        if isinstance(o, CelsiusTime):
            return self._et == o._et
        return self._et == o

    def __ne__(self, o):
        if isinstance(o, CelsiusTime):
            return self._et != o._et
        return self._et != o


# Conversion routines follow:
def utcstr_to_spiceet(time):
    """passthrough to spiceypy.utc2et"""
    if isinstance(time, bytes):
        return spiceypy.utc2et(time.decode('utf-8'))
    return spiceypy.utc2et(time)

def spiceet_to_utcstr(time, fmt='ISOD', precision=5):
    """Time conversion function"""

    length = 22
    if fmt == 'C':
        length = 22
    if not np.isfinite(time):
        raise ValueError("Supplied ET is not finite")

    if fmt == 'CAL':
        raise RuntimeError("Don't use spiceypy.etcal - this is a bad idea.")
        # return spiceypy.etcal(time)
    return spiceypy.et2utc(time, fmt, precision, length + precision)

def mplnum_to_spiceet(time):
    """Time conversion function"""
    return (time - mplnum_epoch) * 86400.0 + spiceet_epoch

def spiceet_to_mplnum(time):
    """Time conversion function"""
    return (time - spiceet_epoch) / 86400.0 + mplnum_epoch

def doy2004_to_spiceet(time):
    """Time conversion function"""
    return (time - doy2004_epoch) * 86400.0 + spiceet_epoch

def spiceet_to_doy2004(time):
    """Time conversion function"""
    return (time - spiceet_epoch) / 86400.0 + doy2004_epoch

def datetime_to_spiceet(t):
    """Time conversion function. N.b.: datetime is not leapseconds-aware"""
    # datetime to string to spiceet
    x = "%04d-%02d-%02dT%02d:%02d:%05.7f" % (
        t.year, t.month, t.day, t.hour, t.minute, 1.0*t.second + t.microsecond/1e6)
    return utcstr_to_spiceet(x)

def spiceet_to_datetime(time):
    """Time conversion function. N.b.: datetime is not leapseconds-aware"""
    # spiceet to string to datetime
    x = spiceet_to_utcstr(time, 'ISOC')
    return datetime.datetime(int(x[0:4]), int(x[5:7]), int(x[8:10]),
            int(x[11:13]), int(x[14:16]), int(x[17:19]), int(float(x[19:])*1e6))

def matlabtime_to_spiceet(time):
    """Time conversion function"""
    return (time - matlabtime_epoch) * 86400.0 + spiceet_epoch

def spiceet_to_matlabtime(time):
    """Time conversion function"""
    return (time - spiceet_epoch) / 86400.0 + matlabtime_epoch

def aisscet_to_spiceet(time):
    """Time conversion function"""
    return (time - aisscet_epoch) * 86400.0 + spiceet_epoch

def spiceet_to_aisscet(time):
    """Time conversion function"""
    return (time - spiceet_epoch) / 86400.0 + aisscet_epoch

def spiceet_to_celsiustime(time):
    """Time conversion function"""
    return CelsiusTime(time)

def celsiustime_to_spiceet(time):
    """Time conversion function"""
    return time.spiceet


# These are then just helper functions that have input_format='AUTO' hardcoded:
def utcstr(time, fmt=None, **kwargs):
    """Helper function, convert input to UTCSTR"""
    if fmt is None:
        return time_convert(time, output_format='UTCSTR', input_format='AUTO')
    else:
        if not isinstance(time, float):
            et = spiceet(time)
        else:
            et = time
        return spiceet_to_utcstr(et, fmt=fmt, **kwargs)

def spiceet(time):
    """Helper function, convert input to SPICEET."""
    return time_convert(time, output_format='SPICEET', input_format='AUTO')

def mplnum(time):
    """Helper function, convert input to MPLNUM."""
    return time_convert(time, output_format='MPLNUM', input_format='AUTO')

def doy2004(time):
    """Helper function, convert input to DOY2004."""
    return time_convert(time, output_format='DOY2004', input_format='AUTO')

def now():
    """Return the current time as a SPICEET, according to the datetime library"""
    return datetime_to_spiceet(datetime.datetime.now())

def spiceet_to_strdict(t):
    """Return a dictionary with input time :t: elements expressed as strings.

YYYY, YY, MON, DD, HH, MM, SS, DOY returned.
    """
    s = spiceet_to_utcstr(t, 'C')
    d = {
            'YYYY':s[0:4],
            'YY':s[2:4],
            'MON':s[5:8],
            'DD':s[9:11],
            'HH':s[12:14],
            'MM':s[15:17],
            'SS':s[18:]
        }
    s = spiceet_to_utcstr(tn)
    d['DOY'] = s[5:8]
    return d

class Orbit(object):
    """Class that describes a spacecraft orbit timing."""

    def __init__(self, number=None, apoapsis=None, periapsis=None, start=None,
        name=None):
        """Initialize.

Args:
    number (int): orbit number
    name (str): spacecraft name, to avoid confusion
    start, periapsis, apoapsis (SPICEET): respective times
        """
        super(Orbit, self).__init__()
        self.number = number

        if name is None:
            raise ValueError("name cannot be none")

        self.name = name

        # Assume we're provided with all parameters required
        if apoapsis is not None:
            self.apoapsis = apoapsis
            self.periapsis = periapsis
            self.start = start
            self.finish = self.apoapsis

            self.next = None
            self.previous = None

    def __str__(self):
        daterep = lambda d: str(d) + ' [' + utcstr(d, 'C') + ', DOY=' + utcstr(d)[5:8] + ']\n'
        val = repr(self) + '\n'
        val += '  Spacecraft = %s\n' % self.name
        val += '  Number = %d\n' % self.number
        val += '  Start = ' + daterep(self.start)
        val += '  Periapsis = ' + daterep(self.periapsis)
        val += '  Finish = ' + daterep(self.finish)
        return val


class OrbitDict(dict):
    """A subclass of dict to deal with Orbits, allowing indexing with a float spiceet, returning the orbit which contains that time.

Example:
    t = celsius.spiceet("2015-01-01T00:00")
    o = maven.orbits[t]
    print(o)"""
    def __init__(self, *args):
        super(OrbitDict, self).__init__(*args)

    def __getitem__(self, key):
        if isinstance(key, float):
            # This is incredibly lazy, but so am I
            # one day, someone will write an index
            for k, v in super(OrbitDict, self).items():
                if (v.start <= key) & (v.finish >= key):
                    return v
            raise KeyError("float key=%f not within range" % key)

        return super(OrbitDict, self).__getitem__(key)

    def __setitem__(self, key, value):
        if not isinstance(value, Orbit):
            raise ValueError("Only allowed to add orbits")

        if self:
            if value.name != self[list(self.keys())[0]].name:
                raise ValueError(
                "Cannot mix Spacecraft in a single OrbitDict: %s, %s" %
                        (value.name, self[list(self.keys())[0]].name))

        super(OrbitDict, self).__setitem__(key, value)


if __name__ == '__main__':
    import matplotlib.pylab as plt
    import matplotlib as mpl

    plt.close('all')

    day = 86400.
    extents = (60., 600, 3600., day/2., day, 2 * day, 10. * day,
            20. * day, 100. * day, 365 * day, 365 * 5 * day, 365 * day * 20.)

    fig, axs = plt.subplots(len(extents), 1, figsize=(8, 13))

    print(spiceet_to_strdict(0.))

    for i, e in enumerate(extents):
        print('---------')
        print(i, e)
        ax = axs[i]
        plt.sca(ax)
        plt.plot((0., e), (0., 1.),'k-')
        plt.plot((0., e), (1., 0.),'k-')
        plt.xlim(0., e)

        l = SpiceetLocator(verbose=True)
        f = SpiceetFormatter()
        ax.xaxis.set_major_locator(l)
        ax.xaxis.set_major_formatter(f)

        #
        # ax2 = plt.twiny()
        # plt.xlim(0., e)
        #
        # l2 = mangled_mpl_dates.AutoDateLocator()
        # f2 = mangled_mpl_dates.AutoDateFormatter(l)
        # ax2.xaxis.set_major_locator(l2)
        # ax2.xaxis.set_major_formatter(f2)

        # l = celsius.SpiceetCalendarLocator()
        #
        # if e > 86400. * 100.:
        #     l = celsius.SpiceetYearLocator()
        # f = celsius.SpiceetFormatter(calendar=True)
        # ax2.xaxis.set_major_locator(l)
        # ax2.xaxis.set_major_formatter(f)

    plt.subplots_adjust(hspace=5.)

    plt.show()
