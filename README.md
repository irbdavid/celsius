=============================================================
Python library for handling time formats, spice kernels, and other space-related things.
=============================================================

Introduction
------------
A suite of tools supporting analyses of spacecraft data.

Routines are provided for managing, manipulating, and plotting versus time.  The principal format used is the Ephemeris Time used by NASA's NAIF SPICE package.  Conversion routines beyond those included in the NAIF software (accessed using the SpiceyPy library) are provided.  Helper classes for making time series plots using MatPlotLib are given.


Requirements
------------

1. Tested against the anaconda python distribution (v 3.5)
2. SpiceyPy library required for NAIF spice interface
3. SpacePy  library required for CDF access

Installation
------------

1. Satisfy requirements above
2. Add this module into your python path.
3. Set the local directory that will be used for storing data:
    export SC_DATA_DIR="~/data"
4. Run tests?

Examples of use
---------------
```python
import celsius
import matplotlib.pyplot as plt

t0 = celsius.spiceet("2015-01-01T00:00")
t1 = celsius.now()

plt.plot((t0, t1), (1., 1.))
celsius.setup_time_axis()

print(celsius.utcstr(t1))
print(celsius.utcstr(t1, 'ISOC'))

t_obj = celsius.CelsiusTime(t0)
print(t_obj.year, t_obj.doy)

```
