=============================================================
Python library for handling time formats, spice kernels, and other space-related
things.
=============================================================

Introduction
------------
A suite of tools supporting analyses of spacecraft data.

Routines are provided for managing, manipulating, and plotting versus time.  The principal format used is the Ephemeris Time used by NASA's NAIF SPICE package.  Conversion routines beyond those included in the NAIF software (accessed using the SpiceyPy library) are provided.  Helper classes for making time series plots using MatPlotLib are given.


Requirements
------------

Tested against the anaconda python distrubition (v 3.5)
SpiceyPy library required for NAIF spice interface
SpacePy  library required for CDF access

Installation
------------

0) Satisfy requirements above
1) Add this module into your python path.
3) Set the local directory that will be used for storing data:
    export SC_DATA_DIR="~/data"
4) Run tests?

Examples of use
---------------
