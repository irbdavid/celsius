"""Time module, with SPICE ET as the reference epoch.

Uses FTP to grab the required leapseconds kernel when imported.
"""

from .celsiustime import *
from .time_axes import *
from .data import *
from .physics import *
from .plot import *
from .spice import *

from os import getenv
from os.path import expanduser
import time
import stat
import re
from spiceypy import furnsh
import ftplib

__author__ = "David Andrews"
__copyright__ = "Copyright 2015, David Andrews"
__license__ = "MIT"
__version__ = "1.0"
__email__ = "david.andrews@irfu.se"

SERVER = 'naif.jpl.nasa.gov'
PATH = '/pub/naif/generic_kernels/lsk'
LSK_FILENAME = getenv("SC_DATA_DIR", default=expanduser('~/data/')) + \
        'latest.tls'

age = time.time() - os.stat(LSK_FILENAME)[stat.ST_MTIME]
if SERVER and (age > 86400*100):

    print('Obtaining newest leap-seconds kernel from ' + SERVER + '...')
    ftp = ftplib.FTP(SERVER)
    ftp.login()
    ftp.cwd(PATH)
    files = ftp.nlst()
    prog = re.compile("naif.*.tls$")
    for f in files:
        if prog.match(f):
            print('DL %s to %s...' % (SERVER + PATH + '/' + f, LSK_FILENAME))
            ftp.retrbinary("RETR " + f,
                 open(LSK_FILENAME, 'wb').write)
            print('...done')
    ftp.quit()

spiceypy.furnsh(LSK_FILENAME)
