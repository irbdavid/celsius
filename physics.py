"""Physical constants and conversions.  SI Units.  CGS banned."""

import numpy as np

eV_in_K = 11604.505

electron_cyclotron_freq_1nT_in_hz = 175.882009 # Hz
proton_cyclotron_freq_1nT_in_hz = 0.0957883358 # Hz


def debye_length_m(electron_density, electron_temperature):
    """Calculate electron Debye length for electron density in cm^-3, electron_temperature in Kelvin"""
    return 0.069 * np.sqrt(electron_temperature / electron_density)

def fp_to_ne(fp, error=None):
    """Convert plasma frequency in Hz to electron density in cm^-3"""
    if error is not None:
        return ((fp / 8980.0)**2.0, 2.0 * error * fp/8980.0**2.0)
    return (fp / 8980.0)**2.0

def ne_to_fp(ne, error=None):
    """Convert electron density in cm^-3 to plasma frequency in Hz"""
    if error is not None:
        return np.sqrt(ne) * 8980.0, 8980. * 0.5 * error / np.sqrt(ne)
    return np.sqrt(ne) * 8980.0

def td_to_modb(td, error=None):
    """Convert cyclotron period in s to magnetic field intensity in T"""
    if error is not None:
        return (2.0 * np.pi / (1.758820150E11 * td), 2.0 * np.pi / (1.758820150E11 * td**2.) * error)
    return 2.0 * np.pi / (1.758820150E11 * td)

def modb_to_td(b):
    """Convert magnetic field intensity in T to cyclotron period in s"""
    return 2.0 * np.pi / (1.758820150E11 * b)

def electron_gyroradius(b, e):
    "e in eV, b in nt, result in km"
    return 1.1E2 * np.sqrt(e/1e3) / b
