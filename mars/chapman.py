import numpy as np
import scipy.optimize as opt
from scipy.integrate import trapz, simps
import matplotlib.pylab as plt
import celsius
import os.path

ais_spacing_seconds = 7.543
ais_number_of_delays = 80
ais_delays = (np.arange(80) * 91.4 + 91.4 + 162.5) * 1E-6
ais_max_delay = 7474.5 * 1.0E-6
ais_min_delay = 253.9 * 1.0E-6
ais_vmin=-16
ais_vmax=-13
speed_of_light_kms = 299792.458

# Should replace these with physical values not just approximations
def fp_to_ne(fp, error=None):
    if error is not None:
        return ((fp / 8980.0)**2.0, 2.0 * error * fp/8980.0**2.0)
    return (fp / 8980.0)**2.0

def ne_to_fp(ne, error=None):
    if error is not None:
        return np.sqrt(ne) * 8980.0, 8980. * 0.5 * error / np.sqrt(ne)
    return np.sqrt(ne) * 8980.0

def get_f10_7(times, fname=None, return_all=False):
    if not fname:
        fname = os.path.expandvars('$SC_DATA_DIR/omni/omni2_daily_19560.lst')

    data = np.loadtxt(fname).T
    years = dict([(k, celsius.spiceet('%d-001T00:00' % k)) for k in np.unique(data[0])])
    time = np.array([years[d] + (v-1.) * 86400. for d, v in zip(data[0], data[1])])
    if return_all:
        return time, 1.0 * data[3]
    return np.interp(times, time, data[3], left=np.nan, right=np.nan)

class IonosphericModel(object):
    """Base class for ionospheric models.  At the minimum, __call__ must be overridden"""
    def __init__(self):
        super(IonosphericModel, self).__init__()

    def __call__(self, alt, theta=0.):
        raise NotImplementedError("Derived must override")

    def get_tec(self, theta=0., hmax=np.inf):
        raise NotImplementedError("Derived must override")

    def get_params(self):
        raise NotImplementedError("Derived must override")

    def set_params(self):
        raise NotImplementedError("Derived must override")

    def ais_response(self, frequencies=None, sc_altitude=None, sc_theta=0.,
                        frequency_range=(0.1, 7.5), frequency_resolution=0.05,
                        altitude_resolution=0.05):
        """Compute the appearance of the model in the AIS instrument, by
propagating rays through it and computing the delay to the reflection point.
This is done empirically, rather than analytically such that models can be more
or less arbitrarily defined"""
        # The inversion must be done again, in reverse to produce the 'dispersed' delays?
        # The sounding frequencies
        if frequencies is None:
            frequencies = np.arange(frequency_range[0], frequency_range[1], frequency_resolution) * 1E6
        else:
            frequency_range = (frequencies[0], frequencies[-1])

        if sc_altitude is None:
            sc_altitude = 500.

        if ~np.isfinite(sc_altitude * sc_theta):
            raise ValueError("Need sensible altitude, sza")

        altitudes = np.arange(sc_altitude, 0., -1. * np.abs(altitude_resolution))
        # print 'Computing AIS Response at %f km, %f deg' % (sc_altitude, np.rad2deg(sc_theta))
        ne = self.__call__( altitudes, sc_theta)
        fp = ne_to_fp(ne) # the plasma frequency vs altitude
        results = np.zeros_like(frequencies)

        # This is wrong
        # prevmax = - np.inf
        # for i in range(altitudes.shape[0]):
        #     if fp[i] <= prevmax:
        #         fp[i] = prevmax
        #     else:
        #         prevmax = fp[i]

        fp[-1] = 1e99

        for i in range(results.shape[0]):
            inx = (fp[:-1] < frequencies[i])
            if (~np.any(inx)) or (not inx[0]):
                continue
            bad_inx, = np.where(~inx)
            if bad_inx.shape[0] > 0:
                inx = inx[0:bad_inx[0]-1]

            if inx.shape[0] == 0: continue
            results[i] = 2.0 / speed_of_light_kms * simps(1. / (1. - (fp[inx]/frequencies[i])**2.)**0.5, altitudes[inx])

        return results, frequencies

    def invert_density(self, *args, **kwargs):
        """Pseudonym for ais_response()"""
        return self.ais_response(*args, **kwargs)

    def __add__(self, other):
        """Returns a ChainedModels instance containing the two models"""
        return ChainedModels((self, other))

class ChapmanLayer(IonosphericModel):
    """Just a regular Chapman layer.  Uses the sec(SZA) approx."""
    def __init__(self, n0=None, z0=None, h=None):
        super(ChapmanLayer, self).__init__()
        self.n0 = n0
        self.z0 = z0
        self.h = h

    def get_tec(self, theta=0.):
        # There are better ways to do this, simply propto h * n0 iirc
        # alt = 10.**(np.arange(-5., 5., 0.1)) - self.z0
        alt = np.hstack((np.arange(0., 500., 1.), np.array((750., 5000., 10000., 100000.))))
        dens = self.__call__(alt, theta)
        plt.plot(alt, dens)
        return trapz(dens, alt)

    def get_params(self):
        return self.n0, self.z0, self.h

    def set_params(self, p):
        self.n0, self.z0, self.h = p

    def __call__(self, alt, theta=0.):
        # The SEC(THETA) approx is only valid for small theta...
        y = (alt - self.z0) / self.h
        return self.n0 * np.exp(0.5 * (1. - y - 1./np.cos(theta) * np.exp(-y)))

    def fit(self, altitude, density, theta):
        """Fit this model to some profile.  Requires values sorted by altitude.
Takes the peak of the layer to be the peak density, rather than fitting it in concert
with the other parameters - see fitx() for this"""

        if np.any(np.diff(altitude) < 0.):
            raise ValueError("fit() requires altitudes sorted increasing, and corresponding densitities")

        i = np.argmax(density)

        self.z0 = altitude[i]
        y_prime = altitude - self.z0

        self.n0 = density[i] / np.exp((1. - 1./np.cos(theta))/2.)

        f = lambda x: self.n0 * np.exp(1.0 - y_prime / x - np.cos(theta)**-1.0 * np.exp(-y_prime / x))
        err = lambda x: density - f(x)

        h, success = opt.leastsq( err, 10.0)
        self.h = h
        self.fit_success = success
        return success

    def fitx(self, altitude, density, theta):
        """Fit this model to some profile.  Requires values sorted by altitude. c.f. fit()"""
        if np.any(np.diff(altitude) < 0.):
            raise ValueError("fit() requires altitudes sorted increasing, and corresponding densitities")

        i = np.argmax(density)

        f = lambda x: x[0] * np.exp(0.5 * (1. - (altitude - x[1]) / x[2] - 1./np.cos(theta) * np.exp(-1. * (altitude - x[1])/x[2])))
        err = lambda x: density - f(x)
        x0 = [density[i], altitude[i], 10.]

        p, success = opt.leastsq( err, x0 )

        self.n0 = p[0]
        self.z0 = p[1]
        self.h  = p[2]
        self.fit_success = success

        return success

    def __str__(self):
        return "ChapmanLayer with peak density %e at altitude %f, scale height %f" % (self.n0, self.z0, self.h)


class Morgan2008ChapmanLayer(ChapmanLayer):
    """Chapman Layer fit described by Morgan et al 2008"""
    def __init__(self):
        super(Morgan2008ChapmanLayer, self).__init__(n0=1.58E5, z0=133.6, h=8.9)

class Fallows15ChapmanLayer(ChapmanLayer):
    """Chapman Layer described by Fallows et al 2015"""
    def __init__(self):
        super(Fallows15ChapmanLayer, self).__init__(n0=1.97E5, z0=130.9, h=5.2)

class Nemec11ChapmanLayer(ChapmanLayer):
    """Chapman Layer described by Nemec et al 2011"""
    def __init__(self):
        super(Nemec11ChapmanLayer, self).__init__(n0=1.59E5, z0=124.7, h=12.16)

class FunctionalModel(IonosphericModel):
    """A model that is defined only by a function that returns density at altitude, sza"""
    def __init__(self, function):
        super(FunctionalModel, self).__init__()
        self._function = function

    def __call__(self, *args, **kwargs):
        return self._function(*args, **kwargs)

    def __str__(self):
        if hasattr(self.__call__, '__doc__'):
            return self.__call__.__doc__
        else:
            return str(self.__call__)

class GaussianModel(IonosphericModel):
    def __init__(self, n0, z0, h):
        super(GaussianModel, self).__init__()
        self.n0 = n0
        self.z0 = z0
        self.h = h

    def get_params(self):
        return self.n0, self.z0, self.h

    def set_params(self, p):
        self.n0, self.z0, self.h = p

    def __call__(self, alt, theta=0.):
        return self.n0 * np.exp(-((alt - self.z0)/self.h)**2.)

    def __str__(self):
        return "Gaussian with peak density %e at altitude %f, scale height %f" % (self.n0, self.z0, self.h)

class TabulatedModel(IonosphericModel):
    """Tabulated values of density versus altitude"""
    def __init__(self, altitude, density, theta=0.):
        super(TabulatedModel, self).__init__()
        self.altitude = altitude
        self.density = density
        self.theta = theta

    def __call__(self, alt, theta=0.):
        if theta != self.theta:
            raise ValueError("Don't know about theta here")
        return np.interp(alt, self.altitude, self.density)

class CompositeModel(IonosphericModel):
    def __init__(self, base=None, local_density=0., local_scale_height=50., bottomside_density=None):
        if base is None:
            self.base = Morgan2008ChapmanLayer()
        else:
            self.base = base
        self.local_density = local_density
        self.local_scale_height = local_scale_height
        self.bottomside_density = bottomside_density

    def __call__(self, alt, theta=0.):
        log_bit = self.local_density * np.exp(-(alt - np.amax(alt)) / self.local_scale_height)
        model_bit = self.base(alt, theta)

        m = np.argmax(model_bit)
        inx = alt > alt[m]

        if self.bottomside_density is not None:
            bottom_bit = np.zeros_like(alt) + self.bottomside_density
        else:
            bottom_bit = np.zeros_like(alt) + np.mean(model_bit[alt < alt[m]])


        bottom_bit[inx] = model_bit[inx]
        inx = (log_bit > model_bit) & (alt > alt[m])
        bottom_bit[inx] = log_bit[inx]
        return bottom_bit

class ChainedModels(IonosphericModel):
    def __init__(self, models=None):
        self._models = []
        if models:
            for m in models:
                if not isinstance(m, IonosphericModel):
                    raise ValueError("WHAT THAR FUCK? Not an IonosphericModel instance!")
                self._models.append(m)

    def add_model(self, m):
        self._models.append(m)

    def __call__(self,  alt, theta=0.):
        results = np.zeros_like(alt)
        for m in self._models:
            results += m(alt, theta)
        return results

    def __str__(self):
        s = 'ChainedModels: \n'
        for m in self._models:
            s += '\t\n%s' % str(m)
        return s

    def get_params(self):
        p = []
        for m in self._models:
            p.extend(m.get_params())
        return tuple(p)

    def set_params(self, pp):
        if isinstance(pp, np.ndarray):
            p = pp.tolist()
        else:
            p = list(pp)

        for m in self._models:
            n = len(m.get_params())
            m.set_params([p.pop(0) for x in range(n) ])

def plot(all_models):

    import matplotlib.pylab as plt
    import numpy.random
    plt.close("all")
    plt.figure()
    plt.subplot(211)
    alt = np.arange(0., 500., 2.)
    sza = 0.

    for m in all_models:
        d = m(alt, sza)
        plt.plot(ne_to_fp(d)/1E6, alt,lw=2)
        # plt.plot(m(alt, sza),alt,lw=2)

    plt.ylim(0., 400.)
    plt.ylabel('Altitude / km')
    # plt.xlabel(r'$n_e / cm^{-3}$')
    plt.xlabel(r'$f / MHz$')
    plt.subplot(212)
    for m in all_models:
        delay, freq = m.ais_response()
        plt.plot(freq/1E6, delay*1E3, lw=2.)

    plt.hlines(-2 * np.amax(alt) / speed_of_light_kms * 1E3, *plt.xlim(), linestyle='dashed')
    # plt.vlines(ne_to_fp(1E5)/1E6, *plt.ylim())
    # plt.hlines(  -(500-150) * 2 / speed_of_light_kms * 1E3, *plt.xlim())
    plt.ylim(-10,0)
    plt.ylabel('Delay / ms')
    plt.xlim(0, 7)
    plt.xlabel('f / MHz')
    plt.show()


def _laminated_delays_test(dd, f, fp_local, altitude=None):
    """Do the inversion - Morgan '08 method
    2012-11-07: all inputs in SI: seconds, Hz and m"""
    const_c = 2.99792458E8 #m/s

    f = np.hstack((fp_local, f))

    nf = f.shape[0]

    range_exp = np.empty_like(f)
    alpha = np.empty_like(f)
    freq_rat = np.empty((nf,nf))
    diff_exp = np.empty((nf,nf))

    l_rat_freq = np.log(f[1:]/f[:-1])
    d = np.hstack((0., dd / 2.))
    app_range = const_c * d
    for i in range(1, nf):
        freq_rat[i,0:i+1] = f[:i+1] / f[i]
    freq_rat = np.arcsin(freq_rat)

    for i in range(1, nf):
        freq_rat[i,i] = np.pi/2.

    cos_f = np.cos(freq_rat)
    exp_w = 0.5 * np.log((1. - cos_f) / (1. + cos_f))

    for i in range(1,nf):
        diff_exp[i,1:i+1] = exp_w[i,1:i+1] - exp_w[i,0:i]

    alpha[1] = -diff_exp[1,1] / app_range[1]
    for i in range(2, nf):
        alpha[i] = -diff_exp[i,i] / (app_range[i] + \
                np.sum(diff_exp[i,1:i]/alpha[1:i]))

    range_exp[0] = 0.
    for i in range(1, nf):
        range_exp[i] = range_exp[i-1] - l_rat_freq[i-1]/alpha[i]

    return altitude - range_exp, fp_to_ne(f)

if __name__ == '__main__':
    plt.close('all')
    from mex.ais import laminated_delays

    all_models = []
    for n in (1e4, 2e4, 4e4, 8e4, 12e4):
        all_models.append(
            ChainedModels( (Morgan2008ChapmanLayer(), GaussianModel(n, 220., 70.))) )

    for n in (1e4, 2e4, 4e4, 8e4, 12e4):
        all_models.append(
            ChainedModels( (Morgan2008ChapmanLayer(), GaussianModel(12e4, 220., 70.), GaussianModel(n/5, 200, 10)) ) )


    # def f(x,t):
    #     fx = np.zeros_like(x)
    #     h = 180.
    #     fx[x > h] = 1e4 * np.exp( -(x[x>h] - h) / 80 )
    #     return fx

    # def f(x,t):
    #     fx = 5e5 * np.exp(-(x-150.)/40.) * (0.5 * np.sin(10. * x) + 0.5)
    #     fx[x < (150. + 10.)] = 0.0
    #     return fx
    #
    # all_models.append(Morgan2008ChapmanLayer())
    # all_models.append(ChainedModels((Morgan2008ChapmanLayer(), FunctionalModel(f))))

    # for h0 in (180, 220, 240, 260):
    #     all_models.append(
    #         ChainedModels( (Morgan2008ChapmanLayer(), GaussianModel(2e4, h0, 70.))) )

    # for h0 in (140, 150, 160, 180):
    #     all_models.append(
    #         ChainedModels( (Morgan2008ChapmanLayer(), GaussianModel(5e4, 220, 70.), GaussianModel(2e4, h0, 30.))) )

    # for n in (1,2,3,4,5,6):
    #     all_models.append(ChapmanLayer(1e5, 150., n * 5.))
    # for dh in (10, 30, 60, 120):
    #     all_models.append(
    #         ChainedModels( (Morgan2008ChapmanLayer(), GaussianModel(5e4, 200, dh))) )
    # plot(all_models)

    model = Morgan2008ChapmanLayer()
    model = ChainedModels( (Morgan2008ChapmanLayer(), ChapmanLayer(1e4,300,100)))


    alt = 800.
    alt_res = 2.

    f = 10.**(np.linspace(np.log10(150e3), np.log10(5.5e6), 160))

    ax = plt.subplot(211)
    h = np.linspace(alt, 0., 160)

    n = model(h)
    plt.plot(n, h, 'kx-')
    plt.ylim(0.,1000.)
    plt.xscale('log')
    t, f = model.ais_response(frequencies=f, sc_altitude=alt, altitude_resolution=alt_res)
    t = -1. * t

    plt.subplot(212)
    plt.plot(f/1e6, t, 'k--')

    # v = np.argmax(n[::-1]) - 1
    # print h.shape, n.shape, t.shape, f.shape

    plt.plot(f/1e6, t, 'k-')
    f0 = ne_to_fp(n[0])

    # inx, = np.where((f > f0) & (f < ne_to_fp(np.max(n))))
    i = np.min(np.where(f > f0)[0])
    df = np.diff(f)
    # print df > 0.
    j = np.min(np.where(df < 0.)[0])
    inx = list(range(i,j+1))
    inx, = np.where((f < ne_to_fp(np.max(n))) & (f > f0))
    print(inx)

    hh2, nn2 = _laminated_delays_test(t[inx], f[inx], f0, altitude=alt*1000.)
    hh, nn = laminated_delays(t[inx], f[inx], f0, altitude=alt*1000.)
    print(hh2[-10:], hh[-10:])
    print()
    # print nn2[:10], nn[:10]

    plt.sca(ax)
    plt.plot(nn2, hh2/1000., 'r+-')
    plt.plot(nn, hh/1000., 'b*-')
    plt.xlim(1., 1e6)
    plt.show()

    # print hh,nn
