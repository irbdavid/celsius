import os
import spiceypy
import numpy as np
import scipy as sp
import mex

try:
    from scipy.io.idl import readsav
except ImportError:
    import idlsave
    readsav = idlsave.read

import celsius
import pickle

import matplotlib.pylab as plt

# import spherical
spherical = None

def schmidt_polynomials(x, nmax):
    """Returns an *nmax*+1 by *nmax*+1 matrix containing Schmidt quasi-normalised polynomials
    evaluated at *x*.  Written with and tested against a bit of code from D. Brain"""
    arr = np.zeros((nmax+1, nmax+1))
    arr[0,0] = 1.

    if nmax > 0:
        twoago = 0.
        for i in range(1, nmax+1):
            arr[i,0] = ( x * (2.*i - 1.) * arr[i-1,0] -  (i - 1.) * twoago ) / i
            twoago = arr[i-1,0]

    cm = 2.**0.5
    for m in range(1, nmax+1):
        cm /= np.sqrt(2. * m * (2.*m - 1.))
        arr[m,m] = (1. - x**2.)**(0.5 * m) * cm

        for i in range(1, m):
            arr[m,m] = (2.*i + 1.) * arr[m,m]

        if nmax > m:
            twoago = 0.
            for i in range(m+1, nmax+1):
                arr[i,m] = (( x * (2.*i - 1.) * arr[i-1,m]
                         - np.sqrt( (i+m-1.) * (i-m-1.) ) * twoago ) /
                         np.sqrt( i**2 - m**2 ))
                twoago = arr[i-1,m]
    return arr

def convert_biau_to_bmso(iau_position_rll, b_iau, times, check_kernels=None):
    """iau_position is in iau_mars frame, (r, lat, lon) in deg, b_iau in (r,theta,phi), times in spice et.  Result is in the MSO frame, cartesian components.

    Will by default assume that the correct spice kernels are loaded (since you have presumably just calculated iau_position_rll).  Otherwise, set check_kernels=True, which defaults to using MEX routines, or the supplied package, e.g. import maven; convert_biau_to_bmso(...,check_kernels=maven).
    """
    b_out = np.empty_like(b_iau) + np.nan

    b_iau_cart = polar_to_cartesian(iau_position_rll, b_iau)

    if check_kernels is not None:
        if check_kernels is True:
            mex.load_kernels(times)
        else:
            check_kernels.load_kernels(times)

    for i in range(times.shape[0]):
        m = spiceypy.pxform("IAU_MARS", "MAVEN_MSO", times[i])
        x = spiceypy.mxv(m,b_iau_cart[:,i]*1.)
        b_out[:,i] = x

    return b_out

class FieldModel(object):
    """docstring for MarsFieldModel"""
    def __init__(self, planet, name, model_type=None):
        super(FieldModel, self).__init__()
        self.planet = planet
        self.name = name
        self.type = model_type
    def __call__(self):
        raise NotImplementedError("Derived class must override")

class SphericalHarmonicModel(FieldModel):
    """Implements a spherical harmonic potential field model"""
    def __init__(self, planet, name, model_type='SphericalHarmonic', g=None, h=None, nmax=None, rp=None):
        super(SphericalHarmonicModel, self).__init__(planet, name, model_type)
        # print "SphericalHarmonicModel: Still not sure about the absolute values calculated..."
        self.g = g
        self.h = h
        self.rp = rp
        if nmax is None:
            if self.g is not None:
                nmax = self.g.shape[0] - 1
        self.nmax = nmax

    def pure_python_call(self, pos, nmax=None):
        """Evaluate this model at position *pos* where pos is a 3xN array containing
        position coordinates in r (km), lat (deg), lon (deg)"""
        if nmax is None:
            nmax = self.g.shape[0] - 1

        if isinstance(pos, np.ndarray):
            result = np.empty_like(pos)
            if len(pos.shape) > 1:
                for i in range(pos.shape[1]):
                    result[:,i] = self._eval(pos[:,i], nmax)
            else:
                result = self._eval(pos, nmax)
        else:
            result = self._eval(pos, nmax)

        return result

    def _eval(self, p, nmax):
        """*p* is position in [r, lat, lon], where r is in km, and lat, lon are in deg"""
        # This is the bit that needs to be written in C
        # This routine is shamelessly stolen from Dave Brain, and transposed to python.  Thanks Dave!, Cheers, /OtherDave.
        a_over_r = self.rp / p[0]
        l        = np.deg2rad(p[2])
        sct      = np.deg2rad(90. - p[1])

        # print 'pos: %f, %f, %f' % (a_over_r, l, sct)

        g = self.g
        h = self.h

        cntarr = np.arange(nmax + 1.)

        # ; Compute R(r) and dR(r) at each n
        # ;  Only compute parts inside the summation over n
        # ;  R(r) = [a/r]^(n+1)
        # ;  dR(r) = (n+1)*[a/r]^(n+1)  ( Factors omitted that can move
        # ;                               outside of summation - see
        # ;                               pg 34 in Thesis Book 2 )
        # R = (a_over_r)^(cntarr+1)
        # dR = R*(cntarr+1)
        r = a_over_r**(cntarr + 1.)
        dr = r * (cntarr + 1.)

        # ; Compute Phi(phi) and dPhi(phi) at each m,n combo
        # ;  Phi(phi) = gnm * cos(m*phi) + hnm * sin(m*phi)
        # ;  dPhi(phi) = m * [-gnm * sin(m*phi) + hnm * cos(m*phi)]
        # cos_m_phi = cos( cntarr * scp )
        # sin_m_phi = sin( cntarr * scp )
        # Phi  = g*0d
        # dPhi = Phi
        # FOR n = 1, nmax DO BEGIN
        #  Phi[n,*]  = cos_m_phi * g[n,*] + sin_m_phi * h[n,*]
        #  dPhi[n,*] = ( cos_m_phi * h[n,*] - sin_m_phi * g[n,*] ) * cntarr
        # ENDFOR; n = 1, nmax
        cos_m_phi = np.cos(cntarr * l)
        sin_m_phi = np.sin(cntarr * l)
        phi  = np.zeros_like(g)
        dphi = np.zeros_like(phi)
        for n in range(1, nmax + 1):
            phi[n,:] = cos_m_phi * g[n,:] + sin_m_phi * h[n,:]
            dphi[n,:] = (cos_m_phi * h[n,:] - sin_m_phi * g[n,:]) * cntarr

        # ; Compute Theta and dTheta at each m,n combo
        # ;  Theta(theta) = P(n,m,x)  the Schmidt normalized associated legendre poly.
        # ;  dTheta(theta) = m * cos(theta) / sin(theta) * P(n,m,x) -
        # ;                  C(n,m) / C(n,m+1) * P(n,m+1,x)
        # ;                  Where C(n,m) = 1 if m=0
        # ;                               = ( 2 * (n-m)! / (n+m)! ) ^ (1/2)
        # ;                  Cool math tricks are involved
        # cos_theta = cos(sct)
        # sin_theta = sin(sct)
        cos_theta = np.cos(sct)
        sin_theta = np.sin(sct)

        # Theta = legendre_schmidt_all(nmax,cos_theta)
        # reftime1 = systime(1)
        # dTheta = g*0d
        # dTheta[1,*] = cntarr * cos_theta / sin_theta * Theta[1,*]
        # dTheta[1,0] = dTheta[1,0] - Theta[1,1]
        #
        # FOR n = 2, nmax DO BEGIN
        #  dTheta[n,*] = cntarr * cos_theta / sin_theta * Theta[n,*]
        #  dTheta[n,0] = dTheta[n,0] - $
        #                sqrt( (n * (n+1)) * 0.5d ) * Theta[n,1]
        #  dTheta[n,1:n] = dTheta[n,1:n] - $
        #                  sqrt( (n-cntarr[1:n]) * (n+cntarr[1:n]+1) ) * $
        #                   [ [ Theta[n,2:n] ], [ 0d ] ]
        # ENDFOR; n = 1, nmax
        theta = schmidt_polynomials(cos_theta, nmax)
        dtheta = np.zeros_like(g)
        dtheta[1,:] = cntarr * cos_theta / sin_theta * theta[1,:]
        dtheta[1,0] = dtheta[1,0] - theta[1,1]
        for n in range(2, nmax + 1):
            dtheta[n,:] = cntarr * cos_theta/sin_theta * theta[n,:]
            dtheta[n, 0] = dtheta[n, 0] - np.sqrt(n * (n + 1.) * 0.5) * theta[n, 1]
            # careful - slices behave differently in np
            dtheta[n, 1:n+1] = dtheta[n, 1:n+1] - (np.sqrt( (n - cntarr[1:n+1]) *
                                                            (n + cntarr[1:n+1] + 1) )
                                                    * np.hstack((theta[n,2:n+1], [0.])))

        # ; Put it all together
        #
        # ; Br = a/r Sum(n=1,nmax) { (n+1) * R(r) *
        # ;      Sum(m=0,n) { Theta(theta) * Phi(phi) } }
        # br = total( Theta*Phi, 2 )      ; Sum over m for each n
        # br = total( br * dR ) * a_over_r ; (0th element contributes 0)

        # def namestr(obj, namespace):
        #     return [name for name in namespace if namespace[name] is obj]
        # print '......'
        # for v in [theta, phi, dr, a_over_r, dtheta, r]:
        #     if isinstance(v, np.ndarray):
        #         print " ".join(namestr(v, globals())) + ' : SHAPE ' + str(v.shape)
        #     else:
        #         print " ".join(namestr(v, globals())) + ' : ' + str(v)

        # tmp = theta * phi
        # print 'br precusor shape'
        # print tmp.shape
        # print np.sum( theta * phi, 1).shape
        # celsius.code_interact(locals())

        br = np.sum( theta * phi, 1)
        br = np.sum(br * dr) * a_over_r

        # ; Btheta = B_SN
        # ; Btheta = a*sin(theta)/r Sum(n=1,nmax) { R(r) *
        # ;          Sum(m=0,n) { dTheta(theta) * Phi(phi) } }
        # bt = total( dTheta*Phi, 2 )      ; Sum over m for each n
        # bt = -1.d * total( bt * R ) * a_over_r ; (0th element contributes 0)
        bt =       np.sum(dtheta * phi, 1)
        bt = -1. * np.sum(bt * r) * a_over_r

        # ; Bphi = B_EW
        # ; Bphi = -a/r/sin(theta) Sum(n=1,nmax) { R(r) *
        # ;        Sum(m=0,n) { Theta(theta) * DPhi(phi) } }
        # bp = total( Theta*dPhi, 2 )      ; Sum over m for each n
        # bp = -1.d * total( bp * R ) * a_over_r / sin_theta ; ( 0th element
        #                                                  ;   contributes 0 )
        bp = np.sum(theta * dphi, 1)
        bp = -1. * np.sum(bp * r) * a_over_r / sin_theta
        # ; Return the vector field
        # return, [br, bt, bp]

        return np.array((br, bt, bp))

    def fast_eval(self, p, nmax=None):
        raise NotImplementedError()
        if not isinstance(p, np.ndarray):
            p = np.array(p)
        if len(p.shape) == 1:
            p = p[:, np.newaxis]

    def __call__(self, pos, nmax=None, python=True):
        if python:
            return self.pure_python_call(pos, nmax=nmax)

        if nmax is None:
            if self.nmax is None:
                self.nmax = self.g.shape[0] - 1
            nmax = self.nmax
        return spherical.compute(pos, self.g, self.h, nmax, self.rp)


class ArkaniMarsFieldModel(SphericalHarmonicModel):
    """docstring for ArkaniMarsFieldModel"""
    def __init__(self, coefficients_file=None, nmax=62):
        super(ArkaniMarsFieldModel, self).__init__("MARS", "Arkani (2004)")
        if coefficients_file is None:
            coefficients_file = mex.data_directory + "brain/martiancrustmodels.sav"

        self.coefficients_file = coefficients_file
        self.nmax = nmax

        tmp = readsav(self.coefficients_file, verbose=False)

        if (self.nmax + 2) > tmp.ga.shape[0]:
            raise ValueError("nmax = %d greater than supplied g,h shape (%d,%d)" % (nmax, tmp.ga.shape[0], tmp.ga.shape[0]))

        # Transpose to change the ordering
        self.g = tmp.ga[0:self.nmax+2,0:self.nmax+2].T.copy()
        self.h = tmp.ha[0:self.nmax+2,0:self.nmax+2].T.copy()
        self.rp = tmp.rplaneta

class CainMarsFieldModel(SphericalHarmonicModel):
    """docstring for CainMarsFieldModel"""
    def __init__(self, coefficients_file=None, nmax=60):
        super(CainMarsFieldModel, self).__init__("MARS", "Cain (2003)")
        if coefficients_file is None:
            # coefficients_file = mex.locate_data_directory() + 'cain03_coefficients.dat'
            coefficients_file = mex.data_directory + "brain/martiancrustmodels.sav"

        self.coefficients_file = coefficients_file
        self.nmax = nmax

        tmp = readsav(self.coefficients_file, verbose=False)

        # Transpose to change the ordering
        self.g = tmp.gc[0:self.nmax+2,0:self.nmax+2].T.copy()
        self.h = tmp.hc[0:self.nmax+2,0:self.nmax+2].T.copy()
        self.rp = tmp.rplanetc * 1.

        # print self.rp is tmp.rplanetc

class LillisMarsFieldModel(SphericalHarmonicModel):
    """Lillis 2010 Mars field model (c.f. Purucker 2008)"""
    def __init__(self, coefficients_file=None, nmax=51):
        super(LillisMarsFieldModel, self).__init__("MARS", "Lillis (2010)")
        if coefficients_file is None:
            coefficients_file = mex.data_directory + "lillis_coefs.txt"

        self.coefficients_file = coefficients_file
        self.nmax = nmax

        self.g = np.zeros((52, 52))
        self.h = np.zeros_like(self.g)

        self.rp = 3393.5
        c = 0
        with open(coefficients_file) as f:
            for i in range(9): f.readline()

            for l in f:
                c += 1
                n, m, gnm, hnm = l.split()
                n = int(n)
                m = int(m)
                self.g[n,m] = float(gnm)
                self.h[n,m] = float(hnm)

        print("Read %d lines of coefficients from %s" % (c,
                            coefficients_file))

class MorschhauserMarsFieldModel(SphericalHarmonicModel):
    """Morschhauser 2014 Mars field model"""
    def __init__(self, coefficients_file=None, nmax=110):
        super(MorschhauserMarsFieldModel, self).__init__("MARS", "Morschhauser (2014)")

        # raise RuntimeError()
        if coefficients_file is None:
            coefficients_file = mex.data_directory + "morschhauser_coeffs.txt"

        self.coefficients_file = coefficients_file
        self.nmax = nmax

        self.g = np.zeros((nmax+1, nmax+1))
        self.h = np.zeros_like(self.g)

        self.rp = 3393.5
        c = 0
        with open(coefficients_file) as f:
            for i in range(3): f.readline()

            for l in f:
                c += 1
                n, m, gnm = l.split()
                n = int(n)
                m = int(m)

                if n > nmax:
                    break
                if m >= 0:
                    self.g[n,m] = float(gnm)
                if m < 0:
                    self.h[n,-m] = float(gnm)

        print("Read %d lines of coefficients from %s" % (c,
                            coefficients_file))


class CainMarsFieldModelAtMEX(object):
    """docstring for CainMarsFieldModelAtMEX"""
    def __init__(self, directory=None):
        super(CainMarsFieldModelAtMEX, self).__init__()
        if directory is None:
            directory = mex.data_directory + 'cain_model_lookup/'
        self.directory = directory
        self.field_model = CainMarsFieldModel()

    def __call__(self, time, force=False):
        t = np.atleast_1d(time)
        t0 = mex.orbits[np.amin(t)].number
        t1 = mex.orbits[np.amax(t)].number + 1
        if t1 > (t0 + 2):
            if force and (((t1 - t0) / time.shape[0]) < 50):
                raise ValueError("%d orbits, but not many points requested?  force=True disables this check")

        chunks = []
        compute_count = 0
        for o in range(t0, t1):
            try:
                chunk = np.load(self.directory + '%05d/%05d.npy' % (o/1000 * 1000, o))
            except IOError as e:
                compute_count+= 1
                print('Computing for orbit %d' % o)
                o_t = np.arange(-40. * 60., 40. * 60., 5.) + mex.orbits[o].periapsis
                pos = mex.iau_r_lat_lon_position(o_t)
                field = self.field_model(pos)
                # for i in range(o_t.shape[0]):
                #     print o_t[i], field[i,:]
                chunk = np.vstack((o_t, field))
                all_dirs = self.directory + '%05d/' % (o/1000 * 1000)
                if not os.path.exists(all_dirs):
                    os.makedirs(all_dirs)
                np.save(all_dirs + '%05d.npy' % o, chunk)
            chunks.append(chunk)
        data = np.hstack(chunks)

        output = np.empty((3,t.shape[0]))
        for i in (0,1,2):
            output[i,:] = np.interp(t, data[0,:], data[i+1,:], left=np.nan, right=np.nan)

        return output


class SaturnFieldModel(SphericalHarmonicModel):
    """Saturnian field models, mainly for comparison purposes"""
    def __init__(self, name="CASSINI_SOI"):
        super(SaturnFieldModel, self).__init__("SATURN", name.upper())

        self.g = np.zeros((4,4))
        self.h = np.zeros((4,4))
        self.rp = 60268.0 # Saturn mean radius, km

        if self.name == 'CASSINI_SOI':
            self.g[:,0] =  0, 21084, 1544, 2150
        elif self.name == 'BURTON_09':
            self.g[:,0] =  0, 21145, 1546, 2241
        else:
            raise ValueError("Model name '%s' not recognized" % self.name)

class RandomFieldModel(SphericalHarmonicModel):
    """Random field models, mainly for comparison purposes"""
    def __init__(self, name="RANDOM", nmax=1):
        super(RandomFieldModel, self).__init__("RANDOM", name.upper())

        self.nmax = nmax
        n1 = self.nmax + 1
        self.g = np.random.randn(n1 * n1).reshape((n1,n1))
        self.h = np.zeros((n1,n1))
        self.rp = 1000.0


def create_snapshot(model, filename=None, resolution=1., altitude=150.):
    latitude = np.arange(90, -90, resolution * -1.)
    longitude = np.arange(0, 360, resolution)
    lat_mesh, lon_mesh = np.meshgrid(latitude, longitude)
    r = np.zeros_like(lat_mesh.flatten()) +  \
        mex.mars_mean_radius_km + altitude
    print('Calculating %d points...' % r.shape[0])
    field = model(np.vstack((r, lat_mesh.flatten(), lon_mesh.flatten())))

    obj = dict(latitude=latitude, longitude=longitude,
                latitude_mesh=lat_mesh, longitude_mesh=lon_mesh,
                altitude=altitude, model=model.name,
                field=field)

    if filename:
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)
        print('... done, wrote to %s' % filename)

    return obj

def plot_lat_lon_field(name=None, value='|B|', zorder=0, cmap=None,
            vmin=None, vmax=None,
            colorbar=True, cax=None, ax=None, labels=True, lims=True,
            return_data=False, label_str=None, full_range=False,
            inclination_min_b=None,
            inclination_draped_imf=0., inclination_draped_imf_orient=0.,
            snapshot_kwargs={}, data_only=False, **kwargs):
    """Plot `value` in Latitude / East Longitude.
    snapshot_kwargs go to create_snapshot if it's called.
    extra args go to imshow."""

    if name is None:
        name = mex.data_directory + 'saved_field_models/Cain_400km1deg.pck'
        if not os.path.exists(name):
            name = mex.data_directory + 'saved_field_models/CainMarsFieldModel_1deg_400km.pck'

    if isinstance(name, str):
        print('Loading field model from %s' % name)
        with open(name, 'rb') as f:
            field_data = pickle.load(f)
    else:
        field_data = create_snapshot(name, **snapshot_kwargs)

    if ax is None:
        ax = plt.gca()
    else:
        plt.sca(ax)

    q_abs = False
    units = 'nT'
    _vmin, _vmax = 0., 50.

    if field_data['altitude'] < 200.:
        _vmax = 500.

    # TeX labels:
    label_dict = {'|B|':r'|\mathbf{B}_c|',
            'Br':r'B_r',
            'Bt':r'B_\theta',
            'Bp':r'B_\varphi',
            'Bh':r'B_H',
            'AbsInclination':r'|\delta|',
            'Inclination':r'\delta'}

    if label_str is None:
        label_str = label_dict[value]

    l_mesh_shp = field_data['longitude_mesh'].shape
    fd         = field_data['field']

    if (value == '|B|'):
        q = np.sqrt((fd[0,:].reshape(l_mesh_shp).T)**2.
                + (fd[1,:].reshape(l_mesh_shp).T)**2.
                + (fd[2,:].reshape(l_mesh_shp).T)**2.)
        q_abs = True

    elif value == 'Br':
        q = fd[0,:].reshape(l_mesh_shp).T

    elif value == 'Bt':
        q = fd[1,:].reshape(l_mesh_shp).T

    elif value == 'Bp':
        q = fd[2,:].reshape(l_mesh_shp).T

    elif value == 'Bh':
        q = np.sqrt((fd[1,:].reshape(l_mesh_shp).T)**2.
        + (fd[2,:].reshape(l_mesh_shp).T)**2.)
        q_abs = True

    elif value == 'Inclination':
        # atan ( br, b_horizontal )

        fd[1,:] += inclination_draped_imf * np.cos(np.deg2rad(inclination_draped_imf_orient))
        fd[2,:] += inclination_draped_imf * np.sin(np.deg2rad(inclination_draped_imf_orient))

        q = np.rad2deg(np.arctan2(fd[0,:].reshape(l_mesh_shp).T,
                np.sqrt((fd[1,:].reshape(l_mesh_shp).T)**2.
                        + (fd[2,:].reshape(l_mesh_shp).T)**2.)))
        _vmax = 90
        units = 'deg.'

    elif value == 'AbsInclination':
        # atan ( br, b_horizontal )

        fd[1,:] += inclination_draped_imf * np.cos(np.deg2rad(inclination_draped_imf_orient))
        fd[2,:] += inclination_draped_imf * np.sin(np.deg2rad(inclination_draped_imf_orient))

        q = np.rad2deg(np.arctan2(fd[0,:].reshape(l_mesh_shp).T,
                np.sqrt((fd[1,:].reshape(l_mesh_shp).T)**2.
                        + (fd[2,:].reshape(l_mesh_shp).T)**2.)))
        q = np.abs(q)
        _vmin = 0
        _vmax = 90
        q_abs = True
        units = 'deg.'
    else:
        raise ValueError('value "%s" not recognized' % str(value))

    if ('Inclination' in value) and (inclination_min_b is not None):
        b = np.sqrt((fd[0,:].reshape(l_mesh_shp).T)**2.
                + (fd[1,:].reshape(l_mesh_shp).T)**2.
                + (fd[2,:].reshape(l_mesh_shp).T)**2.)
        q[b < inclination_min_b] = np.nan

    if not q_abs:
        _vmin = - _vmax

    if not vmin:
        vmin = _vmin

    if not vmax:
        vmax = _vmax

    extent = (0., 360., -90, 90)

    if not cmap:
        cmap = plt.cm.RdBu_r
        if q_abs:
            cmap = plt.cm.Reds

    if data_only:
        if full_range:
            q2 = np.hstack((q, q, q))
            return q2
        return q

    if full_range:
        q2 = np.hstack((q, q, q))
        extent2 = (-360, 720., -90, 90)
        out = plt.imshow(q2, vmin=vmin, vmax=vmax, cmap=cmap, extent=extent2, **kwargs)
    else:
        out = plt.imshow(q, vmin=vmin, vmax=vmax, cmap=cmap, extent=extent, **kwargs)

    if labels:
        plt.xlabel('Longitude / deg')
        plt.ylabel('Latitude / deg')

    if lims:
        plt.xlim(0., 360.)
        plt.ylim(-90, 90)
        ax.yaxis.set_major_locator(celsius.CircularLocator())
        ax.xaxis.set_major_locator(celsius.CircularLocator())

    if colorbar:
        if not cax:
            cax = celsius.make_colorbar_cax()
        if value == 'Inclination':
            c = plt.colorbar(cax=cax, ticks=[-90, -45, 0, 45, 90])
        elif value == 'AbsInclination':
            c = plt.colorbar(cax=cax, ticks=[0, 45, 90])
        else:
            c = plt.colorbar(cax=cax)
        c.set_label(r'$%s / %s$' % (label_str, units))
        plt.sca(ax)

    if return_data:
        return q

    # if celsius.now() > celsius.spiceet("2012 JULY 11"):
    #     raise NotImplementedError("Unfinished / unchecked 2012-07-08")

if __name__ == '__main__':
    # So the conclusion seems to be that the above code reproduces the field geometry well,
    # except insofar as the bigger fields seem a bit damped compared to the duru paper
    # These are due to the higher order terms, which only dominate close to in, so maybe
    # its a precision thing, but then all the above is done in 64bit, so I doubt that the
    # error is here if that's the issue.
    # Doesn't also seem to be particularly sensitive to planetographic coordinates or not
    # (i.e. whether we account for the oblate spheriod, unless again this isn't being done
    # correctly)
    # Same effect present in comparison with Franz's data

    import matplotlib.pylab as plt
    import celsius
    import pickle
    import mex
    import os

    if True:
        model = CainMarsFieldModel()
        name  = 'CainMarsFieldModel_1deg_%dkm.pck'
        for alt in (400., 0., 100., 150., 1000.):
            if alt < 10.: alt = 10.
            fname = name % alt
            fname = os.path.expanduser(
                    '~/data/mex/saved_field_models/') + fname
            create_snapshot(model, fname, resolution=1., altitude=alt)

    if False:
        comparison = 'None'
        a = CainMarsFieldModel()
        print(a.g[:5, :5])
        print(a.h[:5, :5])
        plt.close('all')


        # a = ArkaniMarsFieldModel()
        print(a.rp)
        print(mex.mars_mean_radius_km)
        # r = np.zeros(100) + 1.001 * mex.mars_mean_radius_km
        # lon = np.linspace(0., 360., r.shape[0])
        # lat = np.zeros_like(r) - 60.
        # field = a(np.vstack((r, 90.0 - lat, lon)))
        fig = plt.figure(figsize=(8,6))

        orb = mex.orbits[10470]
        et = np.linspace(orb.periapsis-3600., orb.periapsis+3600., 1000.)



        # Duru 06a
        if comparison == 'Duru':
            et = np.linspace(
                celsius.spiceet("2005-224T04:45:00"), celsius.spiceet("2005-224T05:15:00"), 100)

        # Fraenz 10a
        if comparison == 'Fraenz':
            orb = mex.orbits[5009]
            et = np.linspace(-7 * 60., 7 * 60., 100) + orb.periapsis

        pos = mex.iau_r_lat_lon_position(et)
        # pos[0,:] += mex.mars_mean_radius_km
        # pos = np.empty_like(p)
        # pos[2,:] = -np.rad2deg(p[1,:])
        # pos[1,:] = np.rad2deg(p[2,:])
        # pos[0,:] = p[0,:]

        if comparison == 'Duru':
            pos[0,:] = np.zeros_like(pos[0,:]) + mex.mars_mean_radius_km + 150.

        field = a(pos)
        print(field)

        plt.subplot(311)
        plt.plot(et, pos[0,:] - mex.mars_mean_radius_km)
        # plt.plot(et, p[0,:] - mex.mars_mean_radius_km)

        plt.subplot(312)
        plt.plot(et, pos[1,:])
        plt.plot(et, pos[2,:])

        plt.subplot(313)
        plt.plot(et, field[0,:], 'r-')
        plt.plot(et, field[1,:], 'g-')
        plt.plot(et, field[2,:], 'b-')
        plt.plot(et, np.sqrt(np.sum(field * field, axis=0)), 'k-')
        print(celsius.utcstr(et[0]))
        print(celsius.utcstr(et[-1]))
        plt.gca().xaxis.set_major_locator(celsius.SpiceetLocator())
        plt.gca().xaxis.set_major_formatter(celsius.SpiceetFormatter())

        if comparison == 'Duru':
            plt.ylim(-400,400)

        plt.show()

    if False:
        plt.figure(figsize=(8,6))
        lt = np.arange(90, -90, -1.)
        ln = np.arange(0, 360, 1.)
        lat, lon = np.meshgrid(lt, ln)
        r = np.zeros_like(lat.flatten()) +  mex.mars_mean_radius_km + 150.

        print(lat.shape)
        print(lon.shape)
        field = a(np.vstack((r, lat.flatten(), lon.flatten())))

        # plt.imshow(field[0,:].reshape(lat.shape).T, extent=)

        abs_field = np.sqrt(field[0,:]**2. + field[1,:]**2. + field[2,:]**2.)
        inx = abs_field < 100.
        theta = np.rad2deg(np.arctan2(field[0,:], np.sqrt(field[1,:]**2. + field[2,:]**2.)))
        theta = np.abs(theta)
        theta[inx] = -99.0

        # plt.hot()
        plt.get_cmap().set_under('grey')
        plt.imshow(theta.reshape(lat.shape).T, vmin=0., vmax=90., extent=(0., 360., -90., 90))
        plt.xlabel('Longitude / deg')
        plt.ylabel('Latitude / deg')
        plt.colorbar()
        plt.ylabel(r'$|\theta_B| / deg$')
        plt.show()
        plt.savefig("Crustal_ThetaB.pdf")

    if False:
        for i in range(1, 100):
            print('\n')
            print(i)
            b = RandomFieldModel(nmax=50)
            # print i
            # print b.g[:5,:5]
            # print b.rp

            b((60268. * 12, 68., 238.)).copy()
            # print b.pure_python_call((60268. * 12, 68., 238.))
