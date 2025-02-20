# Codes based on LDC.
import numpy as np
import lisaconstants

from scipy.interpolate import InterpolatedUnivariateSpline as spline

try:
    import cupy as cp
except ImportError:
    cp = None


clight = lisaconstants.SPEED_OF_LIGHT


def AziPol2Psi(theS, phiS, theK, phiK):
    """
    Compute polarisation angle to make a conversion from source frame to SSB
        Args:
        theS (rad): polar angle of sky location in SSB 
        phiS (rad): azimuthal angle of sky location in SSB 
        theK (rad): azimuthal angle describing the orientation of the spin-angular momentum of the MBH
        phiK (rad): polar angle describing the orientation of the spin-angular momentum of the MBH
    """
    #inc = np.arccos( - np.cos(theL)*np.sin(bet) - np.cos(bet)*np.sin(theL)*np.cos(lam - phiL) )
    down_psi = np.sin(theK)*np.sin(phiS - phiK)
    up_psi = np.cos(theS)*np.sin(theK)*np.cos(phiS - phiK) - np.sin(theS)*np.cos(theK)
    psi = np.arctan2(up_psi, down_psi)

    return psi


def source2SSB(hSp, hSc, psi):
    """ Convert h+, hx from source frame to Solar System Barycenter.
            Convention defined in the LDC documentation.
    """
    cos2Pol = np.cos(2.*psi) 
    sin2Pol = np.sin(2.*psi)

    hp = hSp * cos2Pol - hSc * sin2Pol
    hc = hSp * sin2Pol + hSc * cos2Pol
    return hp, hc


def define_basis(theS, phiS, psi):
    """ Compute the basis to project the waveform
    """
    #p = self.source_parameters
        #self.pol = p['Polarization']
        #self.cos2Pol = np.cos(2.*self.pol)
        #self.sin2Pol = np.sin(2.*self.pol)
    #self.eclLat = p['EclipticLatitude']
    #self.eclLon = p['EclipticLongitude']

    ## Projection basis
    sin_theS, cos_theS = np.sin(theS), np.cos(theS) 
    sin_phiS, cos_phiS = np.sin(phiS), np.cos(phiS)

    # Sky-position vector along the line-of-sight
    k = np.array([sin_theS * cos_phiS, sin_theS * sin_phiS, cos_theS]) 

    # Vectors of the spherical basis
    p = np.array([cos_theS * cos_phiS, cos_theS * sin_phiS, -sin_theS]) 
    q = np.array([-sin_phiS, cos_phiS, 0.0])
    
    if isinstance(sin_theS, np.ndarray):
        q = np.array([-sin_phiS, cos_phiS, np.zeros((len(sin_theS)))])
    else:
        q = np.array([-sin_phiS, cos_phiS, 0.])

    # Make a rotation and convert (p,q) to (u,v)
    u = np.cos(psi)*p - np.sin(psi)*q
    v = np.sin(psi)*p + np.cos(psi)*q

    return k, v, u


def interp_cuda_dev(x_vals, x, y):
    """
    Interpolate [x, y] and evaluate in x_vals
    Input and output are cupy arrays  
    """
    return cp.interp(x_vals, x, y)


def _interp(t, x, tnew, kind='spline', der=False, integr=False):
    """ Perform the interpolation of hx, hp at time samples in t. """
    if cp is not None and cp.get_array_module(tnew) is cp:
        # if tnew is in device, use cuda interpolation
        # t, x are in host, tnew is already in device (see method interp_hphc)
        return interp_cuda_dev(tnew, cp.array(t), cp.array(x))
    if kind == 'spline':
        if der:
            splh = spline(t, x).derivative()
        elif integr:
            splh = spline(t, x).antiderivative()
        else:
            splh = spline(t, x)
        return (splh(tnew), splh)
    #hp = np.interp(tnew, t, x)
    itp = interp1d(t, x, kind=kind)
    return itp(tnew), itp



class ProjectedStrain():
    """ Project GW strain on LISA arms """

    def __init__(self, orbits, theS, phiS, theK, phiK, psi, hp, hc, t):
        """  Set orbits and constellation configuration.
        """
        self.orbits = orbits
        self.set_link()
        self.theS = theS
        self.phiS = phiS 
        self.theK = theK
        self.phiK = phiK
        self.psi = psi

        self.hp = hp
        self.hc = hc
        self.t  = t

    def init_links(self, receiver_time, order=0, cuda=False):
        """ Compute and save sc positions and travel time.
        """

        if cuda is True and cp is None:
            print("cupy is not installed, can't use cuda option")
            cuda = False
        #alphas = self.orbits.compute_alpha(receiver_time)

        # Choose between numpy and cupy
        xp = cp if cuda is True else np

        # pos and tt will be saved as numpy/cupy arrays
        self.pos = xp.zeros((self.orbits.number_of_spacecraft, 3, len(receiver_time)))
        for i in range(1,self.orbits.number_of_spacecraft+1):
            self.pos[i-1,:,:] = xp.array(self.orbits.compute_position(i,
                                                                      receiver_time)/clight)

        self.tt = np.zeros((len(receiver_time), self.orbits.number_of_arms))
        for i in range(self.orbits.number_of_arms):
            receiver, emitter = self.orbits.get_pairs()[i]
            pe = self.pos[emitter-1,:]*clight
            pr = self.pos[receiver-1,:]*clight
            self.tt[:,i] = self.orbits.compute_travel_time(emitter, receiver,
                                                           receiver_time, order=order,
                                                           position_emitter=pe,
                                                           position_receiver=pr)


    def arm_response(self, t_min, t_max, dt, tt_order=0, ilink=None):
        """Return projected strain y_t

        For source in interp_type, hphc is computed only once, then
        stored and used for interpolation. Beware of the memory
        consumption of such sources.

        Args:
            t_min, t_max, dt: scalar defining time range
            GWs: list of HpHc object
            ilink: specify the link number (0 to 5) on which to project the signal.
                   If None, all links are computed.
        """
        receiver_time = np.arange(t_min, t_max, dt)
        #extended_time = np.arange(max(t_min-dt*(extended_t//dt), 0),
        #                              t_max+dt*(extended_t//dt), dt)

        ## GW hp,hc precomputation
        #self.source_names = [GW.source_name for GW in GWs]
        #jk = [GW.compute_hphc(receiver_time, **kwargs) #extended_time, approx_t=True)
        #      for GW in GWs if GW.source_type in interp_type]

       # if GWs[0].source_type in interp_type:
       #     hphc_call = 'interp_hphc'
       #     if ('precomputed' in kwargs and kwargs['precomputed']) or GWs[0].source_type == 'Numeric':
       #         pass
       #     else:
       #         jk = [GW.compute_hphc_td(receiver_time, **kwargs) for GW in GWs]
       # else:
       #     hphc_call = 'compute_hphc_td'
       #hphc_call = 'interp_hphc'

        ### Compute GW effect on links
        self.t_min = t_min
        self.t_max = t_max
        self.dt = dt
        if 'tt' not in self.__dict__ or 'pos' not in self.__dict__:
            self.init_links(receiver_time, order=tt_order)

        nArms = self.orbits.number_of_arms if ilink is None else 1
        links = range(nArms) if ilink is None else [ilink]
        self.yArm = np.zeros([len(receiver_time), nArms])
        for link in links:
            self.yArm[:,link] += self._arm_response(receiver_time, link)
        return self.yArm


    def _arm_response(self, receiver_time, link):
        """Compute GW signal at receiver time for a single source and link.

        call_hphc can be switched to interpolation instead of actual
        hphc computation.

        """

        # Arm geometry
        receiver, emitter = self.orbits.get_pairs()[link]
        rem, rrec = self.pos[emitter-1,:,:], self.pos[receiver-1,:,:]
        # Choose between numpy or cupy
        xp = cp.get_array_module(rem) if cp is not None else np

        receiver_time = xp.array(receiver_time)
        tem = receiver_time - self.tt[:,link]
        r_ij = rrec - rem
        n = r_ij/xp.linalg.norm(r_ij, axis=0) # unit vector between receiver and emitter

        #psi = AziPol2Psi(self.theS, self.phiS, self.theK, self.phiK)
        k,v,u = define_basis(self.theS, self.phiS, self.psi)
        k = xp.array(k)
        v = xp.array(v)
        u = xp.array(u)

        un, vn, kn = xp.dot(u, n), xp.dot(v, n), xp.dot(k, n)
        xi_p = 0.5*(un*un - vn*vn)
        xi_c = un*vn

        # Dot products of GW wave propagation vector and emitter, receiver postiional vectors
        kep = xp.dot(k, rem)
        krp = xp.dot(k, rrec)

        te = tem - kep
        tr = receiver_time - krp

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Not sure about this part
        hpe, hce = self.interp_hphc(te)
        hpr, hcr = self.interp_hphc(tr)

        # Projected strain
        y = ((hpe - hpr)*xi_p + (hce - hcr)*xi_c )
        y/= (1.0-kn)

        # Move array from device(cupy) to host(numpy)
        if xp == cp:
            y = xp.asnumpy(y)

        return y


    def set_link(self):
        """ Define link names and indices.
        """
        self.links = ["%d-%d"%(a,b) for a,b in self.orbits.get_pairs()]
        self.dlink = dict(zip(self.links, range(len(self.links))))

    def interp_hphc(self, t, cuda=False):
        """ Interpolate hp,hx on a given time range.

        A first call to compute_hphc_td is assumed here.
        """
        if cuda is True:
            if cp.get_array_module(t) is np:
                # Move array to device if it is not
                cu_t = cp.array(t)
            else:
                cu_t = t
            hp = _interp(self.t, self.hp, cu_t)
            hc = _interp(self.t, self.hc, cu_t) 
            return hp, hc
        if "i_hp" in self.__dict__:
            return self.i_hp(t), self.i_hc(t)
        hp, self.i_hp = _interp(self.t, self.hp, t)
        hc, self.i_hc = _interp(self.t, self.hc, t)
        return hp, hc

    def _interp_hphc(self, tm, hp, hc, t):
        """ Interpolate hp, hc and set them as attr.

        Extrapolation is a 0 padding.
        """
        t_start, t_end = max(tm[0], t[0]), min(tm[-1], t[-1])
        i_st = np.argwhere(t >= t_start)[0][0]
        i_en = np.argwhere(t >= t_end)[0][0]
        self.hp, self.hc = np.zeros(len(t)), np.zeros(len(t))
        self.hp[i_st:i_en+1], self.i_hp = _interp(tm, hp, t[i_st:i_en+1])
        self.hc[i_st:i_en+1], self.i_hc = _interp(tm, hc, t[i_st:i_en+1])
        self.t = t


