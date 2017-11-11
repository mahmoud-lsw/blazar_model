from __future__ import division, print_function
import numpy as np
import astropy.units as u
import astropy.constants as const
from scipy import integrate
from scipy.special import kv
import naima
from naima.utils import trapz_loglog
import yaml
import matplotlib.pyplot as plt


__all__ =  ['Synchrotron', 'InverseCompton', 'numerics']


e = const.e.gauss
m_e = const.m_e.cgs
c = const.c.cgs
mec = (const.m_e * const.c).cgs
mec2 = (const.m_e * const.c**2).cgs
sigma_T = const.sigma_T.cgs
alpha = 1/137 # fine structure constant
B_cr = 4.4 * 1e13 * u.G


class Synchrotron(object):
    """Class for evaluating synchrotron emission

    Literature used for this class:
    1) MNRAS (1999) 306, 551-560
       SAO/NASA ads : http://adsabs.harvard.edu/abs/1999MNRAS.306..551C

    2) MNRAS (1991) 252, 313-318
       SAO/NASA ads : http://adsabs.harvard.edu/abs/1991MNRAS.252..313G

    3) Rybichi and Lightman
       Radiative processes in Astrophysics

    a first part of the function of the model is directly implemented from
    Ref. 1)
    a second part of the function is implemented through naima, so we can
    cross-check both

    Parameters
    ----------
    model : `~ssc.model.BaseModel`
        model defining the attributes of the emitting region
    """

    def __init__(self, model):
        self.model = model

    def G(self, t):
        """Dimensionless part of Eq. (13) in Ref. (1)
        """
        term1 = kv(4/3,t)*kv(1/3,t)
        term2 = -3/5*t*(kv(4/3,t)**2 - kv(1/3,t)**2)
        return t**2 * (term1 + term2)

    def Ps(self, nu, gamma):
        """Single particle synchrotron emissivity, Eq. (13) in Ref. (1)

        Parameters
        ----------
        nu : `~astropy.quantity.Quantity`
            array of frequencies to evaluate the synchrotron radiation
        gamma : float (`~numpy.ndarray`)
            Lorentz factor(s) of the electron generating synchrotron radiation

        Returns
        -------
        `~astropy.quantity.Quantity`
            single particle emissivity [erg s-1 cm-2 sr-1 Hz-1]
        """
        nu = nu.to('Hz').value
        B = self.model.blob.B
        U_B = self.model.blob.U_B
        # Larmor frequency (dimensionless)
        nu_L = (e.value * B.value)/(2 * np.pi * m_e.value * c.value)

        t = nu/(3 * gamma**2 * nu_L)
        pref = 3*np.sqrt(3)/np.pi * (sigma_T.value * c.value * U_B.value)/nu_L
        unit = u.Unit('erg s-1 cm-2 sr-1 Hz-1')
        return pref * t**2 * self.G(t) * unit

    def sigma_syn(self, nu, gamma):
        """Synchrotron cross-section, Eq. (2.17) in Ref. (2)

        same parameters as Ps
        """
        nu = nu.to('Hz').value # strip the units
        B = self.model.blob.B
        nu_L = (e.value * B.value)/(2 * np.pi * m_e.value * c.value) # Larmor Frequency
        nu_c = 3/2 * gamma**2 * nu_L
        x = nu/nu_c

        # term1 has automatically cm2 units
        term1 = np.sqrt(3) * np.pi/10 * sigma_T/alpha * B_cr/B
        term2 = x/gamma**5 * (kv(4/3, x/2)**2 - kv(1/3, x/2)**2)
        return term1 * term2

    def plot_synchro_cross_section(self):
        """Check the cross section againts Fig. 2 of Ref. (2)"""
        B = 1 * u.G
        p = np.asarray([3, 10, 30]) # values of electron kinetic energy
        gamma = np.sqrt(p**2 - 1) # corresponding Lorentz factors

        nu_nuL = np.logspace(-2, 5) # nu/nu_L, x-axis of Fig. (2)
        nu = nu_nuL * (e.value * B.value)/(2 * np.pi * m_e.value * c.value) * u.Hz

        _nu, _gamma = np.meshgrid(nu, gamma)
        cross_section = self.sigma_syn(_nu, _gamma)

        for _,p in enumerate(p):
            plt.loglog(nu_nuL, cross_section[_,:]/sigma_T, lw=2, label='p =' + str(p))

        plt.ylim([1e-2,1e18])
        plt.xlabel(r'$\nu/\nu_L$', size=12)
        plt.ylabel(r'$\sigma_{\rm synchro} / \sigma_{\rm T}$', size=12)
        plt.legend()
        plt.show()

    def eps_nu(self, nu, base_electron):
        """Emissivity of the injected spectrum, Eq. (12) in Ref.(1)

        Parameters
        -----------
        nu : `~astropy.units.Quantity`
            frequency at which the emissivity has to be evaluated
        base_electron : `~ssc.BaseElectron`
            electron distribution over which to evaluate the emissivity
            emissivity = convolution(single particle emitted power, electron distribution)
        Returns
        --------
        `~astropy.units.Quantity`
            emissivity of the population of electrons over the array of frequencies
        """
        # define the parent electron distribution
        N_e = base_electron.density
        gamma = base_electron.gamma
        # there is no need to strip the units from nu, Ps will make it
        # meshgrid on nu and gamma
        _nu, _gamma = np.meshgrid(nu, gamma)
        # evaluate Ps on it
        sing_part_emiss = self.Ps(_nu, _gamma)
        prefactor = 1 / (4 * np.pi)
        unit = u.Unit('erg cm-3 s-1 Hz-1 sr-1')
        return prefactor * integrate.simps(np.vstack(N_e) * sing_part_emiss, gamma, axis=0) * unit

    def k_factor(self, nu, base_electron):
        """Absorption factor as in Eq.(1.3) of Ref.(2)

        The integral is actually the convolution of the synchrotron cross section
        with the injected spectrum.

        same parameters as eps_nu
        """
        # define the parent electron distribution
        N_e = base_electron.density
        gamma = base_electron.gamma
        # there is no need to strip the units from nu, sigma_syn will make it
        # meshgrid on nu and gamma
        _nu, _gamma = np.meshgrid(nu, gamma)
        # evaluate synchrotron cross section on it
        sigma = self.sigma_syn(_nu, _gamma)
        unit = 1/u.cm
        return integrate.simps(np.vstack(N_e) * sigma, gamma, axis=0) * unit

    def synch_abs_factor(self, nu, base_electron):
        """absorption factor in Eq.(14) of Ref.(1)"""
        tau = self.model.blob.R * self.k_factor(nu, base_electron)
        abs_factor = (-np.expm1(-tau)) / tau
        # avoid nan when going to higher frequencies
        abs_factor = np.where(np.isnan(abs_factor), 1.0, abs_factor)
        return abs_factor

    def I_nu(self, nu, base_electron):
        """Intrinsic Synchrotron intensity of the injected spectrum, Eq. (12) in Ref.(1)
        """
        return self.eps_nu(nu, base_electron) * self.model.blob.R * self.synch_abs_factor(nu, base_electron)

    def I_nu_obs(self, nu, base_electron):
        """Observed Synchrotron intensity of the injected spectrum, Eq. (12) in Ref.(1)
        """
        return self.model.blob.delta**3 * self.I_nu(nu, base_electron)

    def F_nu(self, nu, base_electron):
        """Intrinsic Synchrotron Flux of the injected spectrum, for a uniform sphere
        to convert from intensity to flux I use Eq. (1.13) of Ref. (3)
        """
        factor = np.pi * (self.model.blob.R / self.model.blob.distance)**2
        return self.I_nu(nu, base_electron) * factor.decompose() * u.sr # to remove 1/sr

    def F_nu_obs(self, nu, base_electron):
        """Observed Synchrotron Flux of the injected spectrum, for a uniform sphere
        to convert from intensity to flux I use Eq. (1.13) of Ref. (3)
        """
        return self.model.blob.delta**3 * self.F_nu(nu, base_electron)

    def _naima(self, base_electron):
        """Return a naima synchrotron model object

        http://naima.readthedocs.io/en/latest/radiative.html#synchrotron

        Parameters
        ----------
        base_electron : `~ssc.model.BaseElectron`
            electron distribution producing the synchrotron radiation

        Returns
        -------
        `~naima.models.Synchrotron`
        naima class defining the synchrotron emsission
        """
        gamma = base_electron.gamma
        N_e = base_electron.density
        energy = (gamma * mec2).to('eV')
        # make Ghisellini [cm-3] density compatible with naima density [eV-1]
        N_e_differential = N_e * self.model.blob.volume / mec2

        electron_density = naima.models.TableModel(energy,
                                                   N_e_differential,
                                                   amplitude=1)

        return naima.models.Synchrotron(electron_density, B=self.model.blob.B)

    def flux(self, nu, base_electron, self_absorption=True):
        """return the flux [eV-1 cm-2 s-1], this function is based on naima
        we can choose if we want to account or not for synchrotron self absorption

        Parameters
        ----------
        nu : `~astropy.units.Quantity`
            frequency of emitted photons
        base_electron : `~ssc.model.BaseElectron`
            electron distribution producing the synchrotron radiation
        self_absorption : boolean
            choose if to account or not for synchrotron self absorption

        Returns
        -------
        `~astropy.units.Quantity`
        synchrtron flux [cm-2 s-1]
        """
        # this is the energy of the emitted photons, not of the electrons
        # we adopt the convention that `energy` onl refers to electrons
        photon_energy = const.h * nu
        # define a base naima synchrotron object
        synchro = self._naima(base_electron)
        # boost the radiation
        flux = self.model.blob.delta**3 * synchro.flux(photon_energy, distance=self.model.blob.distance)
        if self_absorption:
            return flux * self.synch_abs_factor(nu, base_electron)
        else:
            return flux

    def U_rad_eval(self, gamma, nu, base_electron):
        """Energy density of the synchrotron radiation field
        Eq. (17) of Ref. (1)
        naima can give us a flux [cm-2 s-1], but, according to Eq.(17)
        U_rad = 4 pi/c * integral(I_nu, dnu)
        where I_nu = [erg cm-2 s-1 Hz-1] (sr-1 are removed by 4 pi)
        the outcome of this integral is erg cm-3, what we will do here
        is to get a flux at the source (i.e. [cm-2 s-1]) and then integrate in
        energy, for us:
        U_rad = 4 pi/c * integral(I [cm-2 s], dE)
        this will return erg cm-3 as well

        Parameters
        ----------
        gamma : float
            value of the grid at which the denisty of radiation has to be evaluated
        nu : `~astropy.units.Quantity`
            frequency of emitted photons
        base_electron : `~ssc.BaseElectron`
            electron distribution producing the synchrotron radiation

        Returns
        -------
        `~astropy.units.Quantity`
        density of radiation at that value of the grid
        """
        # this is the factor that converts from I_nu -> F_nu
        factor = np.pi * (self.model.blob.R / self.model.blob.distance)**2
        # invoke a synchrotron object
        synchro = self._naima(base_electron)
        # get the intrinsic SED
        photon_energy = (const.h * nu).to('erg')
        sed = synchro.sed(photon_energy, distance=self.model.blob.distance)
        sed = sed.to('erg cm-2 s-1')
        # we have nu F_nu, get F_nu
        F_nu = sed / nu
        # we divide for that geometric factor and obtain I_nu
        I_nu = (F_nu / factor.decompose()) # there is also sr-1 but we neglect since we will multiply by 4 pi
        # we absorb it for synchrotron self compton
        I_nu * self.synch_abs_factor(nu, base_electron)
        # choose the extreme of integration as shown after Eq. (17)
        nu_min = nu[0].value
        nu_max = np.min([nu[-1].value,
                         (3 * mec2 / (4 * const.h * gamma)).to('Hz').value])
        # define a new nu grid on which to integrate
        nu_integr = np.logspace(np.log10(nu_min),
                                np.log10(nu_max),
                                len(nu)) * u.Hz
        U_rad = 4 * np.pi / const.c * trapz_loglog(I_nu, nu_integr)
        return U_rad.to('erg / cm3')

    def U_rad(self, gamma, nu, base_electron):
        """return U_rad for each value of gamma"""
        U_rad = np.asarray([self.U_rad_eval(_, nu, base_electron).value for _ in gamma])
        return U_rad * u.Unit('erg / cm3')


class InverseCompton(object):
    """Class for evaluating Inverse Compton emission

    based on naima Inverse Compton implementation
    http://naima.readthedocs.io/en/latest/radiative.html#inverse-compton

    We are assuming that the Inverse Compton is happening on Synchrotron photons
    Only this case is considered for now

    Parameters
    ----------
    model : `~ssc.model.BaseModel`
        model defining the attributes of the emitting region
    """

    def __init__(self, model):
        self.model = model

    def _naima(self, base_electron):
        """Return a naima inverse compton model object

        http://naima.readthedocs.io/en/latest/radiative.html#synchrotron

        Parameters
        ----------
        nu : `~astropy.units.Quantity`
            frequency of emitted photons
        base_electron : `~ssc.model.BaseElectron`
            electron distribution producing the IC radiation

        Returns
        -------
        `~naima.models.Synchrotron`
        naima class defining the synchrotron emsission
        """
        gamma = base_electron.gamma
        N_e = base_electron.density
        # energy of the injected electrons
        energy = (gamma * mec2).to('eV')
        N_e_differential = N_e * self.model.blob.volume / mec2
        # conversion Ref. (1) -> naima // [cm-3] -> [eV-1] densities
        electron_density = naima.models.TableModel(energy,
                                                   N_e_differential,
                                                   amplitude=1)

        # here we have to assume a range of frequency to create a IC object
        # therefore we assume a very large range of frequencies
        photon_energy = np.logspace(-5, 4.5, 100) * u.eV # naima standard for SSC example

        # we calculate the ssc as prescribed in
        # http://naima.readthedocs.io/en/latest/radiative.html#inverse-compton
        # see synchrotron self compton
        # first we have to create a Synchroton (ssc, not naima object)
        synchro_ssc = Synchrotron(self.model)
        # we calculate the photn density at source, using naima
        L_syn = synchro_ssc._naima(base_electron).flux(photon_energy, distance=0)
        # we absorb it
        nu = (photon_energy/const.h).to('Hz')
        L_syn *= synchro_ssc.synch_abs_factor(nu, base_electron)
        # Define source radius and compute photon density
        R = self.model.blob.R
        phn_syn = L_syn / (4 * np.pi * R**2 * const.c) * 2.26
        # Create IC instance with CMB and synchrotron seed photon fields:
        IC = naima.models.InverseCompton(electron_density, seed_photon_fields=['CMB',['SSC', photon_energy, phn_syn]])

        return IC

    def flux(self, nu, base_electron, ebl_absorption=False):
        """return the flux [eV-1 cm-2 s-1], this function is based on naima
        we can choose if we want to account or not for synchrotron self absorption

        Parameters
        ----------
        nu : `~astropy.units.Quantity`
            frequency of emitted photons
        base_electron : `~ssc.model.BaseElectron`
            electron distribution producing the synchrotron radiation
        ebl_absorption : boolean
            choose if to consider EBL absorption

        Returns
        -------
        `~astropy.units.Quantity`
        synchrtron flux [cm-2 s-1]
        """
        # this is the energy of the emitted photons, not of the electrons
        # we adopt the convention that `energy` onl refers to electrons
        photon_energy = const.h * nu
        # define a base naima synchrotron object
        ic = self._naima(base_electron)
        # boost the radiation
        flux = self.model.blob.delta**3 * ic.flux(photon_energy, distance=self.model.blob.distance)
        # define EBL absorption
        ebl_model = naima.models.EblAbsorptionModel(self.model.blob.z, ebl_absorption_model='Dominguez')
        transmissivity  = ebl_model.transmission(self.model.blob.delta * photon_energy)

        if ebl_absorption:
            return flux * transmissivity
        else:
            return flux


class numerics(object):
    """Numeric procedures needed to evaluate electron population evolution

    Parameters
    ----------
    model : `~ssc.model.BaseModel`
        model defining the attributes of the emitting region
    """

    def __init__(self, model):
        self.model = model

    def ChaCoo_tridiag_matrix(self, U_rad):
        """Implementing tridiagonal matrix of Eq. (9) in Ref. (1)

        Parameters
        ----------
        U_rad : `~astropy.units.Quantity`
            array of values of energy density per Lorentz factor bin

        Returns
        -------
        `~numpy.ndarray`
        matrix regulating the soultion of the differential equation
        """
        def cool_rate(U_rad, gamma):
            """cooling rate from Eq. (2) in Ref. (1)
            U_B is fixed so we don't need to pass it as argument"""
            _rate =  4 / 3 * sigma_T / mec * (self.model.blob.U_B.value + U_rad) * gamma**2
            return _rate.value

        # implementing the terms in Eq. (10) of Ref. (1)
        # V1 is an array of zeros, no need to be implemented
        # V2
        V2 = 1 + self.model.delta_t/self.model.t_esc + \
                 self.model.delta_t * cool_rate(U_rad, self.model.gamma_midpts[:-1])/delta_gamma

        V3 = - delta_t * cool_rate(U_rad, self.model.gamma_midpts[1:])/delta_gamma

        ChaCoo = np.diagflat(V2, 0) + np.diagflat(V3,-1)
        return ChaCoo

'''
        # loop on the energy to fill the matrix
        for i in range(N):
            gamma_minus_half = self.model.gamma_grid_midpts[i]                              # \gamma_{j-1/2} of Reference
            gamma_plus_half = self.model.gamma_grid_midpts[i] + self.model.delta_gamma[i]   # \gamma_{j+1/2} of Reference
            delta_gamma = self.model.delta_gamma[i]
            delta_t = self.model.delta_t
            t_esc = self.model.t_esc
            U_B = self.model.U_B
            # when we'll introduce synchrotron radiation U_rad will be an array
            # with a value for each point of the gamma_grid (the extremes of integration depend on gamma)
            U_rad = self.model.U_rad[i]
            # Eq.s (10) of Reference
            V2 = 1 + delta_t/t_esc + \
                     (delta_t * cool_rate(U_B, U_rad, gamma_minus_half))/delta_gamma
            V3 = - (delta_t * cool_rate(U_B, U_rad, gamma_plus_half))/delta_gamma
            # let's loop on another dimension to fill the diagonal of the matrix
            for j in range(N):
                if j == i:
                    ChaCoo_matrix[i,j] = V2
                if j == i+1:
                    ChaCoo_matrix[i,j] = V3

        # Chang Cooper boundaries condition, Eq.(18) Park Petrosian, 1996
        # http://adsabs.harvard.edu/full/1996ApJS..103..255P
        ChaCoo_matrix[N-2,N-1] = 0.
        return ChaCoo_matrix


    def evolve(self):

        # injected spectrum
        #N_e = self.model.N_e_inj
        N_e = np.zeros(len(self.model.gamma_grid))
        # injecton term, to be added each delta_t up to the maximum injection time
        # specified by model.inj_time
        Q_e = self.model.powerlaw_injection
        delta_t = self.model.delta_t

        # time grid loop
        time_past_injection = 0.

        for time in self.model.time_grid:
            # here N^{i+1} of Reference --> N_e_tmp
            # here N^{i} of Reference --> N_e

            # solve the system with Eq.(11):
            if time_past_injection <= self.model.inj_time:
                N_e_tmp = np.linalg.solve(self.ChaCoo_tridiag_matrix(), N_e + Q_e*delta_t)
            # no more injection after the established t_inj
            else:
                N_e_tmp = np.linalg.solve(self.ChaCoo_tridiag_matrix(), N_e)

            # swap!, now we are moving to the following istant of time
            # N^{i+1} --> N^{i} and restart
            N_e = N_e_tmp
            # update the time past injection
            #print ('t after injection: ', time_past_injection/self.model.crossing_time, ' crossing time')
            time_past_injection += delta_t

        return N_e
'''
