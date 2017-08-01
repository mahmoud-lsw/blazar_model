from naima.radiative import _validate_ene
from naima.models import PowerLaw as PL
import astropy.units as u
from astropy.constants import m_e, m_p, c, e, h, hbar, k_B
from scipy.integrate import quad, dblquad, nquad
from scipy.interpolate import interp1d
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import timeit
import re

__all__ = ['PionDecay_gamma', 'PionDecay_positron', 'PionDecay_electron']


class PionDecay_gamma(object):
    """ Production spectra of secondary photons from
    neutral pion decay produced as secondaries from p-gamma interaction.

    References
    ----------
    Kelner, S.R., Aharonian, 2008, Phys.Rev.D 78, 034013
    (`arXiv:astro-ph/0803.0688 <https://arxiv.org/abs/0803.0688>`_).

    """

    def __init__(self, particle_dist, T=1e4 *u.K, norm=1.5e21*u.Unit('eV-3 cm-3'), **kwargs):
        """ Particle distribution of protons is the only parameter
        """
        self.particle_dist = particle_dist
        self.T = T.to('K')
        self.norm = norm.to(u.Unit('eV-3 cm-3'))
        self.__dict__.update(**kwargs)

    def _particle_dist(self, E):
        return self.particle_dist(E * u.eV).to('1/eV').value

    def _softphoton_dist(self, e):
        """ Blackbody spectrum : No. of photons / energy / cm3
        T fixed to 2.7 K in this case i.e. CMB radiation field
        Parameters
        ----------
        e : float
            energy of photon (in eV)
        """
        kT = (k_B * self.T).to('eV')
        num = (e * u.eV) ** 2
        denom = (np.exp((e / kT).value)) - 1

        return (self.norm * (num / denom)).value

    def lookup_tab1(self, eta, interp_file = "./interpolation_tables/gamma_tab1_ka08.txt"):
        """
        Interpolate the values of s, delta, B for the
        parametrics form of _phi_gamma.

        Interpolation of TABLE-I Kelner2008
        Parameters
        ----------
        eta : float
              (4 * E_soft * E_proton) / mpc2

        interp_file : string
                      interpolation table according
                      to Kelner2008
        Returns
        -------
        s, delta, B : float
                    Return these quantities as function of eta
        """
        interp_table = open(interp_file, "r")
        rows = interp_table.readlines()
        eta_eta0 = []
        s = []
        delta = []
        B = []

        for row in rows:
            entries = re.split(r"\s{1,}", row)
            eta_eta0.append(float(entries[0]))
            s.append(float(entries[1]))
            delta.append(float(entries[2]))
            B.append(float(entries[3]))

        eta_arr = np.array(eta_eta0)
        s_arr = np.array(s)
        delta_arr = np.array(delta)
        B_arr = np.array(B)

        s_int = interp1d(eta_arr, s_arr, kind='linear',
                         bounds_error=False, fill_value="extrapolate")
        delta_int = interp1d(eta_arr, delta_arr, kind='linear',
                             bounds_error=False, fill_value="extrapolate")
        B_int = interp1d(eta_arr, B_arr, kind='linear',
                         bounds_error=False, fill_value="extrapolate")

        s_new = s_int(eta)
        delta_new = delta_int(eta)
        B_new = B_int(eta)

        return s_new, delta_new, B_new

    def _x_plus_minus(self, eta):
        """
        Eqn. 19 Kelner2008

        Parameters
        ----------
        eta : float

        Returns
        -------
        xplus, xminus
        According to Eqn 19
        """
        r = 0.146
        x_1 = eta + r ** 2
        x_2 = np.sqrt((eta - r ** 2 - 2 * r) * (eta - r ** 2 + 2 * r))
        x_3 = 1 / (2 * (1 + eta))

        x_plus = x_3 * (x_1 + x_2)
        x_minus = x_3 * (x_1 - x_2)
        return x_plus, x_minus

    def _phi_gamma(self, eta, x):
        """ Kelner2008 Eq27-29

        Parameters
        ----------
        x : float
            E_gamma/E_p
        eta : float
            see eqn.10 Kelner2008 [TeV]

        Returns
        -------
        phi_gamma : float
                    Eqn27-29 Kelner2008
        """
        x_p, x_n = self._x_plus_minus(eta)

        s, delta, B = self.lookup_tab1(eta / 0.313)
        power = 2.5 + 0.4 * np.log(eta / 0.313)

        if x > x_n and x < x_p:
            y = (x - x_n) / (x_p - x_n)
            ln1 = np.exp(- s * (np.log(x / x_n)) ** delta)
            ln2 = np.log(2. / (1 + y**2))
            return B * ln1 * ln2 ** power

        elif x < x_n:
            return B * (np.log(2)) ** power

        elif x > x_p:
            return 0

    _mpc2 = (m_p * c ** 2).to('eV')
    #_E = 3.0e20 * u.eV
    _E = 8e16 * u.eV
    #Egamma = (0.5 * _E).value

    def _H_integrand(self, x, eta, Egamma):
        """
        Parameters
        -----------
        eta: (4 * e_soft * Ep) / (mp**2 * c**4)
        x: ''astropy.Units.Quantity' float

        Returns: integrand of Eq70 Kelner2008
        """
        return ((1 / Egamma) *
                self._particle_dist(Egamma / x) *
                self._softphoton_dist((eta * self._mpc2**2 * x) / (4 * Egamma)) *
                self._phi_gamma(eta, x))

    @nb.jit
    def _calc_spec_pgamma(self, Egamma):
        """
        Calculation of differential photon spectra

        Parameters
        ----------
        Egamma: 'astropy.units.Quantity' float
                Output Photon energy

        Returns
        --------
        spec_hi : float
                  Differential photon flux
        """
        Egamma = Egamma.to('eV').value
        x_range = [0, 1]
        eta_range = [0.3443, 31.3]
        #opts = {'epsabs':0}
        spec_hi = self._mpc2 * nquad(self._H_integrand, [x_range, eta_range],
                                    args=[Egamma],)[0]

        return spec_hi.value

    @nb.jit
    def _spectrum(self, photon_energy):
        """
        Loop over the output photon energy array
        """

        outspecene = _validate_ene(photon_energy)
        self.specpg = np.zeros(len(outspecene))

        for i, gamma in enumerate(outspecene):
            self.specpg[i] = self._calc_spec_pgamma(gamma)
            print("Executing {} out of {} steps...\n dNdE={}".format(
                i + 1, len(outspecene), self.specpg[i]))

        return self.specpg

class PionDecay_positron(PionDecay_gamma):
    """Production spectra of secondary positrons from
    charged pion decay produced as secondaries from p-gamma interaction.
    """
    def __init__(self, particle_dist, T=1e4 *u.K, norm=1.5e21*u.Unit('eV-3 cm-3'), **kwargs):
        super(PionDecay_positron, self).__init__(particle_dist, T, norm, **kwargs)

    def lookup_tab1(self, eta, interp_file = "./interpolation_tables/positron_tab2_ka08.txt"):
        return super(PionDecay_positron, self).lookup_tab1(eta, interp_file)

    def _x_pm(self, eta):
        xplus, xminus = self._x_plus_minus(eta)
        #x_plus, x_minus = xplus, xminus / 4.
        return xplus, xminus / 4.

    def power_psi(self, eta):
        eta0 = 0.313
        return 2.5 + 1.5 * np.log(eta / eta0)

    def _phi_gamma(self, eta, x):
        x_p, x_n = self._x_pm(eta)

        s, delta, B = self.lookup_tab1(eta / 0.313)
        power = self.power_psi(eta)

        if x > x_n and x < x_p:
            y = (x - x_n) / (x_p - x_n)
            ln1 = np.exp(- s * (np.log(x / x_n)) ** delta)
            ln2 = ln2 = np.log(2. / (1 + y**2))
            return B * ln1 * ln2 ** power

        elif x < x_n:
            return B * (np.log(2) ** power)

        elif x > x_p :
            return 0
        
    def _H_integrand(self, x, eta, Eeplus):
        return super(PionDecay_positron, self)._H_integrand(x, eta, Eeplus)

    @nb.jit
    def _calc_spec_pgamma(self, Eeplus):
        Eeplus = Eeplus.to('eV').value
        x_range = [0, 1]
        eta_range = [0.3443, 31.3]
        spec_hi = self._mpc2 * nquad(self._H_integrand, [x_range, eta_range],
                                    args = [Eeplus],)[0]
        return spec_hi.value

    @nb.jit
    def _spectrum(self, positron_energy):
        """ The peak of the positron spectrum
         shifts towards lower energies w.r.t
         the peak of the photon spectrum.

         Parameters
         ----------
         positron_energy : array_like
                           same as in the photon E-array
        """
        return super(PionDecay_positron, self)._spectrum(positron_energy)


class PionDecay_electron(PionDecay_positron):
    """
    Production spectra of secondary electrons from
    charged pion decay produced as secondaries from p-gamma interaction.
    """
    def __init__(self, particle_dist, T=1e4 *u.K, norm=1.5e21*u.Unit('eV-3 cm-3'), **kwargs):
        super(PionDecay_electron, self).__init__(particle_dist, T, norm, **kwargs)

    def lookup_tab1(self, eta, interp_file = "./interpolation_tables/electron_tab3_ka08.txt"):
        return super(PionDecay_electron, self).lookup_tab1(eta, interp_file)

    def _x_pm(self, eta):
        r = 0.146
        x_1 = 2 * (1 + eta)
        x_2 = eta - 2 * r
        x_3 = np.sqrt(eta * (eta - 4 * r * (1 + r)))

        x_plus = (x_2 + x_3) / x_1
        x_minus = (x_2 - x_3) / x_1
        return x_plus, x_minus

    def heaviside(self, x):
        return (np.sign(x) + 1) / 2.

    def power_psi(self, eta):
        rho = eta / 0.313
        return 6 * (1 - np.exp(1.5 * (4 - rho))) * self.heaviside(rho - 4)

    def _phi_gamma(self, eta, x):
        return super(PionDecay_electron, self)._phi_gamma(eta, x)


    def _H_integrand(self, x, eta, Eeminus):
        return super(PionDecay_electron, self)._H_integrand(x, eta, Eeminus)

    @nb.jit
    def _calc_spec_pgamma(self, Eeminus):
        Eeminus = Eeminus.to('eV').value
        x_range = [0, 1]
        eta_range = [0.66982, 31.3]
        spec_hi = self._mpc2 * nquad(self._H_integrand, [x_range, eta_range],
                                     args=[Eeminus], )[0]
        return spec_hi.value

    @nb.jit
    def _spectrum(self, electron_energy):
        return super(PionDecay_electron, self)._spectrum(electron_energy)


if __name__ == '__main__':

    start = timeit.default_timer()
    pdist1 = PL(4.3e8 / u.eV, 1e3 * u.GeV, 2.5)
    #pg1 = PionDecay_gamma(pdist1, T=1e4 * u.K, norm = 1.5e21 * u.Unit('erg-3 cm-3'))
    #pg1 = PionDecay_positron(pdist1, T=1e4 * u.K, norm = 1.5e21 * u.Unit('erg-3 cm-3'))
    pg1 = PionDecay_electron(pdist1, T=1e4 * u.K, norm = 1.5e21 * u.Unit('erg-3 cm-3'))

    gamma_arr = np.linspace(0.43e-2, 1, 100) * pg1._E

    sed = pg1._spectrum(gamma_arr)
    stop = timeit.default_timer()
    print("Elapsed time for computation = {} secs".format(stop - start))

    font = {'family': 'serif', 'color': 'black',
            'weight': 'normal', 'size': 16.0}
    plt.loglog(gamma_arr, gamma_arr * sed, label='Proton index=2.5',
               lw=2.2, ls='-', color='blue')
    #plt.title(
    #    'Gamma-ray spec. from Neutral Pion Decay (target : CMB ($T=10^4 K$))', fontsize=9)
    #plt.title(
    #    'Positron spectrum from Charged Pion Decay (target : CMB ($T=10^4 K$))', fontsize=9)
    plt.title(
        'Electron spectrum from Charged Pion Decay (target : CMB (T=2.7 K))', fontsize=9)
    plt.xlabel('$Energy (eV)$')
    plt.ylabel(r'$E*{\rm d}N/{\rm d}E\,[cm^{-3}\,s^{-1}]$')
    plt.legend(loc='best')
    #plt.savefig('./images/pgamma_photons.png')
    #plt.savefig('./images/pgamma_positrons.png')
    plt.savefig('./images/pgamma_electrons.png')
    plt.show()
