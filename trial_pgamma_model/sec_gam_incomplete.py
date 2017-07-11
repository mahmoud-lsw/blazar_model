from naima.models import Synchrotron
from naima.radiative import _validate_ene
from naima.extern.validator import validate_scalar
from naima.models import PowerLaw as PL
import astropy.units as u
from astropy.constants import m_e, m_p, c, e, h, hbar, k_B
from scipy.integrate import quad, dblquad
from scipy.interpolate import interp1d
import numpy as np
import numba as nb
import matplotlib.pyplot as plt


class PionDecay_pgamma(Synchrotron):
    """ Production spectra of secondary photons from
    neutral pion decay produced as secondaries from p-gamma interaction.

    References
    ----------
    Kelner, S.R., Aharonian, 2008, Phys.Rev.D 78, 034013
    (`arXiv:astro-ph/0803.0688 <https://arxiv.org/abs/0803.0688>`_).

    """

    def __init__(self, pd, **kwargs):
        """ Uses the delta function approcimation for Ep < Etrans (default 0.1 TeV)
        """
        self.pd = pd
        self.__dict__.update(**kwargs)

    def _particle_dist(self, E):
        return self.pd(E * u.eV).to('1/eV').value

    def _softphoton_dist(self, e):
        """ Blackbody spectrum : No. of photons / energy / cm3
        Parameters
        ----------
        e : float
            energy of photon (in eV)
        """
        T = 2.73 * u.K
        kT = (k_B * T).to('eV')
        hc = hbar.to('eV s') * c.cgs
        norm = 1 / ((hc ** 3) * (np.pi ** 2))
        num = (e * u.eV) ** 2
        denom = (np.exp((e / kT).value)) - 1

        return (norm * (num / denom)).value

    def interp_tab1(self, eta):

        eta_arr = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                        3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 
                        30.0, 40.0, 100.0])

        s_arr = np.array([0.0768, 0.106, 0.182, 0.201, 0.219, 0.216, 0.233, 0.233,
                      0.248, 0.244, 0.188, 0.131, 0.120, 0.107, 0.102, 0.0932, 
                      0.0838, 0.0761, 0.107, 0.0928, 0.0772, 0.0479])

        delta_arr = np.array([0.544, 0.540, 0.750, 0.791, 0.788, 0.831, 0.839, 0.825,
                          0.805, 0.779, 1.23, 1.82, 2.05, 2.19, 2.23, 2.29, 2.37, 
                          2.43, 2.27, 2.33, 2.42, 2.59])

        B_arr = np.array([2.86E-019, 2.24E-018, 5.61E-018, 1.02E-017, 1.6E-017, 2.23E-017,
                      3.1E-017, 4.07E-017, 5.3E-017, 6.74E-017, 1.51E-016, 1.24E-016, 
                      1.37E-016, 1.62E-016, 1.71E-016, 1.78E-016, 1.84E-016, 1.93E-016, 
                      4.74E-016, 7.7E-016, 1.06E-015, 2.73E-015])

        s_int = interp1d(eta_arr, s_arr, kind='cubic')
        delta_int = interp1d(eta_arr, delta_arr, kind='cubic')
        B_int = interp1d(eta_arr, B_arr, kind='cubic')

        s_new = s_int(eta)
        delta_new = delta_int(eta)
        B_new = B_int(eta)

        return s_new, delta_new, B_new

    def _phi_gamma(self, eta, x):
        """ Kelner2008 Eq27-29

        Parameters
        ----------
        x : float
            E_gamma/E_p
        eta : float
            see eqn.10 Kelner2008 [TeV]
        """
        r = 0.146
        x_1 = eta + r**2
        x_2 = np.sqrt((eta - r**2 - 2 * r)*(eta - r**2 + 2 * r))
        x_3 = 1 / (2 * (1 + eta))
        
        x_p = x_3 * (x_1 + x_2)
        x_n = x_3 * (x_1 - x_2)

        s, delta, B = self.interp_tab1(eta / 0.313)
        power = 2.5 + 0.4 * np.log(eta / 0.313) 

        if x > x_n and x < x_p :
            y = (x - x_n) / (x_p - x_n)
            ln1 = np.exp(- s * (np.log(x / x_n)) ** delta)
            ln2 = np.log(2. / (1 + y**2))
            return B * ln1 * ln2 ** power

        elif x < x_n :
            return B * (np.log(2)) ** power

        elif x > x_p :
            return 0

    mpc2 = (m_p * c ** 2).to('eV')
    _E = 3.0e20 * u.eV
    Egamma = (0.5 * _E).value

    def _H_integrand(self, x, eta):
        """
        eta: (4 * e_soft * Ep) / (mp**2 * c**4)
        x: ''astropy.Units.Quantity' float

        return: integrand of Eq70 Kelner2008
        """
        return ((1 / self.Egamma) *
               self._particle_dist(self.Egamma / x) *
               self._softphoton_dist((eta * self.mpc2**2 *x) / (4 * self.Egamma))*
               self._phi_gamma(eta, x))

    @nb.jit
    def _calc_specpg_hiE(self, eta):
        """
        """
        #Egamma = self.Egamma.to('eV').value
        #x_range = [0, 1]
        #eta_range = [0.313, np.inf]
        spec_hi = self.mpc2 * quad(
            self._H_integrand, 0., 1., args=eta, epsrel=1e-3, epsabs=0)[0]

        return spec_hi.value

    @nb.jit
    def _spect(self, eta_energy):

        #outspecene = _validate_ene(photon_energy)
        self.specpg = np.zeros(len(eta_energy))

        for i, eta in enumerate(eta_energy):
            self.specpg[i] = self._calc_specpg_hiE(eta)

        return self.specpg


if __name__ == '__main__':

    pdist1 = PL(4.3e40 / u.eV, 1e3 * u.GeV, 2.0)
    pg1 = PionDecay_pgamma(pdist1, B=10 * u.G)

    eta0 = 0.313
    eta_arr = np.linspace(1.1, 100, 200) * eta0

    H1 = pg1._spect(eta_arr)

    font = {'family': 'serif', 'color': 'black',
            'weight': 'normal', 'size': 16.0}
    plt.loglog(eta_arr/eta0, H1, label='PL, $E_{\gamma} = 0.5 E, ind = 2.0$',
               lw=2.2, ls='-', color='blue')
    plt.title('H($\eta, E_{\gamma}$) vs  $\eta / \eta_{0}$: Fig13 Kelner2008')
    plt.xlabel('$\eta / \eta_{0}$')
    plt.ylabel('H($\eta, E_{\gamma}$), a.u.')
    plt.legend(loc='best')
    plt.savefig('fig13_kel2008.png')
    plt.show()