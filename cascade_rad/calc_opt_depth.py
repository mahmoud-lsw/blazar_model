from astropy import units as u
from astropy.constants import m_e, k_B, sigma_T, c, hbar
from scipy.integrate import quad
import numpy as np
from numpy import log, sqrt
import matplotlib.pyplot as plt

_sigma_T = sigma_T.cgs.value
_mec2 = (m_e * c ** 2).to('eV').value
_mec2_u = (m_e * c ** 2).to('eV')

__all__ = ['EMCascade']

class EMCascade(object):
    """
    Class to calculate electromagnetic cascade radiation.
    Incomplete at present. At present only a module to calculate
    optical depth as a func of gamma-ray energy.
    Soft photon field is only a BlackBody at present, whose
    temperature is an input parameter of EMC class.

    CAUTION : the 'main' func is very slow. It uses scipy QUADPACK
    and at the same time densely samples the energy.
    Reducing the sample size of E, fastens up the calculation at
    the cost of bad shape of the exp(tau) vs E curve.

    Issues :
    --------
    1) Only 1 BB as soft photon field. To be updated.
    1.a) The soft photon field works for CMB or thermal dust emission.
          But probably not GENERAL enough! #TODO
    2) To get a good shape of the exp(tau) vs Egamma,one has to
       densely sample from Egamma! BUT...scipy QUADPACK extremely slow!!!
       QUES: Direct use of trapezoidal rule probably faster?
    3) Determine a more realistic scale of extent of the soft
       photon field (size parameter)
    4) Is the s range correct (see function 'calc_opt_depth')?
       s : CMS-frame Lorentz factor
    5) Leave the normalizaion of soft ph dist as free parameter?
       (at present fixed)
    """
    def __init__(self, bb_temp, Egamma_min, Egamma_max, size):
        """
        Parameters
        ----------
        bb_temp: astropy.units.Quantity
            blackbody temperature in kelvin
        Egamma_min, max : astropy Quantity
            min, max gamma-ray energy
        size : astropy Quantity
            characteristic size of extent of soft photon field
        """
        self.T = bb_temp.to('K')
        self.Egmin = Egamma_min
        self.Egmax = Egamma_max
        self.size = size.to('cm').value

    def _softphoton_dist(self, e):
        """ Planckian BB spectrum : No. of photons / energy / cm3
        Parameters
        ----------
        e : float
            energy of photon (normalized by mec2)
        """
        kT = ((k_B * self.T).to('eV').value) / _mec2
        hc = hbar.to('eV s') * c.cgs
        norm = (_mec2_u ** 2) / ((hc ** 3) * (np.pi ** 2))
        num = e ** 2
        denom = (np.exp(e / kT)) - 1

        return (norm * (num / denom)).value

    def sigma(self, e, E):
        """
        e = float
            soft photon energy (normalized)
        E = float
            Gamma-ray energy (normalized)

        Reference
        ---------
        'High Energy Radiation From Black Holes'
            (Authors : Charles D. Dermer, Govind Menon)
        Condensed form : Cosimo's presentation
            '2-17-06-27 LITERATURE Meeting Notes'
        """
        s = e * E
        beta = sqrt(1 - (1. / s))
        norm = 0.6 * _sigma_T * (1 - beta ** 2)
        t1 = (3 - beta ** 4) * log((1 + beta) / (1 - beta))
        t2 = 2 * beta * (2 - beta ** 2)
        return norm * (t1 - t2)

    def tau_integrand(self, s, E):
        """
        Integrand of Eqn-3.1 of https://arxiv.org/pdf/1706.07047v1.pdf
        Parameter
        ---------
        s : float
            CMS-frame Lorentz_fac^2 of e+e-
        E : float
            gamma-ray energy (normalized)

        """
        return ((1 / E) * self.sigma(s / E, E) * self._softphoton_dist(s / E))

    def calc_opt_depth(self, E):
        E = E.to('eV') / _mec2_u
        tau = self.size * quad(self.tau_integrand, 1., 1e2, args=E,)[0]
        return tau

    def main_calc_opt_depth(self):
        """main function to produce a plot"""
        Egamma_arr = np.linspace(self.Egmin, self.Egmax, 7000)
        exp_tau = []
        for Egamma in Egamma_arr :
            tau = self.calc_opt_depth(Egamma)
            exp_tau.append(np.exp(-tau))

        plt.semilogx(Egamma_arr, exp_tau, lw=2., label='Thermal dust (T = 50.7 K)')
        plt.title('Optical Depth as a function of Gamma-ray Energy')
        plt.xlabel(r'$E_\gamma$ [TeV]')
        plt.ylabel(r'$exp(- \tau_{\gamma\gamma})$')
        plt.legend(loc='best')
        plt.savefig('./images/exptau_VS_Egam.png')
        plt.show()



if __name__ == '__main__':
    T = 50.7 * u.K
    Emin = 1e0 * u.TeV
    Emax = 7e4 * u.TeV
    s = 100 * u.kpc
    emc = EMCascade(T, Emin, Emax, s)
    emc.main_calc_opt_depth()




