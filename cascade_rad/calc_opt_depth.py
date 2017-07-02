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

    'calc_opt_depth' : method to be called for optical depth calc.

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
    def __init__(self, bb_temp, size):
        """
        Parameters
        ----------
        bb_temp: astropy.units.Quantity
            blackbody temperature in kelvin
        size : astropy Quantity
            characteristic size of extent of soft photon field
        """
        self.T = bb_temp.to('K')
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
        norm = 0.19 * _sigma_T * (1 - beta ** 2)
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


if __name__ == '__main__':
    Tarr = [1000, 500] * u.K
    s = 0.1 * u.kpc

    Emin = 4e2 * u.GeV
    Emax = 2e2 * u.TeV
    Earr = np.linspace(Emin, Emax, 5000)

    tau_dict = {}

    for i, T in enumerate(Tarr) :
        tau_dict[T] = []
        emc = EMCascade(T, s)
        for E in Earr:
            tau = emc.calc_opt_depth(E)
            tau_dict[T].append(np.exp(-tau))

    for key, tau in tau_dict.items():
        plt.semilogx(Earr, tau, lw=2, label='T = {} K'.format(int(key.value)))
        plt.hold('True')
    plt.title('Optical Depth as a function of Gamma-ray Energy')
    plt.xlabel(r'$E_\gamma$ [TeV]')
    plt.ylabel(r'$exp(- \tau_{\gamma\gamma})$')
    plt.legend(loc='best')
    plt.savefig('./images/exptau_comparison.png')
    plt.show()





