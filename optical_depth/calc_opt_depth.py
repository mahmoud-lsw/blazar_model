########################################################################
# Module for optical depth calculation
# CalcOptDepthPWL : Calculates optical depth for PowerLaw target photon field
# CalcOptDepthBB : Calculates optical depth for Blackbody target photon field

# References :
# 1) 'HE Radiation from BH' (authors : Dermer, C. D. et al.)
# 2) 'https://arxiv.org/pdf/1706.07047v1.pdf'
# 3) 'http://onlinelibrary.wiley.com/doi/10.1111/j.1365-2966.2008.13315.x/full'

# 13 July 2017
# Authors : Cosimo Nigro (cosimo.nigro@desy.de),
#           Wrijupan Bhattacharyya (wrijupan.bhattacharyya@desy.de)
#########################################################################
from __future__ import division
from astropy import units as u
from astropy.constants import m_e, k_B, sigma_T, c, hbar
from scipy.integrate import quad
import numpy as np
from numpy import log, sqrt
from math import log, sqrt, pi
import astropy.constants as const
from astropy.cosmology import FlatLambdaCDM
from sympy.functions.special.delta_functions import Heaviside
import matplotlib.pyplot as plt

_sigma_T = sigma_T.cgs.value
_mec2 = (m_e * c ** 2).to('eV').value
_mec2_u = (m_e * c ** 2).to('eV')
# we declare our cosmology for later caluclations
cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

__all__ = ['CalcOptDepthBB', 'CalcOptDepthPWL', ]


class CalcOptDepthBB(object):
    """
    Module to calculate optical depth as a function of gamma-ray energy
    for BlackBody Soft Photon Field, whose temperature is an input
    parameter of TauCalcBB class.

    'calc_opt_depth' : method to be called for optical depth calc.

    Issues :
    --------
    1) To get a wider energy coverage, finer tuning of the energy
       array is required to get a good shape of exp tau curve which
       increases the computation time. Scipy 'QUADPACK' seems to be
       quite slow. Any way to circumvent this problem?
    2) Determine a more realistic scale of extent of the soft
       photon field (size parameter)
    """

    def __init__(self, size, bb_temp=2.7*u.K, norm=1.5e34 * u.Unit('cm-3 eV-3')):
        """
        Parameters
        ----------
        bb_temp: astropy.units.Quantity
            blackbody temperature in Kelvin
        size : astropy Quantity
            characteristic size of extent of soft photon field
        norm : astropy Quantity
             normalization of soft photon field
        """
        self.T = bb_temp.to('K')
        self.size = size.to('cm').value
        self.norm = norm.to(u.Unit('cm-3 eV-3'))

    def _softphoton_dist(self, e):
        """ Planckian BB spectrum : No. of photons / energy / cm3
        Arbitrarily chosen norm. Should norm be left as free param?
        Parameters
        ----------
        e : float
            energy of photon (normalized by mec2)
        """
        kT = ((k_B * self.T).to('eV').value) / _mec2
        num = e ** 2
        denom = (np.exp(e / kT)) - 1

        return (self.norm * (num / denom)).value

    def sigma(self, e, E):
        """
        Eq. (10.1) of Ref. 1)
        e = float
            soft photon energy (normalized)
        E = float
            Gamma-ray energy (normalized)
        """
        s = e * E
        beta = sqrt(1 - (1. / s))
        norm = 3./16 * _sigma_T * (1 - beta ** 2) # we use 3/16 sigma_T -> 1/2 pi r_e**2
        t1 = (3 - beta ** 4) * log((1 + beta) / (1 - beta))
        t2 = 2 * beta * (2 - beta ** 2)
        return norm * (t1 - t2)

    def tau_integrand(self, s, E):
        """
        Integrand of Eq. (3.1) of Ref. 2)
        Parameter
        ---------
        s : float
            CMS-frame Lorentz_fac^2 of e+e-
        E : float
            gamma-ray energy (normalized)

        """
        return ((1 / E) * self.sigma(s / E, E) * self._softphoton_dist(s / E))

    def calc_opt_depth(self, E):
        tau = self.size * quad(self.tau_integrand, 1., 1e2, args=E,)[0]
        return tau


class CalcOptDepthPWL(object):
    """
    Module to calculate optical depth as a func of gamma-ray energy
    for Broken PWL target Photon Field.

    'tau_YY' : Method to be called for optical depth calculation
    """

    def __init__(self):
        pass

    def phi_bar(self, s_0):
        """
        Implementing Eq. (10.9) of Ref.1
        """
        if s_0 > 1:

            beta_0 = sqrt(1 - 1 / s_0)
            w_0 = (1 + beta_0) / (1 - beta_0)

            def _L(w):
                """
                Integrand of Eq. (10.10)
                """
                return 1 / w * log(1 + w)

            # first line of Eq. (10.9)
            term1 = (1 + beta_0 ** 2) / (1 - beta_0 ** 2) * log(w_0) - beta_0 ** 2 * log(w_0) - \
                log(w_0) ** 2 - (4 * beta_0) / (1 - beta_0 ** 2)
            # second line of Eq. (10.9)
            term2 = 2 * beta_0 + 4 * \
                log(w_0) * log(1 + w_0) - 4 * quad(_L, 1., w_0)[0]

            return term1 + term2

        else:
            return 0

    def plot_phi_bar(self):
        """
        simple function for plotting phi_bar and checking it with Figure 10.2 of ref.
        """
        s_0 = np.linspace(1.1, 10, 1e2)
        _phi = np.array([self.phi_bar(s) for s in s_0])

        plt.plot(s_0, _phi / (s_0 - 1), lw=1.5)
        plt.xlabel(r'$s_0$', fontsize=14)
        plt.ylabel(r'$\frac{\overline{\phi}(s_0)}{s_0 - 1}$', fontsize=18)
        plt.ylim([0, 5.])
        plt.show()

    def tau_YY(self, E, z, eps_pk, f_eps_pk, t_var, delta_D, a, b, x_a=1e-4, x_b=1e4):
        """
        Function for evaluating the optical depth as a function of the energy
        in the case of target photon gas with broken power-law photon distribution
        Eq. (10.27) of reference 1.

        Parameters:
        -----------
        E : astropy.units('eV')
            energy of the colliding photon
        z : float
            redshift of the source
        eps_pk : float
            adimensional energy peak of the target photon distribution
        f_eps_pk : astropy.units('erg cm-2 s-1')
            peak of the target photon distribution
        t_var : astropy.units('s')
            measured variability time scale
        delta_D : float
            relativistic doppler factor
        a : float
            first spectral index of the broken power-law describing the target
        b : float
            second spectral index of the broken power-law describing the target

        Returns:
        --------
        float, value of the tau_YY for an incident photon of energy E
        """

        # listed all the values we need for the prefactor calculation
        r_e = 2.8179403227 * 1e-13 * u.cm
        # we have to get the luminosity distance from the redshift
        # we use astropy cosmology
        d_L = cosmo.luminosity_distance(z)
        # dimensionless energy of the colliding photon
        eps_1 = (E / (const.m_e * const.c ** 2)).decompose().value
        # this is the multiplicative factor of the integral in the second line
        # in Eq. (10.27)
        prefactor_num = 3 * pi * r_e ** 2 * d_L ** 2 * f_eps_pk
        prefactor_denom = const.m_e * const.c ** 4 * \
            t_var * eps_1 ** 2 * eps_pk ** 3 * (1 + z) ** 4
        prefactor = (prefactor_num / prefactor_denom).decompose().value

        # the integral
        w = (1 + z) ** 2 * eps_1 * eps_pk / delta_D ** 2

        def integrand1(x): return self.phi_bar(x * w) / (x ** (4 - a))
        integral1 = Heaviside(1 - 1 / w) * \
            quad(integrand1, max(x_a, 1 / w), 1)[0]

        def integrand2(x): return self.phi_bar(x * w) / (x ** (4 - b))
        integral2 = quad(integrand2, max(1, 1 / w), x_b)[0]

        return float(prefactor * (integral1 + integral2))


if __name__ == '__main__':


    ################## Test for Class CalcOptDepthBB #############
    Tarr = [1e3, 1e4, 1e5] * u.K
    # Norm choosen to reproduce Fig. 1 upper panel of ref. [3]
    norm = [1.5e24, 1.5e21, 1.5e18] * u.Unit('cm3 eV')
    s = 1 * u.kpc

    Emin = 1e10 * u.eV
    Emax = 1e13 * u.eV
    Earr = np.linspace(Emin, Emax, 300)

    tau_dict = {}
    Tau = {}

    for T, norm in zip(Tarr, norm):
        tau_dict[T] = []
        Tau[T] = []
        calcdepth = CalcOptDepthBB(s, T, norm)
        for E in Earr:
            E = E.to('eV') / _mec2_u
            tau = calcdepth.calc_opt_depth(E)
            Tau[T].append(tau)
            tau_dict[T].append((1 - np.exp(-tau))/tau)
        print(Tau[T])
    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)
    xticks = Earr.value
    ax.set_xticks(xticks, minor=True)
    ax.grid(which='both', alpha=0.2)
    for key, tau in tau_dict.items():
        ax.loglog(Earr, tau, lw=2, label='$T = 10^{} K $'.format(int(np.log10(key.value))))
        ax.hold('True')
    plt.title('Fig. 1 upper panel : Aharonian et al. 2008 (Ref. [3])')
    plt.xlabel(r'$E_\gamma$ [eV]')
    plt.ylabel(r'$exp(- \tau_{\gamma\gamma})$')
    plt.legend(loc='best')
    fig1.savefig('./images/tau_BB_trials.png')
    plt.show()


    ################## Test for Class CalcOptDepthPWL #############
    taupl = CalcOptDepthPWL()
    taupl.plot_phi_bar()

    En = np.logspace(5, 12, 1000) * u.Unit('eV')

    fig2 = plt.figure()

    for [a, b] in [[1 / 2, -1 / 2], [1, -1], [2, -2]]:
        tau = np.array([taupl.tau_YY(
            _E, 1, 1e-5, 1e-10 * u.Unit('erg cm-2 s-1'), 1e5 * u.s, 10, a, b) for _E in En])
        plt.loglog(En, tau, lw=1.5, label='a=' + str(a) + ' , b=' + str(b))

    plt.legend(loc=2)
    plt.title('Optical Depth vs Gamma-ray Energy (target PowerLaw)')
    plt.xlabel('E[eV]')
    plt.ylabel(r'$\tau_{\gamma \gamma}(E)$')
    plt.ylim([1e-2, 1e6])
    plt.show()
    fig2.savefig('./images/tau_PWL_trials.png')
