################################################################################
# simple attempt of implementing formula (10.27) of "HE radiation from BH"
# 10 July 2017
# cosimo
################################################################################
from __future__ import division
from math import log, sqrt, pi
import numpy as np
from scipy.integrate import quad
import astropy.constants as const
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from sympy.functions.special.delta_functions import Heaviside
import matplotlib.pyplot as plt

# we declare our cosmology for later caluclations
cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

def phi_bar(s_0):
    """
    Implementing Eq. (10.9)
    """
    if s_0 > 1:

        beta_0 = sqrt(1 - 1/s_0)
        w_0 = (1 + beta_0)/(1 - beta_0)

        def _L(w):
            """
            Integrand of Eq. (10.10)
            """
            return 1/w*log(1+w)

        # first line of Eq. (10.9)
        term1 = (1 + beta_0**2)/(1 - beta_0**2)*log(w_0) - beta_0**2*log(w_0) - \
        log(w_0)**2 - (4*beta_0)/(1 - beta_0**2)
        # second line of Eq. (10.9)
        term2 = 2*beta_0 + 4*log(w_0)*log(1 + w_0) - 4*quad(_L,1.,w_0)[0]

        return term1 + term2

    else: return 0


def plot_phi_bar():
    """
    simple function for plotting phi_ba and checking it with Figure 10.2 of ref.
    """
    s_0 = np.linspace(1.1,10,1e2)
    _phi = np.array([phi_bar(s) for s in s_0])

    plt.plot(s_0,_phi/(s_0 - 1), lw=1.5)
    plt.xlabel(r'$s_0$', fontsize=14)
    plt.ylabel(r'$\frac{\overline{\phi}(s_0)}{s_0 - 1}$', fontsize=18)
    plt.ylim([0,5.])
    plt.show()


def tau_YY(E, z, eps_pk, f_eps_pk, t_var, delta_D, a, b, x_a=1e-4, x_b=1e4):
    """
    Function for evaluating the optical depth as a function of the energy
    in the case of target photon gas with broken power-law photon distribution
    Eq. (10.27) of reference

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
    eps_1 = (E/(const.m_e*const.c**2)).decompose().value
	# this is the multiplicative factor of the integral in the second line
	# in Eq. (10.27)
    prefactor_num = 3*pi*r_e**2*d_L**2*f_eps_pk
    prefactor_denom = const.m_e*const.c**4*t_var*eps_1**2*eps_pk**3*(1+z)**4
    prefactor = (prefactor_num/prefactor_denom).decompose().value

	# the integral
    w = (1+z)**2*eps_1*eps_pk/delta_D**2

    integrand1 = lambda x: phi_bar(x*w)/(x**(4-a))
    integral1 = Heaviside(1-1/w)*quad(integrand1, max(x_a,1/w), 1)[0]

    integrand2 = lambda x: phi_bar(x*w)/(x**(4-b))
    integral2 = quad(integrand2, max(1,1/w), x_b)[0]

    return prefactor * (integral1 + integral2)


plot_phi_bar()

En = np.logspace(9,16,50) * u.Unit('eV')

fig = plt.figure()

for [a,b] in [[1/2,-1/2],[1,-1],[2,-2]]:
    tau = np.array([tau_YY(_E, 1, 1e-5, 1e-10*u.Unit('erg cm-2 s-1'), 1e5*u.s , 10, a, b) for _E in En])
    plt.loglog(En, tau, lw=1.5, label = 'a=' + str(a) + ' , b=' + str(b))

plt.legend(loc=2)
plt.xlabel('E[eV]')
plt.ylabel(r'$\tau_{\gamma \gamma}(E)$')
plt.ylim([1e-2, 1e6])
plt.show()
fig.savefig('tau_PWL_trials.png')
