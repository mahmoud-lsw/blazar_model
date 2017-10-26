from naima.models import PionDecayKelner06
from naima.radiative import _validate_ene
from naima.extern.validator import validate_scalar
from naima.models import ExponentialCutoffPowerLaw as ECPL
from naima.models import ExponentialCutoffBrokenPowerLaw as EBPL
from naima.models import PowerLaw as PL
import astropy.units as u
from astropy.constants import m_e, m_p, c, e, h, hbar
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt


def heaviside(x):
    return (np.sign(x) + 1) / 2.


class PionDecay_leptons(PionDecayKelner06):
    """ Production spectra of secondary electrons from pion decay.

    Compute spectra of secondary leptons (e+e-) from charged pion decay
    due to interaction of relativistic protons with stationary targety protons.

    References
    ----------
    Kelner, S.R., Aharonian, F.A., and Bugayov, V.V., 2006 PhysRevD 74, 034018
    (`arXiv:astro-ph/0606058 <http://www.arxiv.org/abs/astro-ph/0606058>`_).

    """

    def __init__(self, pd, nh=1.0 / u.cm**3, Etrans=0.1 * u.TeV, **kwargs):
        """ Uses the delta function approcimation for Ep < Etrans (default 0.1 TeV)
        """
        self.pd = pd
        self.nh = validate_scalar(
            'nh', nh, physical_type='number density')
        self.Etrans = validate_scalar(
            'Etrans', Etrans, domain='positive', physical_type='energy')

        self.__dict__.update(**kwargs)

    def _particle_dist(self, E):
        return self.pd(E * u.TeV).to('1/TeV').value

    def F_e(self, x, Ep):
        """ Kelner2006 Eq62

        Parameters
        ----------
        x : float
            E_e/E_p (NB: typo in paper)
        Ep : float
            Ep [TeV]
        """
        L = np.log(Ep)
        Be = 1. / (69.5 + 2.65 * L + 0.3 * L**2)
        bete = 1. / (0.201 + 0.062 * L + 0.00042 * L**2)**0.25
        ke = (0.279 + 0.141 * L + 0.0172 * L**2) / (0.3 + (2.3 + L)**2)
        xb = x**bete

        F1 = (1 + ke * (np.log(x))**2) ** 3
        F2 = x * (1 + 0.3 / xb)
        F3 = (-np.log(x)) ** 5

        return Be * (F1 / F2) * F3

    _m_pi = 2.4880643e-28 * u.kg
    _m_pi_tev = 1.349766e-4
    _m_p_tev = 9.3828e-4
    _mpc2 = (m_p * c**2).to('TeV').value
    _c = c.cgs.value
    _Kpi = 0.17

    def _sigma_inel(self, Ep):
        """ 
        This has to be re-written because naima calculates Eth with neutral pion mass.
        In our case it should be calulated for charged pion mass.
        Not a huge difference maybe. But just to be more systematic.
        """
        L = np.log(Ep)  # Ep value on TeV
        sigma = 34.3 + 1.88 * L + 0.25 * L**2
        if Ep < 0.1:
            Eth = ((m_p + 2 * self._m_pi + (self._m_pi.value**2 /
                                            2 * m_p.value) * u.kg) * c**2).to('TeV').value
            Eth_Ep4 = (Eth / Ep)**4
            x = Ep - Eth
            sigma *= (1 - Eth_Ep4)**2 * heaviside(x)

        return sigma * 1e-27

    def _elec_integrand(self, x, Ee):
        """Eqn. 72 Kelner2006"""
        return (self._sigma_inel(Ee / x) *
                self._particle_dist((Ee / x)) * self.F_e(x, (Ee / x)) / x)

    def _calc_specpp_hiE(self, Ee):
        """ Differential sec. elec spectrum analogous to Eq72 Kelner2006.
        For Ep > Etrans = 0.1 TeV (default)
        """
        Ee = Ee.to('TeV').value
        spec_hi = self._c * quad(
            self._elec_integrand, 0., 1., args=Ee, epsrel=1e-3,
            epsabs=0)[0]

        return spec_hi * u.Unit('1/(s TeV)')

    def _delta_integrand(self, Epi):
        Ep0 = self._mpc2 + Epi / self._Kpi
        qpi = (self._c *
               (self.nhat / self._Kpi) * self._sigma_inel(Ep0) *
               self._particle_dist(Ep0))

        return qpi / np.sqrt(Epi**2 + self._m_pi_tev**2)

    def _calc_specpp_loE(self, Ee):
        """
        Delta-functional approximation for low energies Egamma < 0.1 TeV
        """
        Ee = Ee.to('TeV').value
        Epimin = Ee + self._m_pi_tev**2 / (4 * Ee)

        spec_lo = 2 * quad(
            self._delta_integrand, Epimin, np.inf, epsrel=1e-3, epsabs=0)[0]

        return spec_lo * u.Unit('1/(s TeV)')

    def _spectrum(self, electron_energy):
        return super(PionDecay_leptons, self)._spectrum(electron_energy)


if __name__ == '__main__':

    pdist1 = ECPL(2.3e25 / u.eV, 1e3 * u.TeV, 2.0, 7e2 * u.TeV)
    ppi1 = PionDecay_leptons(pdist1, B=10 * u.G)
    ppi2 = PionDecayKelner06(pdist1)

    specen = np.logspace(-5, 3, 1e2) * u.TeV
    dist = 1 * u.kpc

    pds = [ppi1, ppi2]
    ls = ['-', '--']
    colors = ['royalblue', 'red']
    labels = ['$e^{+}/e^{-}$', '$\gamma$-rays']

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1, 1, 1)
    font = {'family': 'serif', 'color': 'black',
            'weight': 'normal', 'size': 16.0}
    for pd, ls, cs, lb in zip(pds, ls, colors, labels):
        sed = pd.sed(specen, dist)
        ax.loglog(specen, sed, lw=5,
                  color=cs, ls=ls, label=lb)
        ax.hold('True')
    ax.set_xlabel('Energy (TeV)', fontsize=20)
    ax.set_ylabel(
        r'$E^{2}\times{\rm d}N/{\rm d}E\,[erg\,cm^{-2}\,s^{-1}]$', fontsize=20)
    ax.set_title('FIG: 12b, Kelner & Aharonian (2006)', fontsize=24)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_ylim(1e-19, 1e-16)
    plt.legend(loc='best', borderpad=2, fontsize=19)
    plt.savefig('fig12aK&A.png')
    plt.show()
