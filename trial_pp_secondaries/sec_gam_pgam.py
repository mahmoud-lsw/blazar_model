from naima.models import PionDecayKelner06
from naima.radiative import _validate_ene
from naima.extern.validator import validate_scalar
from naima.models import PowerLaw as PL
import astropy.units as u
from astropy.constants import m_e, m_p, c, e, h, hbar, k_B
from scipy.integrate import quad
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt


class PionDecay_pgamma(PionDecayKelner06):
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
        return self.pd(E * u.TeV).to('1/TeV').value

    def _softphoton_dist(self, e, T):
        """ Blackbody spectrum : No. of photons / energy
        Parameters
        ----------
        e : float
            energy of photon (in TeV)

        T : 'astropy.units.Quantity' float
            Temperature of Blackbody (in Kelvin)
        """
        kt = (k_B * T).to('TeV')
        norm = 1 / np.pi ** 2
        num = e ** 2
        denom = (np.exp((e * u.TeV) / (k_B * T).to('TeV'))).value
        
        return norm * (num / denom)

    def interp_tab1(self, eta):
    """ From Table-1 Kelner2008 : Interpolate numerical values of s, delta, B 
    as a function of eta (= eta/0.313) characterizing the gamma-ray spectrum.
    
    References
    -----------
    Table I : Kelner, S.R., Aharonian, 2008, Phys.Rev.D 78, 034013
(`arXiv:astro-ph/0803.0688 <https://arxiv.org/abs/0803.0688>`_).

    """
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

    def Phi_g(self, eta, x):
        """ Kelner2008 Eq27-29

        Parameters
        ----------
        x : float
            E_g/E_p
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

    pdist1 = ECPL(4.3e33 / u.eV, 1e3 * u.GeV, 1.3, 8e5 * u.TeV)
    ppi1 = PionDecay_leptons(pdist1, B=10 * u.G)

    specfrq = np.logspace(14, 50, 100) * u.Hz
    specen = specfrq.to(u.eV, equivalencies=u.spectral())
    dist = 1 * u.Mpc

    pds = [ppi1, ]
    ls = ['-', ]
    colors = ['royalblue', ]
    labels = ['Exp.cutoff PWL', ]

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1, 1, 1)
    font = {'family': 'serif', 'color': 'black',
            'weight': 'normal', 'size': 16.0}
    for pd, ls, cs, lb in zip(pds, ls, colors, labels):
        sed = pd.sed(specen, dist)
        ax.loglog(specen, sed, lw=2,
                  color=cs, ls=ls, label=lb)
    ax.set_xlabel('Energy (eV)', fontsize=17)
    ax.set_ylabel(
        r'$E^{2}*{\rm d}N/{\rm d}E\,[erg\,cm^{-2}\,s^{-1}]$', fontsize=17)
    ax.set_title('Secondary e+/e- spectra from p-p interactions', fontsize=17)
    ax.set_ylim(1e-60, 1e-10)
    ax.set_xlim(1e0, 1e21)
    plt.legend(loc='best', borderpad=2, fontsize=14)
    plt.savefig('pp_second_leptons.png')
    plt.show()
