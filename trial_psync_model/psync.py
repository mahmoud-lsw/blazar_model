from naima.models import Synchrotron
from naima.models import ExponentialCutoffPowerLaw as ECPL
from naima.models import ExponentialCutoffBrokenPowerLaw as EBPL
from naima.models import PowerLaw as PL
from astropy.constants import m_e, m_p, c, e, h, hbar
from astropy import units as u
from naima.radiative import _validate_ene
from naima.extern.validator import validate_scalar
from naima.utils import trapz_loglog
import numpy as np
from scipy.special import cbrt
import matplotlib.pyplot as plt

__all__ = ['PSynchrotron']

mpc2 = (m_p * c ** 2).cgs
mpc2_unit = u.Unit(mpc2)
e = e.gauss

class PSynchrotron(Synchrotron):
    """Dummy Proton-Synchrotron Model
       Simplest modification of Naima e-synchrotron code.
       Compute synchrotron for random magnetic field according to approximation
        of Aharonian, Kelner, and Prosekin 2010, PhysRev D 82, 3002
        (`arXiv:1006.1045 <http://arxiv.org/abs/1006.1045>`_).
    """

    def __init__(self, particle_distribution, B = 10 * u.G, **kwargs):
        self.particle_distribution = particle_distribution
        self.B = B
        P = self.particle_distribution(1 * u.TeV)
        validate_scalar('particle distribution', P,
                        physical_type='differential energy')
        self.Epmin = 1e12 * u.eV
        self.Epmax = 4e20 * u.eV
        self.nEpd = 300
        self.__dict__.update(**kwargs)


    @property
    def _energyarr(self):
        """Proton energy array
        """
        return np.logspace(np.log10(self.Epmin.to(u.eV).value),
                           np.log10(self.Epmax.to(u.eV).value),
                           self.nEpd * (np.log10(self.Epmax / self.Epmin)))
        
    @property
    def _lorentzfacarr(self):
        """Lorentz factor of protons
        """
        return self._energyarr / mpc2.to(u.eV).value


    @property
    def _nproton(self):
        """Proton dist per unit Lorentz factor
        """
        pd = self.particle_distribution(self._lorentzfacarr * mpc2)
        return pd.to(1 / mpc2_unit).value

    def _spectrum(self, gammaE):
        """ Calculate diffenential photon spectrum for given energies
        """
        outspec = _validate_ene(gammaE)

        def Gtilde(x):
            cb = cbrt(x)
            gt1 = 1.808 * cb / np.sqrt(1 + 3.4 * cb**2.)
            gt2 = 1 + 2.210 * cb**2. + 0.347 * cb**4.
            gt3 = 1 + 1.353 * cb**2. + 0.217 * cb**4.
            return gt1 * (gt2 / gt3) * np.exp(-x)

        CS1_0 = np.sqrt(3) * e.value**3 * self.B.to('G').value
        CS1_1 = (2 * np.pi * m_p.cgs.value * c.cgs.value
                 ** 2 * hbar.cgs.value * outspec.to('erg').value)
        CS1 = CS1_0 / CS1_1


        # Critical energy calculation
        Ec = 3 * e.value * hbar.cgs.value * self.B.to('G').value * self._lorentzfacarr**2
        Ec /= 2 * (m_p * c).cgs.value

        EgEc = outspec.to('erg').value / np.vstack(Ec)
        dNdE = CS1 * Gtilde(EgEc)
        spec = trapz_loglog(np.vstack(self._nproton) * dNdE,
                            self._lorentzfacarr, axis=0) / u.s / u.erg
        spec = spec.to('1/(s eV)')

        return spec


if __name__ == '__main__':

    pdist1 = ECPL(4.3e33 / u.eV, 1e3 * u.GeV, 1.3, 8e5 * u.TeV)
    pdist2 = EBPL(6.3e33 / u.eV, 8.5e3 * u.GeV, 8e4 * u.GeV, 1.35, 1.75, e_cutoff=9e5 * u.TeV)
    pdist3 = PL(1.3e33 / u.eV, 1e3 * u.GeV, 1.3)    

    SYN1 = PSynchrotron(pdist1, B=10 * u.G)
    SYN2 = PSynchrotron(pdist2, B=10 * u.G)
    SYN3 = PSynchrotron(pdist3, B=10 * u.G, Epmin=1e13 * u.eV, Epmax=3.2e18 * u.eV)

    specfrq = np.logspace(14, 50, 100) * u.Hz
    specen = specfrq.to(u.eV, equivalencies=u.spectral())
    dist = 1 * u.Mpc

    pds = [SYN1, SYN2, SYN3]
    ls = ['-', '--', '-.']
    colors = ['royalblue', 'red', 'black']
    labels = ['Exp.cutoff PWL','BrokenPL w/cutoff','PowerLaw']

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
    ax.set_title('Proton Synchrotron Model', fontsize=17)
    ax.set_ylim(1e-21, 1e-3)
    ax.set_xlim(1e0, 1e14)
    plt.legend(loc='best', borderpad=2, fontsize=14)
    plt.savefig('psync_model_comparison.eps')
    plt.show()
