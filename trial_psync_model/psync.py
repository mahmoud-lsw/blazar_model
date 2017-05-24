from naima.models import Synchrotron
from naima.models import ExponentialCutoffPowerLaw as ECPL
from astropy.constants import m_e, m_p, c, e, h, hbar
from astropy import units as u
from naima.radiative import _validate_ene
from naima.extern.validator import validate_scalar
from naima.utils import trapz_loglog
import numpy as np
from scipy.special import cbrt
import matplotlib.pyplot as plt


mpc2 = (m_p *c ** 2).cgs
mpc2_unit = u.Unit(mpc2)
e = e.gauss


class PSynchrotron(Synchrotron):
    """Dummy Proton-Synchrotron Model
       Simplest modification of Naima e-synchrotron code.
       Compute synchrotron for random magnetic field according to approximation
        of Aharonian, Kelner, and Prosekin 2010, PhysRev D 82, 3002
        (`arXiv:1006.1045 <http://arxiv.org/abs/1006.1045>`_).
    """

    def __init__(self, particle_distribution, **kwargs):
        self.particle_distribution = particle_distribution
        P = self.particle_distribution(1*u.TeV)
        validate_scalar('particle distribution', P, physical_type='differential energy')
        self.Eemin = 1e15 * u.eV
        self.Eemax = 4e21 * u.eV
        self.nEed = 200
        self.__dict__.update(**kwargs)

    @property
    def _gam(self):
        """Lorentz factor array of protons"""
        log10gmin = np.log10(self.Eemin / mpc2).value
        log10gmax = np.log10(self.Eemax / mpc2).value
        return np.logspace(log10gmin, log10gmax, 
                           self.nEed * (log10gmax - log10gmin))

    @property
    def _nelec(self):
        """Particles per unit Lorentz factor"""
        pd = self.particle_distribution(self._gam * mpc2)
        return pd.to(1 / mpc2_unit).value


    def _spectrum(self, photon_energy):
        """ Calculate diffenential photon spectrum for given energies
        """
        outspecene = _validate_ene(photon_energy)
        
        def Gtilde(x):
            cb = cbrt(x)
            gt1 = 1.808 * cb / np.sqrt(1 + 3.4 * cb**2.)
            gt2 = 1 + 2.210 * cb**2. + 0.347 * cb**4.
            gt3 = 1 + 1.353 * cb**2. + 0.217 * cb**4.
            #print(gt1 * (gt2 / gt3) * np.exp(-x))
            return gt1 * (gt2 / gt3) * np.exp(-x)      

        CS1_0 = np.sqrt(3) * e.value**3 * self.B.to('G').value
        CS1_1 = (2 * np.pi * m_p.cgs.value * c.cgs.value
                 ** 2 * hbar.cgs.value * outspecene.to('erg').value)
        CS1 = CS1_0 / CS1_1
        #print(np.where(CS1==0)[0])
        
        #Critical energy calculation
        Ec = 3 * e.value * hbar.cgs.value * self.B.to('G').value * self._gam**2
        Ec /= 2 * (m_p * c).cgs.value
        #print(np.where(Ec==0)[0])

        EgEc = outspecene.to('erg').value / np.vstack(Ec)
        #print(np.where(np.exp(-EgEc)==0)[0])
        dNdE = CS1 * Gtilde(EgEc)
        #print(np.where(Gtilde(EgEc)==0)[0])

        spec = trapz_loglog(np.vstack(self._nelec) * dNdE, self._gam, axis=0) / u.s / u.erg
        spec = spec.to('1/(s eV)')
        #print("_OUTPUT FROM NELEC FUNC...\n", self._nelec)
        #print("_OUTPUT FROM DNDE...\n", dNdE)
        #print(np.vstack(self._nelec) * dNdE)
        #print(self._nelec.shape)
        #print(dNdE.shape)

        return spec

if __name__ == '__main__':
   
    #pdist = ECPL(4.3e30/u.eV,10*u.TeV,2.34, 30*u.TeV)
    pdist = ECPL(4.3e33/u.eV,1e3*u.GeV,1.3, 8e5*u.TeV)
   
    #SYN = Synchrotron(ECPL, B=10 * u.G, Eemin=1 * u.TeV, Eemax=1e19 * u.eV)
    SYN = PSynchrotron(pdist, B=10 * u.G)
    #print(SYN._gam)
    specf = np.logspace(14, 28, 100)*u.Hz
    spece = specf.to(u.eV, equivalencies=u.spectral())
    dist = 1 * u.Mpc
    sed_SYN = SYN.sed(spece, distance = dist)
    #print(sed_SYN)
    #print(SYN._nelec)

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(1,1,1)
    font = {'family':'serif', 'color':'black', 'weight':'normal', 'size':16.0}
    ax.plot(spece, SYN.sed(spece, dist), lw=3, color='royalblue', label='PSynchrotron')
    ax.set_xlabel('Energy (eV)', fontsize = 14)
    ax.set_ylabel(r'$E^{2}*{\rm d}N/{\rm d}E\,[erg\,cm^{-2}\,s^{-1}]$', fontsize=17)
    ax.set_title('Proton Synchrotron Model', fontsize = 17)
    ax.set_ylim(1e-21, 1e-3)
    ax.set_xlim(1e0, 1e14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.savefig('psync_ECPL.eps')
    plt.show()
