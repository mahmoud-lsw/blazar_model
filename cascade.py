#######################################################################
# Class to calculate EM cascade Radiation

# References:
# 1) http://iopscience.iop.org/article/10.1088/0004-637X/768/1/54/pdf

# 25 July 2017
# Author(s) : Wrijupan Bhattacharyya (wrijupan.bhattacharyya@desy.de)
########################################################################
from optical_depth import CalcOptDepthBB, CalcOptDepthPWL
from trial_pgamma_model import PionDecay_gamma, \
    PionDecay_positron, PionDecay_electron
import numpy as np
from astropy import units as u
from astropy.constants import m_e, e, c, sigma_T, hbar
from naima.models import PowerLaw as PL
from naima.utils import trapz_loglog
from scipy.special import gamma, cbrt
import matplotlib.pyplot as plt
import timeit

_mec2_u = (m_e * c ** 2).to('eV')
_c = c.cgs
_sigmaT = sigma_T.cgs
_Bcrit = 4.4e13 * u.G
_norm = 1.5e21 * u.Unit('eV-3 cm-3')

class EMCascade(object):
    def __init__(self, particle_dist, gammamin, gammamax, gammagridsize,
                 B, R = 1 * u.kpc, T = 2.7 * u.K, eta=3):
        self.B = B.to('G')
        self.R = R.to('cm')
        self.T = T.to('K')
        self.pdist = particle_dist
        self.gmin = gammamin
        self.gmax = gammamax
        self.bins = gammagridsize
        self.eta = eta

    def Ngam_dot(self, Eph):
        norm = 1.5e21 * u.Unit('erg-3 cm-3')
        norm = norm.to('eV-3 cm-3')
        return PionDecay_gamma(self.pdist, T=1e4 * u.K,
                               norm=norm)._calc_spec_pgamma(Eph)

    def Qe(self, Epair):
        """
        Parameters
        ----------
        Epair : astropy Quantity (float)
                only one energy of the pairs
        Returns
        --------
        Only one particle injection flux at energy = Epair
        """
        norm = 1.5e21 * u.Unit('erg-3 cm-3')
        norm = norm.to('eV-3 cm-3')
        return PionDecay_positron(self.pdist,T=1e4 * u.K,
                                  norm=norm)._calc_spec_pgamma(Epair) + \
               PionDecay_electron(self.pdist,T=1e4 * u.K,
                                  norm=norm)._calc_spec_pgamma(Epair)

    def Ne(self, Epair):
        """
         Parameters
         ----------
         Epair : astropy Quantity (array_like)
                 array of pair energies
        Returns
        -------
        array of particle distribution for Epair array
        """
        norm = 1.5e21 * u.Unit('erg-3 cm-3')
        norm = norm.to('eV-3 cm-3')

        return (PionDecay_positron(self.pdist,T=1e4 * u.K,
                                   norm=norm)._spectrum(Epair) +
                PionDecay_electron(self.pdist,T=1e4 * u.K,
                                   norm=norm)._spectrum(Epair)) * \
               (self.eta * (self.R / _c)).value

    def attenuation_BB(self, E):
        """
        This method is not used at present.
        Optical depth calculated assuming
        a BPL target. See below
        """
        calcdepth = CalcOptDepthBB(self.R, 2.7 * u.K, 1.5e62 * u.Unit('cm-3 eV-3'))
        E = E.to('eV') / _mec2_u
        tau = calcdepth.calc_opt_depth(E)
        return (1 - np.exp(- tau)) / tau, tau

    def tot_attenuation(self, Eph):
        """
        Allows for array operation
        """
        expTau = []
        Tau = []
        for E in Eph:
            tau = self.attenuation(E)
            expTau.append(tau[0])
            Tau.append(tau[1])
        return expTau, Tau

    def attenuation(self, E):
        taupl = CalcOptDepthPWL()
        norm = 1e-10 * u.Unit('erg cm-2 s-1')
        norm = norm.to('eV cm-2 s-1')
        tau = float(taupl.tau_YY(E, 1, 1e-5, norm, 1e5 * u.s, 10, 2, -2))
        return (1 - np.exp(- tau)) / tau, tau


    def Nsync_dot(self, Eph, Ne, Eemin, Eemax):
        """
        Synchrotron emissivity calulated according to Reference.

        With this emissivity, the synch spectrum cuts of at higher
        energies compared to Aharonian's emissivity (synch cuts off
        ~3 orders of magnitude below Eemax whereas with Aharonian's
        emissivity, the synch spectrum cuts off ~9 orders of mag below).
        """
        gmin = Eemin.to('eV') / _mec2_u
        gmax = Eemax.to('eV') / _mec2_u

        def gam_arr():
            log10gmin = np.log10(gmin)
            log10gmax = np.log10(gmax)
            return np.logspace(log10gmin, log10gmax, len(Ne))

        eps = Eph.to('eV') / _mec2_u
        b = self.B / _Bcrit
        emissivity = (gam_arr() ** - (2./3)) * np.exp(- eps / (b * gam_arr()**2))

        num = _c * _sigmaT * self.B ** 2
        denom = 6 * np.pi * _mec2_u * gamma(4. / 3) * b ** (4. / 3)
        A0 = num / denom
        norm = A0.value * (eps ** (-2. / 3))

        spec = norm * trapz_loglog(_mec2_u * Ne * emissivity, gam_arr(), axis=0)
        return spec.value

    def Nsync_dot_ah(self, Eph, Ne, Eemin, Eemax):
        """
        Synchrotron emissivity according to Aharonian, Kelner, and Prosekin
        2010, PhysRev D 82, 3002
        (`arXiv:1006.1045 <http://arxiv.org/abs/1006.1045>`_)

        If this emissivity is used the synchrotron spectrum cuts off at
        much lower energies (~ 9-10 orders of magnitude below elec. energy).
        """
        Eph = Eph.to('eV')
        gmin = Eemin.to('eV') / _mec2_u
        gmax = Eemax.to('eV') / _mec2_u

        def gam_arr():
            log10gmin = np.log10(gmin)
            log10gmax = np.log10(gmax)
            return np.logspace(log10gmin, log10gmax, len(Ne))

        def Gtilde(x):
            cb = x ** (1./3)
            gt1 = 1.808 * cb / np.sqrt(1 + 3.4 * cb ** 2.)
            gt2 = 1 + 2.210 * cb ** 2. + 0.347 * cb ** 4.
            gt3 = 1 + 1.353 * cb ** 2. + 0.217 * cb ** 4.
            return gt1 * (gt2 / gt3) * np.exp(-x)

        CS1_0 = np.sqrt(3) * e.value ** 3 * self.B.to('G').value
        CS1_1 = (2 * np.pi * m_e.cgs.value * c.cgs.value
                 ** 2 * hbar.cgs.value * Eph.to('erg').value)
        CS1 = CS1_0 / CS1_1

        Ec = 3 * e.value * hbar.cgs.value * \
             self.B.to('G').value * gam_arr() ** 2
        Ec /= 2 * (m_e * c).cgs.value

        EgEc = Eph.to('erg').value / Ec
        dNdE = CS1 * Gtilde(EgEc)

        spec = trapz_loglog(Ne * dNdE, gam_arr(), axis=0) /u.s /u.erg
        spec = (spec.to('s-1 eV-1')).value
        print(spec)
        return spec


    def Ngg_dot(self, Ee, Eemin, Eemax, Ne):
        eps1 = Ee / 0.9
        eps2 = Ee / 0.1

        def fabs(eps):
            return 1 - self.attenuation(eps)[0]

        N01_dot = self.Ngam_dot(eps1)
        N02_dot = self.Ngam_dot(eps2)
        Nsync1_dot = self.Nsync_dot(eps1, Ne, Eemin, Eemax)
        Nsync2_dot = self.Nsync_dot(eps2, Ne, Eemin, Eemax)

        term1 = fabs(eps1) * (N01_dot + Nsync1_dot)
        term2 = fabs(eps2) * (N02_dot + Nsync2_dot)
        return term1 + term2


    def gamma_grid(self, gammin, gammax, gambins):
        gamma_grid = []
        for i in range(-1, gambins + 2):
            gamma_grid.append(gammin * (gammax / gammin) ** ((i - 1) / (gambins - 1)))
        gamma_grid_ext = np.array(gamma_grid)
        gamma_grid = gamma_grid_ext[1:-1]
        energy_grid = gamma_grid * _mec2_u

        gamma_mid = (gamma_grid_ext[1:] * gamma_grid_ext[:-1]) ** 0.5
        delta_gamma = gamma_mid[1:] - gamma_mid[:-1]
        return gamma_grid, energy_grid, gamma_mid, delta_gamma

    def tridiag_mat(self, Ne, gammin, gammax, gambins):
        N = len(self.gamma_grid(gammin, gammax, gambins)[3])
        matrix_lhs = np.zeros((N, N), float)
        matrix_rhs = np.zeros(shape=N)
        B = self.B

        def cool_rate(gamm):
            num = _c * _sigmaT * B **2
            denom = 6 * np.pi * _mec2_u
            return (-(num / denom) * (gamm ** 2)).value

        for i in range(N-1, -1, -1):
            gam, ene, gam_mid, del_gam = self.gamma_grid(gammin, gammax, gambins)
            gamma_minus_half = gam_mid[i]
            gamma_plus_half = gamma_minus_half + del_gam[i]
            tesc = (self.eta * (self.R / _c)).value

            V3 = cool_rate(gamma_plus_half) / del_gam[i]
            V2 = (1 / tesc) - (cool_rate(gamma_minus_half) / del_gam[i])
            for j in range(N):
                if j == i :
                    matrix_lhs[i, j] = V2
                elif j == i + 1 :
                    matrix_lhs[i, j] = V3

            matrix_rhs[i] = self.Qe(ene[i]) + self.Ngg_dot(ene[i],
                                                           ene[-1], 8e17 * u.eV, Ne=Ne)
        return matrix_lhs, matrix_rhs

    def Ne_soln(self, Ne, gammin, gammax, gambins):
        lhs, rhs = self.tridiag_mat(Ne, gammin, gammax, gambins)
        Ne_new = np.linalg.solve(lhs, rhs)
        print("\n LHS =", lhs)
        print("\n RHS =", rhs)
        print("\n Solution = {} \n".format(Ne_new))
        return Ne_new

    @property
    def elec_spectrum(self):
        gam, ene, gam_mid, del_gam = self.gamma_grid(self.gmin, self.gmax, self.bins)
        splitted_gam = np.array_split(gam, len(gam) / 6)
        print("\n CALCULATING INITIAL ELECTRON INJECTION ...")
        Ne = self.Ne(splitted_gam[-1] * _mec2_u)
        iter = 1

        for gam in reversed(splitted_gam):
            print("\n CALCULATING ELECTRON DISTRIBUTION FOR CASCADE GENERATION {}".format(iter))
            print("\n Starting Electron distribution : {}".format(Ne))
            gammin, gammax, gambins = gam[0], gam[-1], len(gam)
            Ne_new = self.Ne_soln(Ne, gammin, gammax, gambins)
            Ne = Ne_new
            iter += 1
        print("\n FINAL ELECTRON DISTRIBUTION AFTER N GENERATIONS : {}".format(Ne))
        return Ne

    @property
    def photon_spectrum(self):
        Ne = self.elec_spectrum
        Eph = np.linspace(1e9 * u.eV, 1e12 * u.eV, 1e2)

        # IF Aharonian's parametrization for emissivity is used
        #the synchrotron spectrum cuts off at much lower energies
        # Eph = np.linspace(2 * u.eV, 1e5 * u.eV, 100)

        photon_flux = []
        for E in Eph:
            # Assumption : Only the high energy e- contributes to Synchrotron
            photon_flux.append(self.Nsync_dot(E, Ne, 8e12 * u.eV, 1e17 * u.eV))
        photon_flux = np.array(photon_flux)
        return Eph, photon_flux

    @property
    def esc_photon_spectrum(self):
        Eph, photon_flux = self.photon_spectrum
        exptau, tau = self.tot_attenuation(Eph)
        return Eph, tau, photon_flux, photon_flux * exptau

    @property
    def plot_photon_spectrum(self):
        #Eph, Nph = self.photon_spectrum
        Eph, tau, Nph, esc_Nph = self.esc_photon_spectrum
        esc_Nph_sed = ((Eph ** 2) * esc_Nph).value
        Nph_sed = ((Eph ** 2) * Nph).value

        fig, ax1 = plt.subplots()
        ax1.loglog(Eph.to('eV'), esc_Nph_sed, 'b-', label='escaping photons')
        ax1.loglog(Eph.to('eV'), Nph_sed, 'r-', label='last gen photons before esc')
        ax1.set_ylabel(r'$E^{2}$ $\times$ dN/dE [eV $s^{-1}]$')
        ax1.set_xlabel('Energy [eV]')
        ax1.set_title("Cascade spectrum")
        fig.tight_layout()
        plt.legend(loc='best')
        plt.savefig("./cascade_sync_spec.png")
        plt.show()


if __name__ == '__main__':
    start = timeit.default_timer()
    pdist = PL(4.3e8 / u.eV, 1e3 * u.GeV, 2.5)
    emc = EMCascade(pdist, 1e8, 3e10, 11, 1e-4 * u.G, 1 * u.kpc, 1e4 * u.K)

    emc.plot_photon_spectrum
    stop = timeit.default_timer()
    print("Elapsed time for computation = {} secs".format(stop - start))








