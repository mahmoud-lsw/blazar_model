#######################################################################
# Class to calculate EM cascade Raditation

# References:
# 1) http://iopscience.iop.org/article/10.1088/0004-637X/768/1/54/pdf

# 13 July 2017
# Author(s) : Wrijupan Bhattacharyya (wrijupan.bhattacharyya@desy.de)
########################################################################

# NOTE : This code is incomplete now

from optical_depth import CalcOptDepthBB
from trial_pgamma_model import PionDecay_gamma, \
    PionDecay_positron, PionDecay_electron
import numpy as np
from numpy import pi, exp
from astropy import units as u
from astropy.constants import m_e, c, sigma_T
from naima.models import PowerLaw as PL
from naima.models import Synchrotron, TableModel
from scipy.special import gamma
from scipy.integrate import quad
import matplotlib.pyplot as plt
import timeit

_mec2_u = (m_e * c ** 2).to('eV')
_c = c.cgs
_sigmaT = sigma_T.cgs
_Bcrit = 4.4e13 * u.G

class EMCascade(object):
    def __init__(self, particle_dist, B, R = 1 * u.kpc, T = 2.7 * u.K):
        self.B = B.to('G')
        self.R = R.to('cm')
        self.T = T.to('K')
        self.pdist = particle_dist

    def Ngam_dot(self, Eph):
        return PionDecay_gamma(pdist)._calc_spec_pgamma(Eph)

    def Qe(self, Epair):
        return PionDecay_positron(pdist)._calc_spec_pgamma(Epair)

    def attenuation(self, Eph):
        Eph = Eph.to('eV') / _mec2_u
        calcdepth = CalcOptDepthBB(self.T, self.R)
        tau = calcdepth.calc_opt_depth(Eph)
        return (1 - exp(- tau)) / tau

    def sync_integrand(self, Epair, Eph):
        gam = Epair / _mec2_u.value
        eps = Eph / _mec2_u
        b = self.B / _Bcrit
        emissivity = (gam ** - (2./3)) * exp(- eps / (b * gam**2))
        return _mec2_u.value * self.Ne(Epair * u.eV) * emissivity

    def Nsync_dot(self, Eph, Eemin, Eemax):
        eps = Eph.to('eV') / _mec2_u
        b = self.B / _Bcrit
        num = _c * _sigmaT * self.B ** 2
        denom = 6 * pi * _mec2_u * gamma(4./3) * b**(4./3)
        A0 = num / denom
        norm = A0.value * (eps ** (-2./3))

        Eemin = Eemin.to('eV').value
        Eemax = Eemax.to('eV').value
        return norm * quad(self.sync_integrand, Eemin, Eemax, args=Eph)[0]

    def Ne(self, Epair):
        return self.Qe(Epair)

if __name__ == '__main__':
    ##### General ########
    start = timeit.default_timer()
    E_test = 1e16 * u.eV
    _E = 3.0e20 * u.eV
    pdist = PL(4.3e8 / u.eV, 1e3 * u.GeV, 2.5)
    emc = EMCascade(pdist, 1e-3 * u.G, 1 * u.kpc, 2.7 * u.K)

    #### some function tests #####
    #print(emc.Ngam_dot(E_test), emc.Qe(E_test),
    #      emc.attenuation(E_test), pdist(E_test))

    #### Synchrotron Calculation test ####
    print("Calculating synchrotron flux ...")
    Egam = np.logspace(12, 22, 20) * u.eV
    syncflux = []
    for E in Egam:
        syncflux.append(emc.Nsync_dot(E, 0.43e-2*_E, _E))
        print(syncflux[-1])
    stop = timeit.default_timer()
    print("Elapsed time for computation = {} secs".format(stop - start))
    plt.loglog(Egam, syncflux)
    plt.savefig('cascade.png')



