import numpy as np
import astropy.units as u
import astropy.constants as const
from ssc.model import BaseModel
from ssc.radiative import Synchrotron, InverseCompton, numerics
import matplotlib.pyplot as plt

# test model
model = BaseModel('config.yaml')

# define the base electron population for now just as the injected one
gamma = model.gamma
N_e = model.N_e_inj(gamma)

# test synchrotron
syn = Synchrotron(model)
ic = InverseCompton(model)

obs_nu = np.logspace(7, 28, 100) * u.Hz
# de - boosting, for intrinsic values
nu = obs_nu / model.blob.delta

# plot the SED
# now we compare it with naima's synchrotron
E = const.h * obs_nu

syn_flux = syn.flux(nu, N_e, self_absorption=False)
syn_flux_abs = syn.flux(nu, N_e, self_absorption=True)

ic_flux = ic.flux(nu, N_e, ebl_absorption=False)
ic_flux_abs = ic.flux(nu, N_e, ebl_absorption=True)

fig = plt.figure()
plt.loglog(obs_nu, (E**2*syn_flux_abs).to('erg cm-2 s-1'), lw=2, label='synchr. self absorption')
plt.loglog(obs_nu, (E**2*ic_flux_abs).to('erg cm-2 s-1'), lw=2, label='inverse compton + EBL')
plt.loglog(obs_nu, (E**2*syn_flux).to('erg cm-2 s-1'), lw=2, ls='--', label='synchrotron')
plt.loglog(obs_nu, (E**2*ic_flux).to('erg cm-2 s-1'), lw=2, ls='--',label='inverse compton')

plt.legend()
plt.xlabel(r'$\nu$ [Hz]', size=12)
plt.ylabel(r'E$^2$ d$\phi$/dE [erg cm$^{-2}$ s$^{-1}$]', size=12)
plt.ylim([1e-24, 1e-8])
plt.show()
fig.savefig('ecpl_simple_ssc_trial.png')


# test the evolution part
#num = numerics(nu, model)
# we need U_rad to get a Chang and Cooper matrix
#N_e_evol = num.evolve()
#plt.loglog(model.gamma, N_e_evol[0].density)
#plt.loglog(model.gamma, N_e_evol[-1].density)
#plt.show()
