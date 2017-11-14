################################################################################
# this is a class to use the SSC as a sherpa model
################################################################################

import numpy as np
import astropy.units as u
import astropy.constants as const
from ssc.model import BaseModel
from ssc.radiative import Synchrotron, InverseCompton
import sherpa.models as sherpa_model
import yaml

__all__ = ['_ssc', 'SSC']

def _ssc(pars, nu):
	"""utility functions for building the SSC using the functions in ssc
	for now we consider a static version of the ssc"""
	(log10_gamma_max, redshift, delta, log10_R, log10_B, log10_Norm, index, log10_gamma_c) = pars

	# define from the input parameters the dictionary to be feeded to the model
	# we neglect the time-dependent part for now
	time_grid = dict(time_min=0, time_max=3, time_bins=50, time_inj=2)
	# gamma grid
	gamma_grid = dict(log10_gamma_min=2, log10_gamma_max=log10_gamma_max, gamma_bins=50)
	# emission region, again time dependent part is ignored
	emission_region = dict(log10_R=log10_R, R_unit='cm', delta=delta,
						   log10_B=log10_B, B_unit='G', t_esc=1.5, z=redshift)
	# injected spectrum
	injected_spectrum = dict(type='ExponentialCutoffPowerLaw',
							 log10_Norm=log10_Norm,
							 Norm_unit='cm-3',
							 index=index,
							 log10_gamma_c=log10_gamma_c)

	# dump into a tmp yaml file
	with open('tmp_config.yaml', 'w') as yaml_file:
		yaml.dump({'time_grid': time_grid,
				   'gamma_grid': gamma_grid,
				   'emission_region': emission_region,
				   'injected_spectrum': injected_spectrum},
				   yaml_file, default_flow_style=False)


	# initialize the ssc model
	model = BaseModel('tmp_config.yaml')

	# define the base electron population for now just as the injected one
	gamma = model.gamma
	N_e = model.N_e_inj(gamma)

	# test synchrotron
	syn = Synchrotron(model)
	ic = InverseCompton(model)

	obs_nu = nu * u.Hz
	# de - boosting, for intrinsic values
	nu = obs_nu / model.blob.delta

	# transform to energy
	E = const.h * obs_nu

	syn_flux = syn.flux(nu, N_e, self_absorption=True)
	ic_flux = ic.flux(nu, N_e, ebl_absorption=True)

	sed = (E**2*(syn_flux + ic_flux)).to('erg cm-2 s-1')

	return sed.value

class SSC(sherpa_model.model.ArithmeticModel):
	"""class for implementing the SSC sed
	"""
	def __init__(self, name='SSC'):
		self.log10_gamma_max = sherpa_model.model.Parameter(name, 'log10_gamma_max', 6, min=4, max=7)
		self.redshift = sherpa_model.model.Parameter(name, 'redshift', 0.4, min=0, max=2)
		self.delta = sherpa_model.model.Parameter(name, 'doppler', 10, min=3, max=40)
		self.log10_R = sherpa_model.model.Parameter(name, 'log10_R', 16, min=13, max=20)
		self.log10_B = sherpa_model.model.Parameter(name, 'log10_B', -1, min=-6, max=2)
		self.log10_norm = sherpa_model.Parameter(name, 'log10_Norm', -1, min=-2, max=6)
		self.index = sherpa_model.model.Parameter(name, 'index', 3, min=1, max=5)
		self.log10_gamma_c = sherpa_model.model.Parameter(name, 'log10_gamma_c', 3, min=2, max=6)

		sherpa_model.model.ArithmeticModel.__init__(self, name,
                                       (self.log10_gamma_max,
									    self.redshift,
										self.delta,
									    self.log10_R,
										self.log10_B,
									   	self.log10_Norm,
										self.index,
										self.log10_gamma_c
										))

	def calc(self, pars, nu, *args, **kwargs):
		"""evaluate the model"""
		return _ssc(pars, nu)
