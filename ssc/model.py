from __future__ import division, print_function
import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import Distance
import yaml


__all__ = ['blob', 'power_law_eval', 'exponential_cutoff_power_law_eval', 'gauss_eval',
            'BaseModel', 'BaseElectron']


e = const.e.gauss
m_e = const.m_e.cgs
c = const.c.cgs
mec = (const.m_e * const.c).cgs
mec2 = (const.m_e * const.c**2).cgs


class blob(object):
    """Class defining the emitting region attributes

    Literature used for this code:
    1) MNRAS 306, 551-560 (1999)
       SAO/NASA ads : http://adsabs.harvard.edu/abs/1999MNRAS.306..551C

    Parameters
    ----------
    R : `~astropy.units.Quantity`
        radius of the emitting region (blob)
    B : `~astropy.units.Quantity`
        magnetic field tangled with the emitting region
    t_esc : float
        escape time in units of light crossing time t_cross = R/c
    delta : float
        Doppler facrot of the source
    z : float
        redshift of the source
    """
    def __init__(self, R, B, t_esc, delta, z):
        """Initialize here all the quantities useful for the blob
        """
        # distance / volume units related to the blob
        self.R = R.to('cm') # dimenison of the emitting blob, in cgs
        self.volume = 4/3 * np.pi * self.R**3
        self.light_crossing_time = (self.R / c).to('s') # all the times in units of crossing time
        self.delta = delta
        self.z = z
        self.distance = Distance(z = self.z)
        self.t_esc = t_esc
        # magnetic units
        self.B = B.to('G')
        self.U_B = self.B**2/(8.*np.pi) # magnetic field density


def power_law_eval(amplitude, gamma, index):
    """power-law in Lorentz Factor of the injected electrons"""
    return amplitude * gamma**(-index)

def exponential_cutoff_power_law_eval(amplitude, gamma, gamma_c, index):
    """power-law exponentially cut-off in Lorentz Factor of the injected electrons"""
    return amplitude * gamma**(-index) * np.exp(-gamma/gamma_c)

def gauss_eval(amplitude, mean, sigma):
    """gaussian distribution of injected electrons"""
    _norm = 1 / np.sqrt(2 * np.pi * sigma**2)
    return amplitude * _norm * np.exp( - (gamma - mu)**2 /  2 * sigma**2)


class BaseModel(object):
    """Class for defining the source and the electron population

    The input is taken from a yaml file:
    4 dictionaries are defined in it:

    - time_grid, with keys:
        * time_min : float
            minimum time in units of R/c
        * time_max : float
            maximum time in units of R/c
        * time_bins : float
            number of time bins on which to perform the temporal evolution
        * time_inj : float
            maximum time of injection of the electrons, units of R/c

    - gamma_grid, with keys:
        * log_gamma_min : float
            log10 of the minimum Lorentz factor of the electron distribution
        * log_gamma_max
            log10 of the maximum Lorentz factor of the electron distribution
        * gamma_bins
            number of bins in the grid of Lorentz factor

    - emission_region, with keys:
        * log_R : float
            log10 of the size (radius) of the emitting region
        * R_unit : string
            units of the emeitting region ('cm' favourably)
        * log_B : float
            log10 of the value of the magnetic field tangled with the emitting region
        * B_unit : string
            unit of the magnetic field ('G' favourably)
        * t_esc : float
            escape time for the electrons from the emitting region
        * delta : float
            doppler factor of the jet (along which the blob is moving)
        * z : float
            redshift of the source

    - injected_spectrum (e.g PowerLaw)
        * type : string
            distribution of the electrons injected ('PowerLaw' as example)
        * log_Norm : float
            log10 of the normalization of the injected distribution
        * Norm_units : string
            unit of the normalization ('cm-3' favourably)
        * index : float
            spectral index of the injected distribution (in PowerLaw case)

    Parameters
    ----------
    yaml: string
        path to the .yaml file with the configuration
    """

    def __init__(self, yaml_file):
        # read the input file
        with open(yaml_file, 'r') as ymlfile:
                self.cfg = yaml.load(ymlfile)

        # emission regions parameters definition, blob object creation
        self.blob = blob(
            R = 10**(self.cfg['emission_region']['log_R']) * u.Unit(self.cfg['emission_region']['R_unit']),
            B = 10**(self.cfg['emission_region']['log_B']) * u.Unit(self.cfg['emission_region']['B_unit']),
            t_esc = self.cfg['emission_region']['t_esc'],
            delta =  self.cfg['emission_region']['delta'],
            z = self.cfg['emission_region']['z']
            )

        # time grid, everything in unit of R/c
        self.time_min = self.cfg['time_grid']['time_min']
        self.time_max = self.cfg['time_grid']['time_max']
        self.time_bins = self.cfg['time_grid']['time_bins']
        self.time_grid = np.linspace(self.time_min, self.time_max, self.time_bins)
        # the time grid is equally spaced
        self.delta_t = self.time_grid[2] - self.time_grid[1]
        self.time_inj = self.cfg['time_grid']['time_inj']

        # gamma grid parameters
        self.log_gamma_min = self.cfg['gamma_grid']['log_gamma_min']
        self.log_gamma_max = self.cfg['gamma_grid']['log_gamma_max']
        self.gamma_min = 10**self.log_gamma_min
        self.gamma_max = 10**self.log_gamma_max
        self.gamma_bins = self.cfg['gamma_grid']['gamma_bins']
        # note the in Ref. (1) a grid spanning from N_{-1} to N_{n+1}
        # is considered and then the electron density is evaluated always
        # at the midpoints
        self.gamma_ext = np.logspace(self.log_gamma_min, self.log_gamma_max, self.gamma_bins)
        # we define the midpoints through the extended {-1,n+1} grid
        # in order to have N_{-1/2} N_{n+1/2}
        self.gamma_midpts = np.sqrt(self.gamma_ext[:-1]*self.gamma_ext[1:])
        self.gamma = self.gamma_ext[1:-1]
        self.delta_gamma = self.gamma_midpts[1:] - self.gamma_midpts[:-1]

        # for the moment let's take the parameters of the injected spectrum as a
        # dictionary
        self.inj_spec_dict = self.cfg['injected_spectrum']


    def N_e_inj(self, gamma):
        """particle distribution of injected electrons

        Parameters
        ----------
        gamma : `~numpy.ndarray`
            array of gamma where to evaluate the injected distribution
            it can be either self.gamma or self.gamma_midpts

        Returns
        -------
        `~astropy.quantity.Quantity`
            density [cm-3] of the injected spectrum
        """
        if self.inj_spec_dict['type'] == 'PowerLaw':
            amplitude = 10**(self.inj_spec_dict['log_Norm']) * u.Unit(self.inj_spec_dict['Norm_unit'])
            index = self.inj_spec_dict['index']
            # evaluate the densities
            pwl = power_law_eval(amplitude, gamma, index)
            return BaseElectron(gamma, pwl)

        if self.inj_spec_dict['type'] == 'ExponentialCutoffPowerLaw':
            amplitude = 10**(self.inj_spec_dict['log_Norm']) * u.Unit(self.inj_spec_dict['Norm_unit'])
            index = self.inj_spec_dict['index']
            gamma_c = 10**self.inj_spec_dict['log_gamma_c']
            # evaluate the densities
            pwl = exponential_cutoff_power_law_eval(amplitude, gamma, gamma_c, index)
            return BaseElectron(gamma, pwl)



class BaseElectron(object):
    """Class describing a basic electron population

    For now it will consist only of
    - density [cm-3]
    - Lorentz factor values

    Class introudced to avoid to pass separately densities and gammas separately
    in functions for spectra evaluation

    Parameters:
    -----------
    gamma : `~numpy.ndarray`
        array of gamma where the density values are defined
    density : `~astropy.quantity.Quantity`
        density value [cm-3] at each gamma point
    """
    def __init__(self, gamma, density):
        self.gamma = gamma
        self.density = density
        if len(gamma) != len(density):
            print('x should have the same dimenison of y')
