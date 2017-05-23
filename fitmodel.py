import numpy as np
from math import pi
import matplotlib.pyplot as plt
from ssc_model.model import model as mod
from ssc_model.numerics import numerics
import astropy.units as u
from astropy.io import ascii
import naima
import argparse
#Usage: python toymodel.py -z 0.047 -t 45. -d 1. -x data_table_xray.dat -v data_table_gray.dat -2.0

__all__ = ['Fitmodel','fitter'] 

class Fitmodel:
    ''' DESCIPTION: Class to derive the best fit and uncertainty distribution of the free spectral parameters
        through Markov Chain Monte Carlo sampling of their likelihood distribution.
 
        The number of free parameters (taken as command line input) are only 5 here for simplicity. 
        Format for data file provided from command line : Astropy Table (the way Naima likes it)
 
        usage: fitmodel2.py [-h] [-z --REDSHIFT] [-t --THETA] [-d --LORENTZ]
                    [-x [--XRAY-DATA]] [-v [--VHE-DATA]]
                    FREE_PARAMS [FREE_PARAMS ...]
        find the max. likelihood model to fit to data

        positional arguments:
        FREE_PARAMS       R(cm) B(G) norm index gamma_max

        optional arguments:
        -h, --help        show this help message and exit
        -z --REDSHIFT
        -t --THETA
        -d --LORENTZ
        -x [--XRAY-DATA]
        -v [--VHE-DATA]
    '''

    def __init__(self):
        parser = argparse.ArgumentParser(description='find the max. likelihood model to fit to data')
        parser.add_argument('-z', metavar='--REDSHIFT',type=float)
        parser.add_argument('-t', metavar='--THETA',type=float)     #in degrees
        parser.add_argument('-d', metavar='--LORENTZ',type=float)   #Bulk Lorentz factor
        parser.add_argument('-x', metavar='--XRAY-DATA', nargs='?',type=str, default='data_table_xray.dat')
        parser.add_argument('-v', metavar='--VHE-DATA', nargs='?',type=str, default='data_table_gray.dat')
        parser.add_argument('free', metavar='FREE_PARAMS', nargs='+', type=float, help="R(cm) B(G) norm index gamma_max")    
        args = parser.parse_args() 
        self.xray = args.x
        self.vhe = args.v
        self.z = args.z
        self.theta = args.t
        self.lorentz = args.d
        self.dat_type = 'i'     #Hard-coded DEFAULT='i'. 'i'=intrinsic data; 'o'=observed
                                ##set self.dat_type='o' for observed data. Not done at present because
                                #doppler boosting implementation needs further investigation.
        self.p0 = args.free    
        #When considering Doppler boosting 2 more free params (not done at present)
        if self.dat_type == 'o':
           self.p0.extend([args.t, args.d])                  
        print('Provided x-ray astropy table = "{}" & gamma-ray astropy table = "{}"'.format(self.xray, self.vhe))

        
    def model_func(self, pars, data):
        '''
        The model function will be called during fitting to compare with obsrvations. 

        Parameters:
        ------------
        pars: list
              list of free parameters of the model
        data: astropy_table
              observational data. Multiple tables can also be passed to 'data'

        Returns: 
        ---------
        Flux model to compare with observations.
        '''
    
        #free parameters for emission region
        #R = int(pars[0]) * u.cm 
        R = 1e16 * u.cm
        #B = pars[1] * u.G
        #B = 2*naima.estimate_B(soft_xray, vhe).to('G')
        B = 1.0 * u.G
        emission_region = dict(R = R.value, B = B.value, t_esc = 1.5)

        #free parameters for the particle spectral distribution
        #norm = pars[0] * u.Unit('1/erg')
        norm = 1e+0 * u.Unit('1/erg')
        index = pars[0]
        injected_spectrum = dict(norm = norm.value, alpha = index, t_inj = 1.5)
        distance = 2.0*u.Mpc

        #free parameters for the gamma grid
        #gamma_max = pars[3]
        gamma_max = 2e5
        gamma_grid = dict(gamma_min = 2., gamma_max = gamma_max, gamma_bins = 20)

        #Fixed parameters
        time_grid = dict(time_min = 0., time_max = 3., time_bins = 50)
      
        #with the above parameter set, we now obtain a particle distibution from numerics class
        SSC = mod(time_grid, gamma_grid, emission_region, injected_spectrum)
        num = numerics(SSC)
        N_e = num.evolve()

        #Obtain the Syn and IC instances by feeding in the particle distribution model
        SYN = num.synchrotron(N_e)
        IC = num.inverse_compton(N_e)

        #create the flux model to be given as input to run_sampler 
        if self.dat_type == 'i':
            energy = np.logspace(-7, 13, 25) * u.eV
            #model_flux = (IC.flux(energy, distance) + SYN.flux(energy, distance))
            model_flux = (IC.sed(energy, distance=distance) + SYN.sed(energy, distance=distance))
        elif self.dat_type == 'o':
            beta=np.sqrt(1.-1./(self.theta**2))
            doppler = 1./(self.lorentz*(1.-beta*np.cos(self.theta)))
            #energy = (np.logspace(-7, 13, 25) * doppler)* u.eV
            energy = np.logspace(-7, 13, 25) * u.eV
            model_flux = (IC.flux(energy, distance) + SYN.flux(energy, distance)) * (doppler**3.5)
        return model_flux


    def ebl():
        '''
        This function is not used at present. 
        Only makes sense to use after implementing Doopler boosting.
      
        Returns:
        --------
        opacity values which can be multiplied to a flux model
        '''
        z = self.z
        opacity = naima.models.EblAbsorptionModel(redshift=z)
        return opacity

    def prior_func(self, pars):
        '''
        The prior function that encodes any previous knowledge we have about the parameter 
        space constraints.

        Parameters:
        ------------
        pars: list_like
              list of free parameters of the model

        Returns:
        ---------
        Uniform prior (in this case) distribution of the parameters.
        The ranges given in this scipt have to be made more compact of course! 
        '''
        if self.dat_type == 'i':
           prior = naima.uniform_prior(pars[0], -2.5, -1.5) \
                 #+naima.uniform_prior(pars[0], -np.inf, np.inf) \
                 #+ naima.uniform_prior(pars[0], 1e14, 1e21) \
                 #+ naima.uniform_prior(pars[1], 0, 500) \
                 #+ naima.uniform_prior(pars[3], 2, 1e15)
        
        elif self.dat_type == 'o':
            prior = naima.uniform_prior(pars[0], 1e14, 1e21) \
                  + naima.uniform_prior(pars[1], 0, 500) \
                  + naima.uniform_prior(pars[2], 0, np.inf) \
                  + naima.uniform_prior(pars[3], 1, 5) \
                  + naima.uniform_prior(pars[4], 2, 1e15) 
              #    + naima.uniform_prior(pars[5], 0, 60.) \
              #    + naima.uniform_prior(pars[6], 1, 100)          
        return prior

    
    def fitter(self, p0, labels, xray_data, vhe_data):
        '''
        This is the actual fitting function.

        Parameters:
        ------------
        p0: list 
            free parameters; 1st guess (compact using InteractiveModelFitter)
        labels: list
                names of the free parameters
        xray_data: astropy Table
                    x-ray data table for fitting
        vhe_data: astropy Table
                    gamma-ray data table for fitting
 
        After the fit a few files and plots are generated -
         - SED with the maximum lilelihood model fit in black
         - Posterior distribution of the free parameters
         - An ascii table containing results of the fitted parameters,
           goodness of fit (given by BIC in metadata), etc.
        '''

        print("Executing the fit...")
  
        #An interactive window helps to adjust the starting point of sampling
        #before calling run_sampler. 
        imf = naima.InteractiveModelFitter(self.model_func, p0, 
                                           e_range=[1e-7*u.eV , 1e13*u.eV], 
                                           e_npoints=25, labels=labels)
        #data = [xray_data,vhe_data]
        #imf = naima.InteractiveModelFitter(self.model_func, p0, data=data, labels=labels)
        p0 = imf.pars

        #Run sampler. nwalkers > len(parameter space). 
        #Numbers for nwalkers, nburn, nrun are only preliminary here for fast run.
        sampler, pos = naima.run_sampler(data_table=[xray_data,vhe_data],
                       p0=p0,  
                       labels=labels,
                       model=self.model_func,
                       prior=self.prior_func,
                       nwalkers=16,
                       nburn=8,
                       nrun=6,
                       threads=4,
                       prefit=False,
                       interactive=False)

        #save run to hdf5 file which can be accesed later by naima.read_run
        naima.save_run('data_fit_run', sampler)
        #Diagnostic plots
        naima.save_diagnostic_plots('data_fit_plots', sampler, sed=True, blob_labels=['Spectrum'])
        naima.save_results_table('data_fit_table', sampler)

    def main(self):
        '''
        Main function
        '''
        # Read data
        xray = ascii.read(self.xray)
        vhe = ascii.read(self.vhe)

        #initial guess. 
        p_init = self.p0
        #if self.dat_type == 'i':
        #   labels = ['R(cm)','B(G)','norm','index', 'gamma_max']
        if self.dat_type == 'i':
           labels = ['index']
        elif self.dat_type == 'o':
           labels = ['R(cm)','B(G)','norm','index', 'gamma_max', 'theta', 'delta']
        #NOTE: NAIMA can also be used to guess a starting value of magnetic field
        #from the ratio of X-ray to gamma-ray luminosity.
        ##B0 = 2*naima.estimate_B(xray_data, vhe_data).to('G').value

        #call the fitter function
        self.fitter(p_init, labels, xray, vhe)
        

if __name__ == '__main__':
   
    fitter_obj = Fitmodel()
    fitter_obj.main()
