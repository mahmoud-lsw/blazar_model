import numpy as np
from ssc_model.model import model as mod
from ssc_model.numerics import numerics
import astropy.units as u
from astropy.table import Table
import naima
import argparse
# usage eg (for 'intrinsic = True'):
#-----------------------------------
# python fitmodel.py -z 0.047 -f data_table_gamma.dat -free 2.1 1e16 0.9 1.7e5
#-f argument takes more than one files as well. Any number of data files
#Order of the positional args (all free params) have to be maintained
# 'python fitmodel.py -h' : shows the order of positional args
#python fitmodel.py -z 0.047 -f data_table_xray.dat data_table_gamma.dat -free 2.1 1e16 0.9 1.7e5

__all__ = ['Fitmodel', 'fitter']


class Fitmodel:
    ''' DESCRIPTION: Class to derive the best fit and uncertainty distribution
        of the free spectral parameters through Markov Chain Monte Carlo
        sampling of their likelihood distribution.

        find the max. likelihood model to fit to data
    '''

    def __init__(self, intrinsic=True):
        """
        Parameters:
        -----------
        intrinsic : bool
                    whether the given data files represent the intrinsic
                    or the observed spectral points (bt default intrinsic)
        """
        parser = argparse.ArgumentParser(
                    description='find the max. likelihood model to fit to data')
        parser.add_argument('-z', metavar='--REDSHIFT', type=float)
        parser.add_argument('-f', metavar='--DATA-FILES', type=str, nargs='+')
        parser.add_argument('-free', metavar='--FREE_PARAMS', nargs='+',
                                    type=float, help="index R(cm) B(G) "
                                                     "gamma-max theta(deg) "
                                                     "bulk-lorentz-fac...")
        args = parser.parse_args()

        self.intrinsic = intrinsic
        self.files = args.f
        self.z = args.z
        self.p0 = args.free
        print('Provided {} astropy table(s): {}'.format(len(self.files), self.files))
        self.e_npoints = 0
        for file in self.files:
            f = Table.read(file, format='ascii')
            n = len(f['energy'])
            assert len(f['energy']) == len(f['flux'])
            self.e_npoints += n


    def model_func(self, pars, data):
        '''
        The model function will be called during fitting to compare
        with obsrvations.

        Parameters:
        ------------
        pars: list
              list of free parameters of the model
        data: astropy_table
              observational data. Multiple tables can also be
              passed to 'data'

        Returns: 
        ---------
        Flux model to compare with observations.
        '''
        R = pars[1] * u.cm
        B = pars[2] * u.G
        emission_region = dict(R=R.value, B=B.value, t_esc=1.5)

        norm = 1.0e-4 * u.Unit('erg-1')
        index = pars[0]
        injected_spectrum = dict(norm=norm.value, alpha=-index, t_inj=1.5)

        distance = 8.0 * u.kpc

        gamma_max = pars[3]
        gamma_grid = dict(gamma_min=2., gamma_max=gamma_max, gamma_bins=20)

        time_grid = dict(time_min=0., time_max=3., time_bins=50)

        # with the above parameter set, we now obtain a particle distibution
        # from numerics class
        SSC = mod(time_grid, gamma_grid, emission_region, injected_spectrum)
        num = numerics(SSC)
        N_e = num.evolve()

        # Obtain the Syn and IC instances by feeding in the particle
        # distribution model
        SYN = num.synchrotron(N_e)
        IC = num.inverse_compton(N_e)

        # create the flux model to be given as input to run_sampler
        if self.intrinsic:
            energy = np.logspace(-7, 15, self.e_npoints) * u.eV
            ic_flux = IC.sed(energy, distance)
            syn_flux =  SYN.sed(energy, distance)
            model_flux = ic_flux + syn_flux
        else:
            beta = np.sqrt(1. - 1. / (pars[5] ** 2))
            doppler = (1 + self.z) * \
                      (1. / (pars[5] * (1. - beta * np.cos(pars[4]))))
            energy = (np.logspace(-7, 15, 28) * u.eV) * doppler
            model_flux = (IC.sed(energy, distance) +
                          SYN.sed(energy, distance)) * (doppler**3)
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
        The prior function that encodes any previous knowledge
        we have about the parameter space constraints.
        Good choice of prior function is necessary for the fit
        to converge correctly. Parameter space can be best
        constrained from previous observations if any.

        Parameters:
        ------------
        pars: list_like
              list of free parameters of the model

        Returns:
        ---------
        Uniform prior (in this case) distribution of the parameters.
        '''
        #The order of the command line args is very imp
        if self.intrinsic:
            prior = naima.uniform_prior(pars[0], 1.8, 2.5) \
                +naima.uniform_prior(pars[1], 1e14, 8e16) \
                + naima.uniform_prior(pars[2], 0.9, 2.1) \
                + naima.uniform_prior(pars[3], 1.5e5, 2.5e5) \

        else:
            prior = naima.uniform_prior(pars[0], 1.8, 2.5) \
                    + naima.uniform_prior(pars[1], 7e15, 8e15) \
                    + naima.uniform_prior(pars[2], 0.9, 2.1) \
                    + naima.uniform_prior(pars[3], 1.5e5, 2.5e5) \
                    + naima.uniform_prior(pars[4], 3, 35) \
                    + naima.uniform_prior(pars[5], 10, 80)
        return prior

    def fitter(self, p0, labels, datatable):
        '''
        This is the actual fitting function.

        Parameters:
        ------------
        p0: list 
            free parameters; 1st guess (compact using InteractiveModelFitter)
        labels: list
                names of the free parameters
        data_table: astropy Table
                    list of data tables for fitting

        Results of the fit (an astropy table with best param estimate and
        uncertainties & the sed fit) are stored inside 'results_ssc_fit'
        '''

        print("Executing the fit...")

        # An interactive window helps to adjust the starting point of sampling
        # before calling run_sampler.
        imf = naima.InteractiveModelFitter(self.model_func,
                                           p0, sed=True,
                                           e_range=[1e-3 * u.eV, 1e15 * u.eV],
                                           e_npoints = self.e_npoints,
                                           labels=labels)

        p0 = imf.pars

        # Run sampler. nwalkers > len(parameter space).
        # Numbers for nwalkers, nburn, nrun are only preliminary here
        # to achieve fast computation.
        sampler, pos = naima.run_sampler(data_table=datatable,
                                         p0=p0,
                                         labels=labels,
                                         model=self.model_func,
                                         prior=self.prior_func,
                                         nwalkers=20,
                                         nburn=16,
                                         nrun=16,
                                         threads=4,
                                         prefit=False,
                                         data_sed=True,
                                         interactive=False)

        naima.save_results_table('./results_ssc_fit/data_fit_table', sampler)
        fig = naima.plot_fit(sampler, n_samples=50, e_range=[
                             1e-3 * u.eV, 1e15 * u.eV], e_npoints=self.e_npoints)
        #fig.savefig("./results_ssc_fit/likelihoodfitresult_sed.png")
        fig.savefig("./results_ssc_fit/test_sed.png")

    def main(self):
        '''
        Main function
        '''
        readdata = []
        for file in self.files:
            f = Table.read(file, format='ascii')
            readdata.append(f)

        p_init = self.p0

        if self.intrinsic:
            labels = ['index','R (cm)', 'B (G)', 'gamma_max']
        else:
            labels = ['index','R (cm)', 'B (G)', 'gamma_max', \
                      'theta(deg)', 'lorentz']

        self.fitter(p_init, labels, readdata)


if __name__ == '__main__':

    fitter_obj = Fitmodel(intrinsic=True)
    fitter_obj.main()
