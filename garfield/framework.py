#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:20:11 2020

@author: sjoly
"""
import numpy as np
from Genetic_algorithm import *
from scipy.optimize import minimize

'''def generate_parameterBounds(pars, margin=0.1):
    """
    Function used to regenerate parameter bounds when using the minimization algorithm.
    Margin argument can be given to allow each parameter to deviate by 100*margin [%].
    """
    pars_reshape = pars.reshape(-1, 3)
    new_bounds = []
    for par in pars_reshape:
        new_bounds.extend(tuple(sorted(((1-margin)*p, (1+margin)*p))) for p in par)
    return new_bounds'''

def display_resonator_parameters(solution):
    """
    Displays resonance parameters in a formatted table using ASCII characters.

    Args:
        solution: A NumPy array of resonator parameters, typically shaped (n_resonators, 3).
    """

    n_resonators, _ = solution.reshape(-1,3).shape
    header_format = "{:^10}|{:^24}|{:^18}|{:^18}"
    data_format = "{:^10d}|{:^24.2e}|{:^18.2f}|{:^18.3e}"

    print("\n")
    print("-" * 70)

    # Print header
    print(header_format.format("Resonator", "Rs [Ohm/m or Ohm]", "Q", "fres [Hz]"))
    print("-" * 70)

    # Print data
    for i, parameters in enumerate(solution.reshape(-1,3)):
        print(data_format.format(i + 1, *parameters))

    print("-" * 70)

class GeneticAlgorithm:
    def __init__(self, 
                 frequency_data, 
                 impedance_data, 
                 time_data, 
                 wake_data, 
                 N_resonators, 
                 parameterBounds,
                 minimizationFunction, 
                 fitFunction
                ):
    
        self.frequency_data = frequency_data
        self.impedance_data = impedance_data
        self.time_data = time_data
        self.wake_data = wake_data
        self.N_resonators = N_resonators
        self.parameterBounds = parameterBounds
        self.minimizationFunction = minimizationFunction
        self.fitFunction = fitFunction
        
        self.geneticParameters = None
        self.minimizationParameters = None
                
    def check_impedance_data(self):
        """
        Small function to avoid 0 frequency leading to zero division when using resonators.
        """
        mask = np.where(self.frequency_data > 0.)[0]
        self.frequency_data = self.frequency_data[mask]
        self.impedance_data = self.impedance_data[mask]
    
    
    def run_geneticAlgorithm(self, 
                             maxiter=2000, 
                             popsize=15, 
                             mutation=(0.1, 0.5), 
                             crossover_rate=0.8, 
                             tol=0.01, 
                             #workers=-1, 
                             vectorized=False,
                             solver='scipy',
                             iteration_convergence=False, debug=False):
        geneticParameters, warning = generate_Initial_Parameters(self.parameterBounds, 
                                                           self.minimizationFunction, 
                                                           self.fitFunction, 
                                                           self.frequency_data, 
                                                           self.impedance_data, 
                                                           maxiter=maxiter, 
                                                           popsize=popsize, 
                                                           mutation=mutation, 
                                                           crossover_rate=crossover_rate,
                                                           tol=tol,
                                                           solver=solver
                                                           #workers=workers, vectorized=vectorized,
                                                           #iteration_convergence=iteration_convergence
                                                                )

        self.geneticParameters = geneticParameters
        self.warning = warning
        display_resonator_parameters(self.geneticParameters)
            
    def run_minimizationAlgorithm(self, margin=0.1, method='Nelder-Mead'):
        """
        Minimization algorithm is used to refine results obtained by the genetic algorithm. 
        They are used as initial guess for the algorithm and each parameter is allowed to be
        increased or decreased by 100*margin [%].
        """
        print('Method for minimization : '+method)
        minimization_function = partial(self.minimizationFunction, fitFunction=self.fitFunction,
                                x=self.frequency_data, y=self.impedance_data)
        
        if self.geneticParameters is not None:
            minimizationBounds = [sorted(((1-margin)*p, (1+margin)*p)) for p in self.geneticParameters]
            minimizationParameters = minimize(minimization_function, 
                                              x0=self.geneticParameters,
                                              bounds=minimizationBounds,
                                              tol=1, #empiric value, documentation is cryptic
                                              method=method, 
                                              options={'maxiter': self.N_resonators * 1000,
                                                       'maxfev': self.N_resonators * 1000,
                                                       'disp': False,
                                                       'adaptive': True}
                                             )
        else:
            print('Genetic algorithm not run, minimization only')
            minimizationParameters = minimize(minimization_function, 
                                              x0=np.mean(self.parameterBounds, axis=1),
                                              bounds=self.parameterBounds, 
                                              method=method, 
                                              tol=1,
                                              options={'maxiter': self.N_resonators * 5000,
                                                       'maxfev': self.N_resonators * 5000,
                                                       'disp': False,
                                                       'adaptive': True}
                                             ) 
        self.minimizationParameters = minimizationParameters.x
        display_resonator_parameters(self.minimizationParameters)