#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 16:33:41 2020

@author: sjoly
"""
import sys
from functools import partial
import numpy as np
from scipy.optimize import differential_evolution

def progress_bar_gui(total, progress, extra=""):
    """
    Displays or updates a console progress bar.

    Original source: https://stackoverflow.com/a/15860757/1391441
    """
    barLength, status = 20, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(barLength * progress))
    text = "\r Convergence to optimal solution : [{}] {:.0f}% {}{}".format(
        "#" * block + "-" * (barLength - block),
        round(progress * 100, 0), extra, status)
    if 100 * progress // 1 == 100:
        text = "\r Convergence to optimal solution : [{}] 100%\n".format("#" * 20)
        sys.stdout.write(text)
        sys.stdout.flush()
    else:
        sys.stdout.write(text)
        sys.stdout.flush()
        

params_each_iteration = []
progress = []
    
def progress_bar(x, convergence, *karg):
    progress.append(convergence)
    params_each_iteration.append(x)
    progress_bar_gui(1, progress[-1], extra="")

def generate_Initial_Parameters(parameterBounds, minimizationFunction, fitFunction,
                                x_values_data, y_values_data,
                                maxiter=2000, popsize=150, 
                                mutation=(0.1, 0.5), crossover_rate=0.8,
                                tol=0.01, 
                                workers=-1, #vectorized=False,
                                iteration_convergence=False):

    """ This function allows to generate adequate initial guesses to use in the minimization algorithm.
    In this step we do not necessarily need to have converged results. Approximate results are enough.
    We use the complex sum of squared errors (sumOfSquaredError) as a loss function, it can be changed
    depending on the problem type.
    One can tweak the number of iterations (maxiter) to stop the algorithm earlier for faster computing time.
    All the arguments are detailed on this page :
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)
    Setting workers= -1 means all CPUs available will be used for the computation.
    Parameters used for the DE algorithm taken from https://www.mdpi.com/2227-7390/9/4/427
    """

    
    minimization_function = partial(minimizationFunction, fitFunction=fitFunction,
                                    x=x_values_data, y=y_values_data)
    
    result = differential_evolution(minimization_function,parameterBounds, 
                                    popsize=popsize, tol=tol, maxiter=maxiter,
                                    mutation=mutation, recombination=crossover_rate, polish=False, 
                                    init='latinhypercube',
                                    callback=progress_bar,
                                    updating='deferred', workers=workers, #vectorized=vectorized
                                   )
    
    while ((result.message == 'Maximum number of iterations has been exceeded.') and (iteration_convergence)):
        warning = 'Increased number of iterations by 10% to reach convergence. \n'
        maxiter = int(1.1*maxiter)
        result = differential_evolution(minimization_function,parameterBounds, 
                                        popsize=popsize, tol=tol, maxiter=maxiter,
                                        mutation=mutation, recombination=crossover_rate, polish=False, 
                                        init='latinhypercube',
                                        callback=progress_bar,
                                        updating='deferred', workers=workers, #vectorized=vectorized
                                       )
        
    else:
        warning = ''
        
    params_each_iteration = []
    progress = []
    
    for i, resonator_parameters in enumerate(result.x.reshape(-1,3)):
        print('Resonator {}'.format(i+1))
        print('Rt = {:.2e} [Ohm/m], Q = {:.2f}, fres = {:.2e} [Hz]'.format(*resonator_parameters))
        print('-'*60)
    
    return result.x, warning + result.message