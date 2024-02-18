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
from pyfde import ClassicDE, JADE

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
    
def show_progress_bar(x, convergence, *karg):
    """
    Callback function to display a progress bar with scipy.
    """
    progress_bar_gui(1, convergence, extra="")
    
def stop_criterion(solver):
    '''
    Based on the criterion used in scipy.optimize.differential_evolution 
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)
    Check that the ratio of the spread of the population fitness compared to its average.
    In other words, if most of the population individuals converge to the same solution indicating
    an optimal solution has been found.
    '''
    population_cost = np.vstack(solver)[:,1]
    population_mean, population_std = np.mean(population_cost), np.std(population_cost)
    criterion = population_std / np.abs(population_mean)
    return criterion

def run_scipy_solver(parameterBounds, 
                     minimization_function,
                     maxiter=2000, 
                     popsize=150, 
                     mutation=(0.1, 0.5), 
                     crossover_rate=0.8,
                     tol=0.01,
                    **kwargs):
    """
    Runs the SciPy differential_evolution solver to minimize a given function.
    
    All the arguments are detailed on this page :
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)
    Setting workers= -1 means all CPUs available will be used for the computation.
    Default parameters used for the DE algorithm taken from https://www.mdpi.com/2227-7390/9/4/427

    Args:
        parameterBounds: A list of tuples representing the upper and lower bounds for each parameter.
        minimization_function: The function to be minimized.
        maxiter: The maximum number of iterations to run the solver for.
        popsize: The population size for the differential evolution algorithm.
        mutation: A tuple of two floats representing the mutation factors.
        crossover_rate: The crossover rate for the differential evolution algorithm.
        tol: The tolerance for convergence.

    Returns:
        A tuple containing:
            - The solution found by the solver.
            - A message indicating the solver's status.
    """    
    result = differential_evolution(minimization_function, 
                                    parameterBounds, 
                                    popsize=popsize, 
                                    tol=tol, 
                                    maxiter=maxiter,
                                    mutation=mutation, 
                                    recombination=crossover_rate, 
                                    polish=False, 
                                    init='latinhypercube',
                                    callback=show_progress_bar,
                                    updating='deferred', 
                                    workers=-1, #vectorized=vectorized
                                   )
    
    # Need to be reworked to use the last population as the new initial population to speed up convergence
    """while ((result.message == 'Maximum number of iterations has been exceeded.') and (iteration_convergence)):
        warning = 'Increased number of iterations by 10% to reach convergence. \n'
        maxiter = int(1.1*maxiter)
        result = differential_evolution(minimization_function,parameterBounds, 
                                        popsize=popsize, tol=tol, maxiter=maxiter,
                                        mutation=mutation, recombination=crossover_rate, polish=False, 
                                        init='latinhypercube',
                                        callback=show_progress_bar,
                                        updating='deferred', workers=-1, #vectorized=vectorized
                                       )

    else:
        warning = ''
    """
    
    solution, message = result.x, result.message

    return solution, message


def run_pyfde_solver(parameterBounds, 
                     minimization_function,
                     maxiter=2000, 
                     popsize=150, 
                     mutation=(0.3, 0.5), 
                     crossover_rate=0.8,
                     tol=0.01,
                    **kwargs):
    """
    Runs the pyfde ClassicDE solver to minimize a given function.

    Args:
        parameterBounds: A list of tuples representing the bounds for each parameter.
        minimization_function: The function to be minimized.
        maxiter: The maximum number of iterations to run the solver for.
        popsize: The population size for the differential evolution algorithm.
        mutation: A tuple of two floats representing the mutation factors.
        crossover_rate: The crossover rate for the differential evolution algorithm.
        tol: The tolerance for convergence.

    Returns:
        A tuple containing:
            - The solution found by the solver.
            - A message indicating the solver's status.
    """

    solver = ClassicDE(
        minimization_function,
        n_dim=len(parameterBounds),
        n_pop=popsize * len(parameterBounds),
        limits=parameterBounds,
        minimize=True,
    )    
    solver.cr, solver.f = crossover_rate, np.mean(mutation)

    for i in range(maxiter):
        best, _ = solver.run(n_it=1)
        progress_bar_gui(1, np.max((tol / stop_criterion(solver), i / maxiter)))
        if stop_criterion(solver) < tol:
            break

    solution, message = best, "Convergence achieved" if i < maxiter else "Maximum iterations reached"

    return solution, message

def run_pyfde_jade_solver(parameterBounds, 
                          minimization_function,
                          maxiter=2000, 
                          popsize=150, 
                          tol=0.01,
                         **kwargs):
    """
    Runs the pyfde JADE solver to minimize a given function.

    Args:
        parameterBounds: A list of tuples representing the bounds for each parameter.
        minimization_function: The function to be minimized.
        maxiter: The maximum number of iterations to run the solver for.
        popsize: The population size for the differential evolution algorithm.
        tol: The tolerance for convergence.

    Returns:
        A tuple containing:
            - The solution found by the solver.
            - A message indicating the solver's status.
    """
    solver = JADE(
        minimization_function,
        n_dim=len(parameterBounds),
        n_pop=popsize * len(parameterBounds),
        limits=parameterBounds,
        minimize=True,
    )

    for i in range(maxiter):
        best, _ = solver.run(n_it=1)
        progress_bar_gui(1, np.max((tol / stop_criterion(solver), i / maxiter)))
        if stop_criterion(solver) < tol:
            break

    solution, message = best, "Convergence achieved" if i < maxiter else "Maximum iterations reached"

    return solution, message

def generate_Initial_Parameters(parameterBounds, minimizationFunction, fitFunction,
                                x_values_data, y_values_data,
                                maxiter=2000, popsize=150, 
                                mutation=(0.1, 0.5), crossover_rate=0.8,
                                tol=0.01,
                                solver='scipy',
                               ):    
    """
    Generates initial parameter guesses for a minimization algorithm.

    This function uses a differential evolution (DE) solver to find approximate
    solutions to a minimization problem, which can be used as initial guesses for
    more precise optimizers.

    Args:
        parameterBounds: A list of tuples representing the upper and lower bounds
                         for each parameter.
        minimization_function: A function that calculates the cost given a set of
                         parameters and data. The signature should be
                         `minimization_function(parameters, fit_function, x_data, y_data)`.
        fit_function: A function that calculates the fit between a model and data.
                       The signature should be `fit_function(parameters, x_data, y_data)`.
        x_values_data: The x-values of the data to fit.
        y_values_data: The y-values of the data to fit.
        maxiter: The maximum number of iterations for the DE solver.
        popsize: The population size for the DE algorithm.
        mutation: A tuple of two floats representing the mutation factors.
        crossover_rate: The crossover rate for the DE algorithm.
        tol: The tolerance for convergence.
        solver: The solver to use for differential evolution. Valid options are
                "scipy", "pyfde", or "pyfde_jade". Defaults to "scipy".

    Returns:
        A tuple containing:
            - The estimated initial parameters found by the DE solver.
            - A message indicating the solver's status.
    """

    
    minimization_function = partial(minimizationFunction, 
                                    fitFunction=fitFunction,
                                    x=x_values_data, 
                                    y=y_values_data
                                   )
    
    # Map solver names to functions
    solver_functions = {
        "scipy": run_scipy_solver,
        "pyfde": run_pyfde_solver,
        "pyfde_jade": run_pyfde_jade_solver,
    }

    solver_function = solver_functions.get(solver)
    if solver == "pyfde_jade":
        mutation, crossover_rate = None, None
    
    if not solver_function:
        raise ValueError(f"Invalid solver name: {solver}")
        
    solution, message = solver_function(parameterBounds, 
                                        minimization_function,
                                        maxiter=maxiter, 
                                        popsize=popsize, 
                                        mutation=mutation, 
                                        crossover_rate=crossover_rate,
                                        tol=tol)
    
    return solution, message