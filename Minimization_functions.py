#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 16:34:10 2020

@author: sjoly
"""
import numpy as np

def pars_to_dict(pars):
    """Converts a list of parameters into a dictionary of parameter groups.

    This function takes a list of parameters `pars` and groups them into
    dictionaries of three parameters (e.g. Rs, Q, resonant_frequency) each. 
    The keys of the resulting dictionary are integers starting from 0, 
    and the values are lists containing three consecutive parameters from 
    the input list.

    Args:
        pars: A list or array of parameters to be grouped.

    Returns:
        dict: A dictionary where keys are integers and values are
             lists of three parameters.

    Raises:
        ValueError: If the length of `pars` is not a multiple of 3 or is empty.
    """

    if not pars:
        raise ValueError("Input list cannot be empty")

    if len(pars) % 3 != 0:
        raise ValueError("Input list length must be a multiple of 3")

    grouped_parameters = {}
    for i in range(0, len(pars), 3):
        grouped_parameters[i // 3] = pars[i : i + 3]

    return grouped_parameters

def sumOfSquaredError(parameters, fit_function, x, y):
    """Calculates the sum of squared errors (SSE) for a given fit function.

    This function computes the SSE between the predicted values from a fit function and
    the actual data points. It works with both real and imaginary components of the data.

    Args:
        parameters: Array of parameters used by the fit_function.
        fit_function: Function that takes parameters and x values as input and
            returns predicted y values (including real and imaginary parts).
        x: Array of x values for the data.
        y: Array of y values for the data (including real and imaginary parts).

    Returns:
        The sum of squared errors (SSE).
    """

    grouped_parameters = pars_to_dict(parameters)
    predicted_y = fitFunction(x, grouped_parameters)
    squared_error = np.nansum((y.real - predicted_y.real)**2 + (y.imag - predicted_y.imag)**2)
    return squared_error

def sumOfSquaredErrorReal(pars, fitFunction, x, y):
    """Calculates the real sum of squared errors (SSE) for a given fit function.

    This function computes the SSE between the predicted values from a fit function and
    the actual data points. It works only with the real component of the data.

    Args:
        parameters: Array of parameters used by the fit_function.
        fit_function: Function that takes parameters and x values as input and
            returns predicted y values (including only the real part).
        x: Array of x values for the data.
        y: Array of y values for the data (including only the real part).

    Returns:
        The real sum of squared errors (SSE).
    """

    grouped_parameters = pars_to_dict(parameters)
    predicted_y = fitFunction(x, grouped_parameters)
    squared_error = np.nansum((y.real - predicted_y.real)**2)
    return squared_error

def logsumOfSquaredError(parameters, fitFunction, x, y):
    """Calculates the sum of log squared errors for a given fit function.

    This function computes the log squared errors between the predicted values from a fit function 
    and the actual data points. It works with both real and imaginary components of the data.

    Args:
        parameters: Array of parameters used by the fit_function.
        fit_function: Function that takes parameters and x values as input and
            returns predicted y values (including real and imaginary parts).
        x: Array of x values for the data.
        y: Array of y values for the data (including real and imaginary parts).

    Returns:
        The sum of log squared errors.
    """
    grouped_parameters = pars_to_dict(parameters)
    predicted_y = fitFunction(x, grouped_parameters)
    log_squared_error = np.nansum(np.log((y.real - predicted_y.real)**2 + (y.imag - predicted_y.imag)**2))
    return log_squared_error

def logsumOfSquaredErrorReal(parameters, fitFunction, x, y):
    """Calculates the real sum of log squared errors for a given fit function.

    This function computes the real log squared errors between the predicted values 
    from a fit function and the actual data points. 
    It works only with the real component of the data.

    Args:
        parameters: Array of parameters used by the fit_function.
        fit_function: Function that takes parameters and x values as input and
            returns predicted y values (including only the real part).
        x: Array of x values for the data.
        y: Array of y values for the data (including only the real part).

    Returns:
        The real sum of log squared errors.
    """
    
    grouped_parameters = pars_to_dict(parameters)
    predicted_y = fitFunction(x, grouped_parameters)
    log_squared_error = np.nansum(np.log((y.real - predicted_y.real)**2))
    return log_squared_error