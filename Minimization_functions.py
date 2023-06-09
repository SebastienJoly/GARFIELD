#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 16:34:10 2020

@author: sjoly
"""
import numpy as np

def pars_to_dict(pars):
    dict_params = {}
    for i in range(int(len(pars)/3)):
        dict_params[i] = pars[3*i:3*i+3]
    return dict_params

'''def sumOfSquaredError(pars, fitFunction, frequencies, impedance_data):
    """
    Complex sum of squared errors
    """
    dict_params = pars_to_dict(pars)
    predicted_impedance = fitFunction(frequencies, dict_params)
    ErrorReal = impedance_data.real - predicted_impedance.real
    ErrorImag = impedance_data.imag - predicted_impedance.imag
    sumOfSquaredErrorReal = np.nansum(ErrorReal ** 2)
    sumOfSquaredErrorImag = np.nansum(ErrorImag ** 2)
    return sumOfSquaredErrorReal + sumOfSquaredErrorImag'''

def sumOfSquaredError(pars, fitFunction, x, y):
    """
    Complex sum of squared errors
    
    Parameters:
        pars (array-like): Array of fit parameters.
        fit_function (callable): Function to compute the model impedance.
        x (array-like): Array of x values.
        y (array-like): Array of y values.
        
    Returns:
        float: Sum of squared errors.
    """
    dict_params = pars_to_dict(pars)
    predicted_y = fitFunction(x, dict_params)
    SquarredError = (y.real - predicted_y.real)**2 + (y.imag - predicted_y.imag)**2
    return np.nansum(SquarredError)

def sumOfSquaredErrorReal(pars, fitFunction, x, y):
    """
    Complex sum of squared errors
    
    Parameters:
        pars (array-like): Array of fit parameters.
        fit_function (callable): Function to compute the model impedance.
        x (array-like): Array of x values.
        y (array-like): Array of y values.
        
    Returns:
        float: Sum of squared errors.
    """
    dict_params = pars_to_dict(pars)
    predicted_y = fitFunction(x, dict_params)
    SquarredError = (y.real - predicted_y.real)**2
    return np.nansum(SquarredError)

'''def logsumOfSquaredError(pars, fitFunction, frequencies, impedance_data):
    """
    Complex sum of squared errors
    """
    dict_params = pars_to_dict(pars)
    predicted_impedance = fitFunction(frequencies, dict_params)
    ErrorReal = np.log(np.abs(impedance_data.real)) - np.log(np.abs(predicted_impedance.real))
    ErrorImag = np.log(np.abs(impedance_data.imag)) - np.log(np.abs(predicted_impedance.imag))
    sumOfSquaredErrorReal = np.nansum(ErrorReal ** 2)
    sumOfSquaredErrorImag = np.nansum(ErrorImag ** 2)
    return sumOfSquaredErrorReal + sumOfSquaredErrorImag'''

def logsumOfSquaredError(pars, fitFunction, x, y):
    """
    Complex log sum of squared errors
    
    Parameters:
        pars (array-like): Array of fit parameters.
        fit_function (callable): Function to compute the model impedance.
        x (array-like): Array of x values.
        y (array-like): Array of y values.
        
    Returns:
        float: Log sum of squared errors.
    """
    dict_params = pars_to_dict(pars)
    predicted_y = fitFunction(x, dict_params)
    LogSquarredError = np.log((y.real - predicted_y.real)**2 + (y.imag - predicted_y.imag)**2)
    return np.nansum(LogSquarredError)

def logsumOfSquaredErrorReal(pars, fitFunction, x, y):
    """
    Complex log sum of squared errors
    
    Parameters:
        pars (array-like): Array of fit parameters.
        fit_function (callable): Function to compute the model impedance.
        x (array-like): Array of x values.
        y (array-like): Array of y values.
        
    Returns:
        float: Log sum of squared errors.
    """
    dict_params = pars_to_dict(pars)
    predicted_y = fitFunction(x, dict_params)
    LogSquarredError = np.log((y.real - predicted_y.real)**2)
    return np.nansum(LogSquarredError)