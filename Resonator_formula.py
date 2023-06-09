#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:20:11 2020

@author: sjoly
"""
import numpy as np

""" All formulas used come from E. Metral USPAS lecture, availible here :
    http://emetral.web.cern.ch/emetral/USPAS09course/WakeFieldsAndImpedances.pdf """
     
def Resonator_longitudinal_imp(frequencies, Rs, Q, resonant_frequency):
    Zl = Rs / (1 + 1j*Q*(
            frequencies/resonant_frequency - resonant_frequency/frequencies))
    return Zl
 
def Resonator_transverse_imp(frequencies, Rs, Q, resonant_frequency):
    Zt = resonant_frequency / frequencies * Rs / (1 + 1j*Q*(
            frequencies/resonant_frequency - resonant_frequency/frequencies))
    return Zt


def n_Resonator_longitudinal_imp(frequencies, dict_params):
    """
    frequencies : frequencies used in broadband resonator formula
    dict_params : dict where each key corresponds to a single resonator
    resonator parameters are stored in a list following this format : 
        [Rs, Q, resonant_frequency]
    Example dict_params = {1: [1000000.0, 1, 1000000000.0],
                           2 : [2000000.0, 1, 500000000.0]}
    """
    Zl = sum(Resonator_longitudinal_imp(frequencies, *params) for params in dict_params.values())
    return Zl

def n_Resonator_transverse_imp(frequencies, dict_params):
    """
    frequencies : frequencies used in broadband resonator formula
    dict_params : dict where each key corresponds to a single resonator
    resonator parameters are stored in a list following this format : 
        [Rs, Q, resonant_frequency]
    Example dict_params = {1: [1000000.0, 1, 1000000000.0],
                           2 : [2000000.0, 1, 500000000.0]}
    """
    Zt = sum(Resonator_transverse_imp(frequencies, *params) for params in dict_params.values())
    return Zt


def Resonator_longitudinal_wake(times, Rs, Q, resonant_frequency):
    """Be careful, units for this formula are Rs [Ohm/m], Q [], fres [hz]"""
    return (Rs*2*np.pi*resonant_frequency)*np.exp(-2*np.pi*resonant_frequency*times/(2*Q))/Q*(np.cos(2*np.pi*resonant_frequency*np.sqrt(np.abs(1-(1/(4*Q*Q))))*times)+(-2*np.pi*resonant_frequency/(2*Q)/2*np.pi*resonant_frequency*np.sqrt(np.abs(1-(1/(4*Q*Q)))))*np.sin(2*np.pi*resonant_frequency*np.sqrt(np.abs(1-(1/(4*Q*Q))))*times))


def Resonator_longitudinal_wake(times, Rs, Q, resonant_frequency):
    """Be careful, units for this formula are Rs [Ohm/m], Q [], fres [hz]"""
    omega_r = 2*np.pi*resonant_frequency
    omega_r_bar = omega_r*np.sqrt(np.abs(1-(1/(4*Q**2))))    
    cos_term = np.cos(omega_r_bar*times)
    sin_term = np.sin(omega_r_bar*times)
    exp_term = np.exp(-omega_r*times/(2*Q))
    return Rs*omega_r/Q*exp_term*(cos_term - omega_r / (2*Q*omega_r_bar)*sin_term)

def Resonator_transverse_wake(times, Rs, Q, resonant_frequency):
    """Be careful, units for this formula are Rs [Ohm/m], Q [], fres [hz]"""
    return ((Rs*2*np.pi*resonant_frequency)/(Q*np.sqrt(np.abs(1-(1/(4*Q*Q))))))* np.exp(-1*np.pi*resonant_frequency*times/Q)*np.sin(2*np.pi*resonant_frequency*np.sqrt(np.abs(1-(1/(4*Q*Q))))*times)

def Resonator_transverse_wake(times, Rs, Q, resonant_frequency):
    """Be careful, units for this formula are Rs [Ohm/m], Q [], fres [hz]"""
    omega_r = 2*np.pi*resonant_frequency
    omega_r_bar = omega_r*np.sqrt(np.abs(1-(1/(4*Q**2))))
    sqrt_term = np.sqrt(np.abs(1-(1/(4*Q*Q))))
    exp_term = np.exp(-omega_r*times/2/Q)
    sin_term = np.sin(omega_r*sqrt_term*times)
    return omega_r**2*Rs/(Q*omega_r_bar) * exp_term * sin_term

def n_Resonator_longitudinal_wake(times, dict_params):
    """
    frequencies : frequencies used in broadband resonator formula
    dict_params : dict where each key corresponds to a single resonator
    resonator parameters are stored in a list following this format : 
        [Rs, Q, resonant_frequency]
    Example dict_params = {1: [1000000.0, 1, 1000000000.0],
                           2 : [2000000.0, 1, 500000000.0]}
    """
    Wl = sum(Resonator_longitudinal_wake(times, *params) for params in dict_params.values())
    return Wl

def n_Resonator_transverse_wake(times, dict_params):
    """
    frequencies : frequencies used in broadband resonator formula
    dict_params : dict where each key corresponds to a single resonator
    resonator parameters are stored in a list following this format : 
        [Rs, Q, resonant_frequency]
    Example dict_params = {1: [1000000.0, 1, 1000000000.0],
                           2 : [2000000.0, 1, 500000000.0]}
    """
    Wt = sum(Resonator_transverse_wake(times, *params) for params in dict_params.values())
    return Wt

def Resonator_potential_transverse_wake(times, sigma, Rs, Q, resonant_frequency):
    """Be careful, units for this formula are Rs [Ohm/m], Q [], fres [hz]"""
    Q_prime = np.sqrt(Q**2 - 1/4)
    z1 = Q_prime/Q*2*np.pi*resonant_frequency*sigma + 1j*(np.pi*resonant_frequency*sigma/Q - times/sigma)
    wt = 2*np.pi*resonant_frequency*Rs/2/Q_prime * np.exp(-times**2/2/sigma**2) * np.imag(erf(z1/np.sqrt(2)))
    return wt