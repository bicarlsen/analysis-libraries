#!/usr/bin/env python
# coding: utf-8

# # QCM Analysis

# ## Imports

# In[1]:


import logging

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from bric_analysis_libraries import standard_functions as std


# # Analysis

# In[29]:


def sauerbrey( freq, f0, density = 2.648, shear = 2.947e11 ):
    """
    The Sauerbrey equation, solved for mass change per unit area.
    The realtive change in frequency should be less than 5%,
    otherwise use Z-matching.
    
    :param freq: Measured frequency in Hertz.
    :param f0: Fundamental frequency in Hertz.
    :param density: Density of sensor substrate in g/cm^3. [Default: Quartz (2.648)]
    :param shear: Shear modulus of sensor substrate in g/( cm* s ). [Default: Quartz (2.947 e11) ]  
    """
    # check if larger than 5% change
    delta = np.abs( ( freq - f0 )/ f0 )
    if delta.max() > 0.05:
        logging.warning( 'Frequency change is large than 5%. Consider using Z-match method instead.' )
    
    coeff = np.sqrt( density* shear )/ ( 2* np.square( f0 ) )
    m_delta = -coeff* ( freq - f0 )
    
    return m_delta
    
    
def z_match( 
    freq, 
    f0, 
    film_density, 
    film_shear,
    freq_constant = 1.668e13,
    sub_density = 2.648,
    sub_shear = 2.974e11
):
    """
    The Z-match equation.
    Used when relative frequency change is larger than 5%.
    
    :param freq: Frequency of the loaded sensor in Hertz.
    :param f0: Frequency of the unloaded sensor in hertz.
    :param film_density: Density of the film in g/cm^3.
    :param film_shear: Shear modulus of the film in g/( cm* s ).
    :param freq_constant: Frequency constant of the sensor in Hz* Angstrom. [Default: Quartz (1.66 e13)]
    :param sub_density: Density of sensor substrate in g/cm^3. [Default: Quartz (2.648)]
    :param sub_shear: Shear modulus of sensor substrate in g/( cm* s ). [Default: Quartz (2.947 e11) ] 
    """
    z = np.sqrt( sub_density* sub_shear/( film_density* film_shear ) )
    coeff = freq_constant* sub_density/( np.pi* z* freq )
    tan_arg = np.pi*( f0 - freq )/ f0
    
    m = coeff* np.arctan( z* np.tan( tan_arg ) )
    return m
    


# In[27]:


def sauerbrey_mass_change( df, f0 = 5e6, density = 2.648, shear = 2.947e11  ):
    """
    Creates a DataFrame of mass changes calculated with the Sauerbrey equation.
    
    :param df: DataFrame containing frequencies in Hertz.
    :param f0: The undamental freqeuncy of the sensor. [Default: 5 MHz]
    :param density: Density of sensor substrate in g/cm^3. [Default: Quartz (2.648)]
    :param shear: Shear modulus of sensor substrate in g/( cm* s ). [Default: Quartz (2.947 e11) ]  
    :returns: DataFrame of mass changes in grams.
    """
    return df.apply( lambda x: sauerbrey( x, f0, density, shear ) )
    
    
def z_match_mass_change(
    df,
    f0,  
    film_density, 
    film_shear,
    freq_constant = 1.668e13,
    sub_density = 2.648,
    sub_shear = 2.974e11
):
    """
    The Z-match equation.
    Used when relative frequency change is larger than 5%.
    
    :param freq: Frequency of the loaded sensor in Hertz.
    :param f0: Frequency of the unloaded sensor in hertz.
    :param film_density: Density of the film in g/cm^3.
    :param film_shear: Shear modulus of the film in g/( cm* s ).
    :param freq_constant: Frequency constant of the sensor in Hz* Angstrom. [Default: Quartz (1.66 e13)]
    :param sub_density: Density of sensor substrate in g/cm^3. [Default: Quartz (2.648)]
    :param sub_shear: Shear modulus of sensor substrate in g/( cm* s ). [Default: Quartz (2.947 e11) ] 
    """
    return df.apply( lambda x:
        z_match( 
            x, 
            f0, 
            film_density, 
            film_shear,
            freq_constant = freq_constant,
            sub_density = sub_density,
            sub_shear = sub_shear
        )
    )


# # Work

# In[ ]:




