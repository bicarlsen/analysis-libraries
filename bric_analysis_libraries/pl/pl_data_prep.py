#!/usr/bin/env python
# coding: utf-8

# # Photoluminescence Analysis

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy as sp
import scipy.constants as phys
from scipy.optimize import curve_fit

from bric_analysis_libraries import standard_functions as std


# In[ ]:


def wl_to_en( l ):
    """
    Converts a wavelength, given in nm, to an energy in eV.
    
    :param l: The wavelength to convert, in nm.
    :returns: The corresponding energy in eV.
    """
    a = phys.physical_constants[ 'electron volt-joule relationship' ][ 0 ] # J
    return phys.Planck* phys.c/( a* l* 1e-9 )


def en_to_wl( e ):
    """
    Converts an energy, given in eV, to a wavelength.
    
    :param e: The energy to convert, in eV.
    :returns: The corresponding wavelength in nm.
    """
    a = phys.physical_constants[ 'electron volt-joule relationship' ][ 0 ] # J
    return 1e9* phys.Planck* phys.c/( a* e )


# In[2]:


def normalize( df ):
    """
    Normalize all spectrum.
    
    :param df: The Pandas DataFrame to normalize.
    :returns: The normalized DataFrame.
    """
    return df.apply( lambda x: x/ x.max() )


def index_to_energy( df, scale = False ):
    """
    Converts a Pandas Index with wavelength index to energy index.
    
    :param df: The Pandas DataFrame, with wavelength indices in nanometers, to convert.
    :param scale: Scale counts to maintain bin volume. [Default: False]
    :returns: A new DataFrame indexed by energy in eV.
    """
    edf = df.copy()
    
    en = wl_to_en( df.index.values )
    edf.index = pd.Index( en )
    
    if scale:
        ratio = np.diff( df.index ) / np.diff( edf.index ) # find ratio of widths
        ratio /= ratio.max()
        ratio = np.insert( ratio, 0, ratio[ 0 ] )
        
        edf = edf.multiply( ratio, axis = 0 ) # multiply counts by normalized ratio
    
    return edf


def df_to_energy( df ):
    """
    Converts a DataFrame indexed by wavelength to energy.
    
    :param df: DataFrame indexed by wavelength.
    :returns: DataFrame indexed by energy.
    """
    

