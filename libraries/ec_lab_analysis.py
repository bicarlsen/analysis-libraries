
# coding: utf-8

# # EC Lab Analysis

# ## Imports

# In[3]:


import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from importlib import reload
import standard_functions as std


# In[ ]:


reload( std )


# # Analysis

# In[ ]:


def get_rest_times( df, index = 'voltage', threshold = 0 ):
    """
    Gets the amount of time 
    :param df: The Pandas DataFrame containing the data
    :param index: The column name to 
    """


# In[ ]:


def plot_measurement( ax, data, name ):
    # remove extra levels
    levels = len( data.columns.levels )
    if levels > 1:
        data.columns = data.columns.droplevel( list( range( levels - 1 ) ) )
        
    ax.scatter( x = data.index.values, y = data.voltage )
    ax.set_title( name, fontsize = 10 )
    ax.set_xlabel( '' )


def plot_measurement_density( ax, data, name ):
    """
    Plot measurement density.
    """
    if 'metric' in data.columns.names:
        data = data.xs( 'voltage', level = 'metrics', axis = 1 )
    
    data.plot.hist( ax = ax, legend = False, logy = True )
    ax.set_ylim( bottom = 1, top = 1e6 )
    ax.set_title( name, fontsize = 10 )
    ax.set_xlabel( '' )

