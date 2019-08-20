#!/usr/bin/env python
# coding: utf-8

# # QCM Data Prep

# ## Imports

# In[1]:


import os
import io
import sys
import re
import glob
import math
import logging

import numpy as np
import pandas as pd

from bric_analysis_libraries import standard_functions as std


# ## Data Prep

# In[10]:


def import_datum( file, reindex = True, sep = ',' ):
    """
    Imports a .txt file from QSense.
    
    :param file: The file path to load.
    :param reindex: Set time as DataFrame Index. [Default: True]
    :param sep: The data seperator. [Default: ,]
    :param
    :returns: A Pandas DataFrame.
    """
    # parse header
    time_pattern = 'Time_(\d+)'  # used to get channel
    dissipation_pattern = 'D(\d+)_(\d+)'
    frequency_pattern = 'f(\d+)_(\d+)'

    with open( file ) as f:
        header = f.readline().split( sep )
        
        time_matches = [ re.match( time_pattern,        head ) for head in header ]
        diss_matches = [ re.match( dissipation_pattern, head ) for head in header ]
        freq_matches = [ re.match( frequency_pattern,   head ) for head in header ]
                
        headers = [ 0 ]* len( header ) # df headers: ( channel, parameter, mode )
        for i in range( len( header ) ):
            time_match = time_matches[ i ]
            diss_match = diss_matches[ i ]
            freq_match = freq_matches[ i ]
                
            if time_match is not None:
                headers[ i ] = ( time_match.group( 1 ), 'time', 0 )
               
            elif diss_match is not None:
                headers[ i ] = ( diss_match.group( 2 ), 'dissipation', int( diss_match.group( 1 ) ) )
                
            elif freq_match is not None:
                headers[ i ] = ( freq_match.group( 2 ), 'frequency', int( freq_match.group( 1 ) ) )
        
    df = pd.read_csv( file, skiprows = 0 )
    df.columns = pd.MultiIndex.from_tuples( headers, names = [ 'channel', 'parameter', 'mode' ] )
    
    if reindex:
        channels = []
        
        # split data by channel
        for name, data in df.groupby( level = 'channel', axis = 1 ):
            channel = name[ 0 ]
            data = data.set_index( ( channel, 'time', 0 ) )
            data.index.rename( 'time', inplace = True )
            channels.append( data )
            
            channels = std.common_reindex( channels )
            df = pd.concat( channels, axis = 1 )
    
    return df


def import_data( folder, file_pattern = '*.txt' ):
    """
    Imports QCM data from a QSense experiment.
    
    :param folder: Folder path containing the data files.
    :param file_pattern: File pattern of data files, in glob format. [Default: *.txt]
    :returns: DataFrame containg imported data.
    """
    return std.import_data( import_datum, folder, file_pattern = file_pattern )


# # Work

# In[ ]:




