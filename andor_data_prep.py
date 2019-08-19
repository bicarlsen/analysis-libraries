#!/usr/bin/env python
# coding: utf-8

# # Andor Data Prep

# ### Imports

# In[1]:


import os
import io
import sys
import re
import glob
import math

import numpy as np
import pandas as pd

import import_ipynb
import standard_functions as std


# # Data Prep

# In[1]:


# convenience functions for common extractions

def sample_from_file_name( file ):
    name_search  = '^(.*?)' # use lazy matching
    return std.metadata_from_file_name( name_search, file )
    

def angle_from_file_name( file ):
    angle_search = '<>deg'
    return std.metadata_from_file_name( angle_search, file, True )
      
    
def power_from_file_name( file ):
    power_search = '<>mw'
    return std.metadata_from_file_name( power_search, file, True )
    
    
def time_from_file_name( file ):
    time_search  = '<>' 
    return std.metadata_from_file_name( time_search, file, True, decimal = 's' )


def wavelength_from_file_name( file ):
    wavelength_search = '<>nm'
    return std.metadata_from_file_name( wavelength_search, file, True )


def temperature_from_file_name( file ):
    temperature_search = '<>'
    return std.metadata_from_file_name( temperature_search, file, True, decimal = 'k' )


def pressure_from_file_name( file ):
    pressure_search = '<>hpa'
    return std.metadata_from_file_name( pressure_search, file, True, decimal = 'p' )


# In[5]:


def get_standard_metadata_value( file, metadata ):
    """
    Gets metadata values from a file path
    
    :param file: The file path to search
    :param metadata: The key of a standard metadata to retrieve
        [ 'sample', 'power', 'wavelength', 'time' ]
    :returns: A list of metadata values
    """
    return getattr( sys.modules[ __name__ ], '{}_from_file_name'.format( metadata ) )( file )


def get_standard_metadata_values( file, metadata ):
    """
    Gets metadata values from a file path
    
    :param file: The file path to search
    :param metadata: A list of standard metadata to retrieve
        [ 'sample', 'power', 'wavelength', 'time' ]
    :returns: A list of metadata values
    """
    return [ getattr( sys.modules[ __name__ ], '{}_from_file_name'.format( meta ) )( file ) for meta in metadata ]


def get_metadata_values( file, metadata ):
    """
    Gets metadata values from a file path
    
    :param file: The file path to search
    :param metadata: Metadata from the file name is turned into MultiIndex columns
        + If list, use standard keywords to include in index [ 'sample', 'power', 'wavelength', 'time' ]
        + If Dictionary, keys indicate level name, value is pattern to match
            or another dictionary with 'search' key being the pattern to match, and additional
            entries matching arguments passed to standard_functions#metadata_from_filename.
            + Reseserved key 'standard' can be provided with a list value to get standard metadata
    :returns: A list of metadata values
    """

    if isinstance( metadata, list ):
        # use standard metadata
        return get_standard_metadata_values( file, metadata )

        
    if isinstance( metadata, dict ):
        # use custom metadata
        # key is name, value is regexp pattern or dictionary with pattern and arguments

        header_names = list( metadata.keys() )
        
        # get number of values
        val_len = len( header_names )
        if 'standard' in header_names:
            val_len += len( metadata[ 'standard' ] ) - 1 
           
        vals = header_names.copy()
        for name, search in metadata.items():
            index = header_names.index( name )
            
            if name == 'standard':
                # insert standard keys
                vals[ index ] = get_standard_metadata_values( file, search )

            else:
                # custom key
                if isinstance( search, dict ):
                    pattern = search[ 'search' ]
                    kwargs = search.copy()
                    del kwargs[ 'search' ]
                    
                else:
                    pattern = search
                    kwargs = {}
                
                vals[ index ] = std.metadata_from_file_name( pattern, file, **kwargs )
        
        # fllatten standard keys
        if 'standard' in header_names:
            index = header_names.index( 'standard' )
            vals = vals[ :index ] + vals[ index ] + vals[ index + 1: ]

        return vals


# In[2]:


# TODO: Handle file metadata from Andor if present in file
def import_datum( file, metadata = None, reindex = True, cps = False ):
    """
    Imports data from a single Andor output files
    
    :param file: The file path to read
    :param metadata: Metadata from the file name is turned into MultiIndex columns
        + If list, use standard keywords to include in index [ 'sample', 'power', 'wavelength', 'time' ]
        + If Dictionary, keys indicate level name, value is wither the pattern to match
            or another dictionary with 'search' key being the pattern to match, and additional
            entries matching arguments passed to standard_functions#metadata_from_filename.
            + Reseserved key 'standard' can be provided with a list value to get standard metadata
    :param reindex: Use wavelength as index [Default: True] 
    :param cps: Converts the data to counts per second. 
        A valid time string of the form XsX must be present.
    :returns: A Pandas DataFrame with MultiIndexed columns
    """
    
    data_names = [ 'wavelength', 'counts' ] 
    
    # check for metadata at end of file
    file_metadata = ''
    file_data = ''
    for line in open( file ):
        # check if line begins with a number
        data_match = re.match( '^\d', line )
        if data_match is None:
            # did not match numeric data, place in metadata
            file_metadata += line
            
        else:
            # numeric data
            file_data += line

    file_data = io.StringIO( file_data ) # turn data into file object for reading in
    
    if cps:
        int_time = time_from_file_name( file )
    
    # no metadata, import file
    if metadata is None:
        df = pd.read_csv( file_data, names = data_names, header = None  )
        
        if cps:
            df.counts /= int_time
        
        if reindex:
            df = df.set_index( 'wavelength' )
        
        return df
    
    # get metadata
    f_name = os.path.basename( file )
    
    if isinstance( metadata, list ):
        # use standard metadata
        header_names = metadata.copy()

    elif isinstance( metadata, dict ):        
        header_names = list( metadata.keys() )
        
        if 'standard' in header_names:
            # replace standard with values
            index = header_names.index( 'standard' )
            header_names = header_names[ :index ] + metadata[ 'standard' ] + header_names[ index + 1: ]
    
    header_names.append( 'metrics' )
    
    header_vals = get_metadata_values( os.path.basename( file ), metadata )
    header_vals = [ [ val ] for val in header_vals ] # convert levels to lists for taking product
    header_vals.append( data_names )
    
    header = pd.MultiIndex.from_product( header_vals, names = header_names )
    
    df = pd.read_csv( file_data, header = None )
    
    if cps:
        df.iloc[ :, 1 ] /= int_time
    
    df.columns = header
    
    if reindex:
        if metadata is None:
            # simple index
            df.set_index( 'wavelength' )
            
        else:
            # multindex
            df.index = df.xs( 'wavelength', level = 'metrics', axis = 1 ).values.flatten()
            df.drop( 'wavelength', level = 'metrics', axis = 1, inplace = True )
            df.columns = df.columns.droplevel( 'metrics' )
        
    return df
            
        
        
def import_data( folder_path, file_pattern = '*.csv', metadata = None, cps = False, interpolate = 'linear', fillna = 0 ):
    """
    Imports data from Andor output files
    
    :param folder_path: The file path containing the data files
    :param file_pattern: A glob pattern to filter the imported files [Default: '*']
    :param metadata: Metadata from the file name is turned into MultiIndex columns.
        + If list, use standard keywords to include in index 
            [ 'sample', 'power', 'wavelength', 'time', 'temperature' ]
        + If Dictionary, keys indicate level name, value is pattern to match
            + Reseserved key 'standard' can be provided with a list value to get standard metada
    :param cps: Converts the data to counts per second. 
        A valid time string of the form XsX must be present.        
    :param interpolate: How to interpolate data for a common index [Default: linear]
        Use None to prevent reindexing
    :param fillna: Value to fill NaN values with [Default: 0]
    :returns: A Pandas DataFrame with MultiIndexed columns
    :raises: RuntimeError if no files are found
    """
    
    # get dataframes from files
    df = []
    files = std.get_files( folder_path, file_pattern )
    if len( files ) == 0:
        # no files found
        raise RuntimeError( 'No files found matching {}'.format( os.path.join( folder_path, file_pattern ) ) )
    
    for file in files:
        data = import_datum( file, metadata = metadata, cps = cps ) # run local import datum function
        df.append( data )
        
    if interpolate is not None:
        df = std.common_reindex( df, how = interpolate, fillna = fillna )
        
    df = pd.concat( df, axis = 1 )
    return df



def correct_spectra( df, correction ):
    """
    Applies a correction spectral data.
    
    :param df: The Pandas DataFrame to correct.
    :param correction: The correction data to apply.
        Should be a tuple of ( camera, grating )
        Camera values are [ 'idus' ]
        Grating values are [ '300', '600' ]
    :returns: The corrected data.
    """
    data_path =  os.path.dirname( __file__ ) + '/data/andor-corrections.pkl'  
    cdf = pd.read_pickle( data_path )
    
    corrections = cdf.xs( ( 'grating', *correction ), axis = 1 )
    cdf = std.common_reindex( [ df, corrections ] )
    corrections = cdf[ 1 ].reindex( df.index )
    
    return df.multiply( corrections, axis = 0 )


# # Work

# In[ ]:




