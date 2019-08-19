#!/usr/bin/env python
# coding: utf-8

# # Ocean Optics Data Prep

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

# In[101]:


def sample_from_file_name( file ):
    name_search  = '^(.*?)' # use lazy matching
    return std.metadata_from_file_name( name_search, file, delimeter = '_' )


# In[102]:


def get_standard_metadata_value( file, metadata ):
    """
    Gets metadata values from a file path
    
    :param file: The file path to search
    :param metadata: The key of a standard metadata to retrieve
        [ 'sample' ]
    :returns: A list of metadata values
    """
    return getattr( sys.modules[ __name__ ], '{}_from_file_name'.format( metadata ) )( file )


def get_standard_metadata_values( file, metadata ):
    """
    Gets metadata values from a file path
    
    :param file: The file path to search
    :param metadata: A list of standard metadata to retrieve
        [ 'sample' ]
    :returns: A list of metadata values
    """
    return [ getattr( sys.modules[ __name__ ], '{}_from_file_name'.format( meta ) )( file ) for meta in metadata ]


def get_metadata_values( file, metadata ):
    """
    Gets metadata values from a file path.
    
    :param file: The file path to search.
    :param metadata: Metadata from the file name is turned into MultiIndex columns.
        + If list, use standard keywords to include in index [ 'sample' ]
        + If Dictionary, keys indicate level name, value is pattern to match.
            or another dictionary with 'search' key being the pattern to match, and additional
            entries matching arguments passed to standard_functions#metadata_from_filename.
            + Reseserved key 'standard' can be provided with a list value to get standard metadata.
    :returns: A list of metadata values.
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


# In[124]:


def import_datum( file, time = False, cps = False, metadata = None, reindex = True ):
    """
    Imports data from a single Ocean Optics spectrometer file.
    
    :param file: The file path to read.
    :param time: Read the integration time in from the file as a header. [Default: False]
    :param cps: Converts the data to counts per second. [Default: False]
    :param metadata: Metadata from the file name is turned into MultiIndex columns.
        + If list, use standard keywords to include in index. [ 'sample' ]
        + If Dictionary, keys indicate level name, value is either the pattern to match
            or another dictionary with 'search' key being the pattern to match, and additional
            entries matching arguments passed to standard_functions#metadata_from_filename.
            + Reseserved key 'standard' can be provided with a list value to get standard metadata
    :param reindex: Use wavelength as index [Default: True] 
    :returns: A Pandas DataFrame with MultiIndexed columns
    """
    def get_time_from_file( metadata ):
        """
        Get integration time metadata from file.
        
        :param metadata: A list of metadata from the file.
        :returns: The integration time.
        """
        for prop, value in metadata.items():
            if re.match( 'Integration Time', prop ):
                # found integration time
                return float( value )
        
        if 'time' not in header_names:
            # did not find integration time
            raise RuntimeError( 'Could not find integration time in file {}'.format( file ) )
    
    
    data_names = [ 'wavelength', 'counts' ] 
    
    # check for metadata at beginning of file
    lines = [ line.rstrip( '\n' ) for line in open( file ) ]
    
    # serach for data break
    data_break = 'Begin Spectral Data'
    break_line = None
    for index, line in enumerate( lines ):
        if re.search( data_break, line ):
            break_line = index
            break
    
    if break_line is not None:
        # break line found, metadata above, data below
        file_metadata = lines[ :break_line ]
        file_metadata = [ line.split( ':', 1 ) for line in file_metadata ]
        file_metadata = { d[ 0 ].strip(): d[ 1 ].strip() for d in file_metadata if len( d ) == 2 }  
        
    if cps or time:
        int_time = get_time_from_file( file_metadata )
    
    # no metadata, import file
    if ( metadata is None ) and ( time is False ):
        df = pd.read_csv( 
            file, 
            names = data_names, 
            header = None, 
            skiprows = break_line,
            sep = '\t'
        )
        
        if cps:
            # divide counts by integration time
            df = df/ [ 1, int_time ]
        
        if reindex:
            df = df.set_index( 'wavelength' )
        
        return df
    
    # include metadata
    header_names = []
    header_vals = []
    
    if metadata is not None:
        # get metadata form file name
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

        header_vals = get_metadata_values( os.path.basename( file ), metadata )
        header_vals = [ [ val ] for val in header_vals ] # convert levels to lists for taking product
    
    if time:
        header_names.append( 'time' )
        header_vals.append( [ int_time ] )
               
    header_vals.append( data_names )
    header_names.append( 'metrics' )
    
    header = pd.MultiIndex.from_product( header_vals, names = header_names )
    
    df = pd.read_csv( 
        file, 
        header = None, 
        skiprows = break_line,
        sep = '\t'
    )
    
    if cps:
        df = df/ [ 1, int_time ]
    
    df.columns = header
    
    if reindex:
        # multindex
        df.index = df.xs( 'wavelength', level = 'metrics', axis = 1 ).values.flatten()
        df.index = df.index.rename( 'wavelength' )
        df.drop( 'wavelength', level = 'metrics', axis = 1, inplace = True )
        df.columns = df.columns.droplevel( 'metrics' )
        
    return df
        
        
def import_data( 
    folder_path, 
    file_pattern = '*.txt', 
    metadata = None,
    time = False,
    cps = False, 
    interpolate = 'linear', 
    fillna = 0
):
    """
    Imports data from Andor output files
    
    :param folder_path: The file path containing the data files
    :param file_pattern: A glob pattern to filter the imported files [Default: '*']
    :param metadata: Metadata from the file name is turned into MultiIndex columns.
        + If list, use standard keywords to include in index 
            [ 'sample', 'power', 'wavelength', 'time', 'temperature' ]
        + If Dictionary, keys indicate level name, value is pattern to match
            + Reseserved key 'standard' can be provided with a list value to get standard metada
    :param time: Read the integration time in from the file as a header. [Default: False]
    :param cps: Converts the data to counts per second. 
        A valid time string of the form XsX must be present.        
    :param interpolate: How to interpolate data for a common index [Default: linear]
        Use None to prevent reindexing
    :param fillna: Value to fill NaN values with [Default: 0]
    :param reindex: Reindex the DataFrame using 
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
        data = import_datum( 
            file, 
            metadata = metadata, 
            cps = cps, 
            time = time
        ) # run local import datum function
        df.append( data )
        
    if interpolate is not None:
        df = std.common_reindex( df, how = interpolate, fillna = fillna )
        
    df = pd.concat( df, axis = 1 )
    return df


# # Work
