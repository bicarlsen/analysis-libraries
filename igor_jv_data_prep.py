
# coding: utf-8

# # Igor JV Curve Analysis
# Used to import and analyze JV curves taken from Igor

# ### Imports

# In[1]:


import os
import sys
import re
import glob
import math

import numpy as np
import pandas as pd

import import_ipynb
import standard_functions as std


# # Data Prep

# In[2]:


def get_cell_name( content ):
    """
    Get the cell name from an igor output file
    
    :param content: The content to search
    :returns: The cell name of the file
    """
    name_search = 'X S_cellname="(.*)"'
    match = re.search( name_search, content, flags = re.IGNORECASE )
    
    if match is None:
        # no match found, raise exception
        raise RuntimeError( 'Cell name not found' )
    
    # match found
    name = match.group( 1 )
    if name == '':
        # empty name
        return None
    
    return name


def get_cell_area( content ):
    """
    Get the cell area from an igor output file
    
    :param content: The content to search
    :returns: The area of the cell in cm^2
    """
    area_search = '"Units.*:(.*?);CM:.*"'
    match = re.search( area_search, content, flags = re.IGNORECASE )
    
    if match is None:
        # no match found, raise exception
        raise RuntimeError( 'Cell area not found' )
        
    return float( match.group( 1 ) )


def get_currents( content ):
    """
    Gets the currents from an Igor file
    
    :param content: The content to search
    :returns: A numpy array of the currents
    """
    j_search = 'WAVES\s*PhotoCurrent1\nBEGIN\n(.*)\nEND'
    match = re.search( j_search, content, flags = ( re.IGNORECASE | re.DOTALL ) )

    if match is None:
        # no match found, raise exception
        raise RuntimeError( 'Currents not found' )
    
    currents = match.group( 1 ).split( '\n' )  
    return [ float( j ) for j in currents ]



def get_voltage_start_step( content ):
    """
    Gets the voltage start point and step used during the sweep
    
    :param content: The content to search
    :returns: A tuple of floats ( start, step )
    """
    volt_search = ' SetScale/P x (.*?),(.*?),"V"'
    match = re.search( volt_search, content, flags = re.IGNORECASE )
    
    if match is None:
        # no match found, raise exception
        raise RuntimeError( 'Voltage start, step not found' )
    
    # match found
    if match.group( 1 ) is None:
        # voltage start not found
        raise RuntimeError( 'Voltage start not found' )
        
    if match.group( 2 ) is None:
        # voltage step not found
        raise RuntimeError( 'Voltage step not found' )
    
    return ( float( match.group( 1 ) ), float( match.group( 2 ) ) )
    
    
    
def create_jv_pairs( content, density = True ):
    """
    Creates JV pairs from an igor output file
    
    :param content: The content to search
    :param density: Divide by cell area to create density, or use raw data [Default: True] 
    :returns: A numpy array with first dimension of voltage, and second dimension of current values
    """
    area = get_cell_area( content ) if density else 1
    ( voltage, step ) = get_voltage_start_step( content )
    currents = get_currents( content )
    
    pairs = []
    for j in currents:
        voltage = round( voltage, 6 ) # round to truncate floating point errors
        pairs.append( ( voltage, j/ area ) ) 
        voltage += step
        
    return pairs


    
def import_datum( file, reindex = True, cell_delim = '-' ):
    """
    Create a DataFrame from a single Igor output file
    
    :param file: The file path to import
    :param reindex: Whether to use the voltage as the index or not [Default: True]
    :param cell_delim: The sample-cell delimeter [Defualt: -]
        Use None if no differentiation between cells on same sample
    :returns: A Pandas DataFrame with columns 'voltage' and 'current'
    """
    with open( file ) as f:
        content = f.read()
        name = get_cell_name( content )
        data = np.array( create_jv_pairs( content ) )
        
    metrics = [ 'voltage', 'current' ]
    if cell_delim is None:
         header = pd.MultiIndex.from_product( [ [ name ], metrics ], names = [ 'sample', 'metrics' ]  )
        
    else:
        name = name.split( cell_delim ) # name-cell
        header = pd.MultiIndex.from_product( [ [ name[ 0 ] ], [ name[ 1 ] ], metrics ], names = [ 'sample', 'cell', 'metrics' ]  )
    
    df = pd.DataFrame( data )
    df.columns = header
    
    
    if reindex:
        df.index = df.xs( 'voltage', level = 'metrics', axis = 1 ).values.flatten()
        df.drop( 'voltage', level = 'metrics', axis = 1, inplace = True )
        df.columns = df.columns.droplevel( 'metrics' )
        
    return df



def import_data( folder_path, file_pattern = '*.sIV', metadata = None, interpolate = 'linear', fillna = 0 ):
    """
    Imports data from Andor output files
    
    :param folder_path: The file path containing the data files
    :param file_pattern: A glob pattern to filter the imported files [Default: '*']
    :param metadata: Metadata from the file name is turned into MultiIndex columns.
        + If list, use standard keywords to include in index [ 'sample', 'power', 'wavelength', 'time' ]
        + If Dictionary, keys indicate level name, value is pattern to match
            + Reseserved key 'standard' can be provided with a list value to get standard metadata
    :param interpolate: How to interpolate data for a common index [Default: linear]
        Use None to prevent reindexing
    :param fillna: Value to fill NaN values with [Default: 0]
    :returns: A Pandas DataFrame with MultiIndexed columns
    """
    
    # get dataframes from files
    df = []
    files = std.get_files( folder_path, file_pattern )
    for file in files:
        data = import_datum( file ) # run local import datum function
        df.append( data )

    if interpolate is not None:
        df = std.common_reindex( df, how = 'interpolate', fillna = fillna, add_values = [ 0 ] )
        
    df = pd.concat( df, axis = 1 )
    return df
    

    
    
def reindex_jv_df( df ):
    """
    Reindexes a Pandas DataFrame with MultiIndexes columns matching those from import_igor_data().
    Uses linear interpolation between points to create a common index for all samples,
    setting the voltage as the index.
    
    :param df: A Pandas DataFrame matching the indexing from import_igor_data()
    :returns: A Pandas DataFrame with linearly interpolated current, and voltage as index 
    """
    pass



    
def group_data( df, groups ):
    """
    Group a DataFrame using the cell names
    
    :param df: A Pandas DataFrame with column indices as those form import_igor_data()
    :param groups: A dictionary with keys of the group names, and iterables of the cell names in that group
    :returns: A Pandas DataFrame with a new top-level column index representing the groups
    """
    pass


# # Work
