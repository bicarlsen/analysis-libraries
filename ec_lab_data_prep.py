
# coding: utf-8

# # EC Lab Data Prep

# ### Imports

# In[1]:


import os
import sys
import re
import glob
import math
import logging

from collections import OrderedDict

import numpy as np
import pandas as pd

import import_ipynb
import standard_functions as std


# In[2]:


from importlib import reload


# In[7]:


reload( std )


# ## Data Prep

# In[3]:


# helper functions

def get_channel_from_file_name( file, prefix = 'C' ):
    """
    :param prefix: The prefix of the channel. [Default: 'C']
    """
    channel_search = '{}(\d+)'.format( prefix )
    return std.metadata_from_file_name( 
        channel_search, 
        file, 
        is_numeric = True, 
        delimeter = '',
        abs_path = True
    )


def get_holder_from_file_name( file ):
    holder_search = 'holder\-(\d+)'
    return std.metadata_from_file_name( 
        holder_search, 
        file, 
        is_numeric = True, 
        delimeter = '',
        abs_path = True
    )


def get_program_from_file_name( file ):
    program_search = '(\d+)'
    return std.metadata_from_file_name(
        program_search,
        file,
        is_numeric = True,
        delimeter = '_'
    )


def metadata_from_file_name( file, metadata, channel_prefix = 'C' ):
    """
   Returns a dictionary of metadata values
    
    :param metadata: A list of keywords of metadata to use for indexing. 
        Values: [ 'holder', 'channel', 'program' ]
        Default: [ 'holder', 'channel', 'program' ]
    :param channel_prefix: The prefix for matching channel metadata. [Defualt: 'C']
    :return: A dictionary of metadata values
    """
    # create multiindex from metadata
    header_vals  = {}

    if 'channel' in metadata:
        header_vals[ 'channel' ] = int( get_channel_from_file_name( file, prefix = channel_prefix ) )

    if 'holder' in metadata:
        header_vals[ 'holder' ] = int( get_holder_from_file_name( file ) )
        
    if 'program' in metadata:
        header_vals[ 'program' ] = int( get_program_from_file_name( file ) )

    return header_vals


def create_column_index( df, metadata, file, channel_prefix = 'C' ):
    """
    Creates a MultiIndex containing metadata values for a single data file.
    
    :param df: The DataFrame that will use the MultiIndex.
    :param metadata: A list of metadata keyword to use, or name-pattern dictionary pairs.
    :param file: The file name containing the data.
    :param channel_prefix: The prefix for matching channel metadata. [Defualt: 'C']
    :returns: A MultiIndex representing the file metadata.
    """
    metrics = df.columns.values
        
    header_names = metadata.copy()
    header_names.append( 'metrics' )
    header_vals = metadata_from_file_name( file, metadata )
    
    header = [ [ header_vals[ val ] ] for val in metadata ]
    header.append( metrics )
    header = pd.MultiIndex.from_product( header, names = header_names )
    
    return header


# In[4]:


def import_jv_datum( 
    file, 
    precision = 1e-4, 
    reindex = True, 
    invert_current = True, 
    ret_bins = False, 
    metadata = [ 'holder', 'channel' ]
):
    """
    Imports JV data from a .use file.
    Due to the large file size and small voltage step 
        resoluion data can be grouped and averaged over.
    
    :param file: The file name holding the data.
    :param precision: The resolution of data bins. 
        If two voltage measurements agree up to this 
        precision they will be binned together. 
        None to keep all measurements.
        [Default: 1e-4]
    :param reindex: Use votlage as index. [Default: True]
    :param invert_current: Flips the sign of current readings. [Default: True]     
    :param ret_bins: Return the voltage bins. [Default: False]
    :param metadata: A list of keywords of metadata to use for indexing. 
        Values: [ 'holder', 'channel', 'program' ]
        Default: [ 'holder', 'channel' ]
    :returns: A Pandas DataFrame representing the data.
    :returns: Votage bins, if ret_bins is True
    """    
    # find header lines
    header_pattern = 'number of E\(V\), I\(A\) couples: (\d+)'
    header_lines = 0
    with open( file ) as f:
        for num, line in enumerate( f ):    
            match = re.match( header_pattern, line )
            if match is not None:
                # found header line
                header_lines = num + 1
                break
      
    # read data
    df = pd.read_csv( file, names = [ 'voltage', 'current' ], skiprows = header_lines )    
    
    if invert_current:
        df.current *= -1
    
    # average data
    if precision is not None:
        v_min = df.voltage.min()
        v_max = df.voltage.max()
        bins = math.ceil( (v_max - v_min )/ precision )
        bins = np.linspace( v_min, v_max, bins  )
        
        cuts = pd.cut( df.voltage, bins, include_lowest = True )
        df = df.groupby( cuts, as_index = False ).mean()
        df = df.dropna( how = 'all' ) # remove empty bins
        
        # remove insignificant figures
        insig = resolution/ 10            
        df.index = pd.IntervalIndex( df.index ).mid
        df.index = pd.Index( 
            df.index.to_series().apply( lambda x: np.round( x, int( -np.log10( insig ) ) ) ),
            name = 'time'
        )
    
    else:
        bins = None
        
    # reindex
    if reindex:
        df = df.set_index( 'voltage' )
        
    # metadata
    if metadata is not None:
        if type( metadata ) is list:
            cci = create_column_index # needed for encapsulation
            header = cci( df, metadata, file, channel_prefix = channel_prefix )    
            
        else:
            header = std.get_metadata_values( file, metadata, channel_prefix = channel_prefix )
            header = pd.MultiIndex( header )
            
        df.columns = header
    
    if ret_bins:
        return ( df, bins )
    
    else:
        return df
    
    
def import_datum( 
    file, 
    use_cols = [ 'time', 'voltage' ], 
    reindex = None,
    metadata = [ 'holder', 'channel' ],
    channel_prefix = 'C'
):
    """
    Imports measurement data from a .mpt file.
    Due to the large file size and small voltage step 
        resoluion data can be grouped and averaged over.
    
    :param file: The file name holding the data.
    :param use_cols: The name of the columns to include in the dataset.
        Values are: [ 'mode', 'ox/red', 'error', 'control changes', 'Ns changes',
        'counter inc', 'time', 'control',  'voltage', 'current',  'dq', 'dQ',
        '(Q-Qo)', 'Q charge/discharge/', 'half cycle', 'energy charge',
        'energy discharge', 'capacitance charge', 'capacitance discharge',
        'Q discharge', 'Q charge', 'capacity', 'efficiency', 'cycle number',
        'P/W' ]
    :param reindex: The column name to use as an index. Values are the 
        same as in use_cols, and must be included in use_cols. 
        [Default: None]
    :param metadata: A list of keywords of metadata to use for indexing. 
        Values: [ 'holder', 'channel', 'program' ]
        Default: [ 'holder', 'channel' ]
    :param channel_prefix: Prefix indicating the channel in the filename. [Default: 'C']
    :returns: A Pandas DataFrame representing the data.
    :raises RuntimeError: If an invalid column is included in use_cols.
    :raises RunTimeError: If reindex column is not included in use_cols.
    """
    encoding = 'unicode_escape'
    
    # find header lines
    header_pattern = 'Nb header lines : (\d+)'
    header_lines = None
    with open( file, encoding = encoding ) as f:
        for line in f:    
            match = re.match( header_pattern, line )
            if match is not None:
                # found header line
                header_lines = int( match.group( 1 ) )
                break
      
    # read data
    columns = OrderedDict( {
        'mode':                     'mode',
        'ox/red':                   'ox/red',
        'error':                    'error',
        'control changes':          'control changes',
        'Ns changes':               'Ns changes',
        'counter inc.':             'counter inc',
        'time/s':                   'time',
        'control/mA':               'control',
        '<Ewe>/V':                  'voltage',
        'I/mA':                     'current',
        'dq/mA.h':                  'dq',
        'dQ/mA.h':                  'dQ',
        '(Q-Qo)/mA.h':              '(Q-Qo)',
        'Q charge/discharge/mA.h':  'Q charge/discharge/',
        'half cycle':               'half cycle',
        'Energy charge/W.h':        'energy charge',
        'Energy discharge/W.h':     'energy discharge',
        'Capacitance charge/µF':    'capacitance charge',
        'Capacitance discharge/µF': 'capacitance discharge',
        'Q discharge/mA.h':         'Q discharge',
        'Q charge/mA.h':            'Q charge',
        'Capacity/mA.h':            'capacity',
        'Efficiency/%':             'efficiency',
        'cycle number':             'cycle number',
        'P/W':                      'P/W'
    } )
    
    # check all use_cols are valid
    for col in use_cols:
        if not col in columns.values():
            raise RuntimeError( 'Invalid column {}.'.format( col ) )
 
    df = pd.read_csv( 
        file, 
        sep = '\t', 
        names = columns.values(), 
        skiprows = header_lines, 
        usecols = use_cols 
    )    
    
    if reindex is not None:
        if reindex in use_cols:
            df = df.set_index( reindex )
            
        else:
            # reindex column not included in data
            raise RuntimeError( 'Invalid index value {}, must be included in columns.'.format( reindex ) )
    
    # metadata
    if metadata is not None:
        if type( metadata ) is list:
            cci = create_column_index # needed for encapsulation
            header = cci( df, metadata, file, channel_prefix = channel_prefix )    
            
        else:
            header = std.get_metadata_values( file, metadata, channel_prefix = channel_prefix )
            header = pd.MultiIndex( header )
            
        df.columns = header
    
    return df  

        
        
def import_jv_data( 
    folder_path, 
    file_pattern = '*.use', 
    interpolate = 'linear', 
    fillna = np.nan,
    reindex = 'voltage',
    **kwargs
):
    """
    Imports JV data from EC Lab .use output files.
    
    :param folder_path: The file path containing the data files.
    :param file_pattern: A glob pattern to filter the imported files [Default: '*.use']
    :param interpolate: How to interpolate data for a common index [Default: linear]
        Use None to prevent reindexing
    :param fillna: Value to fill NaN values with [Default: np.nan]
    :param kwargs: Additional parameters to pass to import_datum.
    :returns: A Pandas DataFrame with MultiIndexed columns
    :raises RuntimeError: if no files are found
    """
    df = std.import_data( 
        import_jv_datum, 
        folder_path, 
        file_pattern = file_pattern, 
        interpolate = interpolate, 
        fillna = fillna,
        reindex = reindex,
        **kwargs
    )
    
    return df.sort_index( axis = 1 )
    
    
def import_data( 
    folder_path, 
    file_pattern = '*.mpt', 
    interpolate = 'linear', 
    fillna = np.nan,
    programs = False,
    resolution = None,
    **kwargs
):
    """
    Imports measurement data from EC Lab .mpt output files.
    
    :param folder_path: The file path containing the data files.
    :param file_pattern: A glob pattern to filter the imported files [Default: '*.use']
    :param interpolate: How to interpolate data for a common index [Default: linear]
        Use None to prevent reindexing
    :param fillna: Value to fill NaN values with [Default: np.nan]
    :param programs: Concatenates programs. [Default: False] 
    :param resolution: The time resolution to use, in same units as measured time (usually seconds). 
        If None, does not change time step. [Default: None]
    :param kwargs: Additional parameters to pass to import_datum.
    :returns: A Pandas DataFrame with MultiIndexed columns
    :raises RuntimeError: if no files are found
    """
    if programs:
        if 'metadata' in kwargs:
            if 'program' not in kwargs[ 'metadata' ]:
                kwargs[ 'metadata' ].append( 'program' )
            
        else:
            kwargs[ 'metadata' ] = [ 'program' ]
    
    df = std.import_data( 
        import_datum, 
        folder_path, 
        file_pattern = file_pattern, 
        interpolate = None, 
        fillna = fillna,
        **kwargs
    )
    
    # merge programs
    if programs:
        tdf = []
        
        # group data by levels above program
        program_level = df.columns.names.index( 'program' )
        group_levels = list( range( program_level ) )
        program_groups = (
            df.groupby( level = group_levels, axis = 1 ) 
            if len( group_levels ) > 0
            else [ ( '', df ) ]
        )
        
        for name, data in program_groups:
            # for each group of programs concatenate them along the 0 axis
            pdf = []
            time_header = ( *name, 'time' ) if ( type( name ) is tuple ) else 'time' 
    
            for pname, pdata in data.groupby( level = 'program', axis = 1 ):
                pdata.columns = pdata.columns.droplevel( 'program' )
                pdata = pdata.dropna()
                pdata.set_index( time_header, inplace = True )
                pdata.index = pdata.index.rename( 'time' )
                pdf.append( pdata )

            pdf = pd.concat( pdf )    
            tdf.append( pdf )
        
        df = pd.concat( tdf, axis = 1 ).interpolate( interpolate ).sort_index( axis = 1 )
    
        # average data
        if resolution is not None:
            t_min = df.index.min()
            t_max = df.index.max()
            bins = math.ceil( (t_max - t_min )/ resolution )
            bins = np.linspace( t_min, t_max, bins  )

            times = df.index.to_series()
            cuts = pd.cut( times, bins, include_lowest = True )
            df = df.groupby( cuts, as_index = True ).mean()
            df = df.dropna( how = 'all' ) # remove empty bins

            # remove insignificant figures
            insig = resolution/ 10            
            df.index = pd.IntervalIndex( df.index ).mid
            df.index = pd.Index( 
                df.index.to_series().apply( lambda x: np.round( x, int( -np.log10( insig ) ) ) ),
                name = 'time'
            )
            
    return df


# In[ ]:


def density_bifurcate( df, threshold = 1e-2, divs = 10 ):
    """
    Takes all data above the first thresholded values.
    
    :param df: The DataFrame to calculate a threshold value for.
    :param threshold: Threshold value relative to most populated state. [Default: 0.01]
    :param divs: Number of divisions in the histogram. [Default: 10]
    """
    ( counts, edges ) = np.histogram( df, divs )
    counts = counts/ counts.max()
    centers = np.mean( [ edges[ 1: ], edges[ :-1 ] ], axis = 0 )
    
    # mask values below, find indices below first values above threshold
    below = np.where( counts < threshold )[ 0 ]
    
    if ( divs - 1 ) in below:
        # remove indices at end
        index = 0
        cont = True
        prev = divs
        while cont:
            index -= 1
            cont = ( below[ index ] == prev - 1 ) # continuous
            
        below = below[ :index ]
        
    # no data exceeding threshold
    if len( below ) == 0:
        logging.warning( 'No data exceeding threshold.' )
        return None
    
    return centers[ below[ -1 ] ]


def threshold( df, threshold = density_bifurcate ):
    """
    Thresholds a data set, keeping values above the threshold.
    
    :param df: The DataFrame to threshold.
    :param threshold: A function to calculate the threshold values. [Default: density_bifurcate]
    :returns: A new DataFrame with all values below threshold removed.
    """
    # iterate over levels except metrics
    groups = data.columns.names
    groups = groups[ :-1 ] if ( 'metrics' in groups ) else groups
    
    td = []
    for name, data in df.groupby( level = groups, axis = 1 ):
        data = data.copy()
        voltage = data.xs( 'voltage', axis = 1, level = 'metrics', drop_level = False )

        threshold = threshold( voltage.dropna().values, 1e-3 )
        td_th[ name ] = threshold
        voltage = voltage.where( voltage > threshold )  
        data.loc[ :, ( *name, 'voltage' ) ] = voltage.values.reshape( voltage.shape[ 0 ] )
        td.append( data )

    td = pd.concat( td, axis = 1 )
    return td


# In[1]:


def split_cycle_data( df, split_time = 10, interpolate = 'linear' ):
    """
    Split data into cycles if a time break of longer than split_time occurs
    """ 
    names = df.columns.names
    df = df.dropna()
    
    # remove extra levels
    if type( df.columns ) is pd.MultiIndex:
        levels = len( df.columns.levels )
        if levels > 1:
            df.columns = df.columns.droplevel( list( range( levels - 1 ) ) )
        
    breaks = np.where( np.diff( df.index.values ) > split_time )[ 0 ] # non continuous index values, before break
    
    cycles = []
    prev = 0
    for i in range( len( breaks ) ):
        ib = breaks[ i ] + 1; 
        cycle = df.iloc[ prev:ib ].reset_index()
        min_time = cycle.iloc[ 0 ].time
        cycle = cycle.assign( rel_time = lambda x: x.time - min_time )
        
        cycle.columns = pd.MultiIndex.from_product( 
            [ [ channel ], [ i ], [ 'time', 'voltage', 'rel_time' ] ], 
            names = names
        )
        
        cycle.index = pd.Float64Index( cycle.index )
        cycles.append( cycle )
        prev = ib
        
    return pd.concat( cycles, axis = 1 )


        
def split_cycles( df, split_time = 10, level = 'channel' ):
    """
    Splits data by cycle.
    """
    groups = ( df.groupby( level = level, axis = 1 )
        if ( type( df.columns ) is pd.MultiIndex )
        else df.items() )
    
    split = []
    for name, data in groups:
        sdf = split_cycle_data( data )
        split.append( sdf )
        
    return pd.concat( split, axis = 1 ).sort_index( axis = 1 )


# In[2]:


def gradient_threshold( data, div = 'slope', threshold = -1, calc = 'error' ):
    """
    Thresholds data based on the local curvature.
    
    :param data:
    :param div: The type of derivative to examine. Use 'slope' or 'curvature'. [Default: slope]
    :param threshold: [Default: -1]
    :param calc: The type of threshold to use. 
        Use 'absolute' or 'error'. [Default: error]
    :returns: Thresholded data
    """
    diffs = data.diff()
    metric_level = data.columns.names.index( 'metrics' )
    for i in range( metric_level ):
        diffs.columns = diffs.columns.droplevel( 0 )
    
    grads = diffs.voltage/ diffs.rel_time
    
    if div == 'curvature':
        diffs = diffs.diff()
        grads = diffs.voltage/ diffs.rel_time
    
    if calc == 'error':
        # compute error threshold
        threshold = threshold* grads.rolling( window = 5 ).mean().abs()
        grads = grads.abs()
        
        data = data.where( grads < threshold )
    
    else:
        # absolute threshold
        if threshold > 0:
            data = data.where( grads < threshold )

        if threshold < 0:
            data = data.where( grads > threshold )
        
    data = data.dropna()
    data = data.reset_index( drop = True )
    return data


def clean_gradients( df, threshold = 1 ):
    clean = []
    group_levels = list( range( df.columns.names.index( 'metrics' ) ) )
    for name, data in df.groupby( level = group_levels, axis = 1 ):
        cd = gradient_threshold( data, threshold = threshold, calc = 'error' ).dropna()
        if cd.shape[ 0 ] > 0:
            # only append columns with valid voltage data
            clean.append( cd )
        
    return pd.concat( clean, axis = 1 )


# # Work

# In[36]:


# data_path = 'data/temp/holder-01/ch1/'
# jv_path = 'data/holder-01/ch1/1-1-1-3temp-dep_06_CV_C16.use'
# df = import_data( data_path, programs = True )
# jv_df = import_jv_datum( jv_path, precision = 1e-4 )

