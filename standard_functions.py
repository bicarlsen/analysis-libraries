#!/usr/bin/env python
# coding: utf-8

# # Standard Functions
# Common amongst analysis libraries

# # Imports

# In[1]:


import os
import sys
import re
import glob
import math
import inspect

import numpy as np
import pandas as pd

import scipy as sp
import scipy.constants as phys
from scipy.optimize import curve_fit

import matplotlib as mpl
import matplotlib.pyplot as plt


# In[ ]:


def export_df( df, path, name = 'df' ):
    """
    Export a DataFrame as a .csv and .pkl.
    
    :param df: The DataFrame to export.
    :param path: The path to save the files.
    :param name: The name of the file. [Default: 'df']
    """
    path = os.path.join( path, name )
    df.to_csv( path + '.csv' )
    df.to_pickle( path + '.pkl' )


# In[26]:


def file_shape( file, sep = ',' ):
    """
    Returns the shape of the file.
    
    :param file: The file to examine.
    :param sep: The seperator between data. Can be a regular expression. [Default: ,]
    :returns: A tuple of ( rows, columns ), where a column is deined by the given seperator.
    """
    sep = re.compile( sep )
    cols = -1
    with open( file ) as f:
        for rows, line in enumerate( f ):
            r_cols = len( sep.findall( line ) )
            if r_cols > cols:
                cols = r_cols
        
    return ( rows + 1, cols )



def metadata_from_file_name( 
    search, 
    file, 
    is_numeric = False,
    decimal = 'd',
    delimeter = '-', 
    group = 0,
    full_path = False,
    abs_path = False,
    flags = 0
):
    """
    Extract metadata from a file name.
    
    :param search: A RegExp string to search for with one group to extract.
        Delimeters are included automatically.
        For numeric values see the <is_numeric> argument for details on how to format the regExp. 
    :param file: The file name to search in
    :param is_numeric: Is the extracted parameter numeric?
        Numeric values take the form of (\d+)?(<decimal>)?(\d+)? where 
            <decimal> is the decimal argument.
        <search> argument should take the form of a RegExp where the special token 
            '<>' is used to indicate where the numeric part of the pattern lies.
            (e.g. 'ex<>nm' will match ex<number>nm )
        A preceeding 'm' will negate the value. 
        A trailing e will be used as a magnitude, where 'em' is a negative magnitude factor,
            as in scientific notation,
        [Default: False]
        (e.g. 4d3e2, )
    :param decimal: The decimal marker for numeric values. Only matters if <numeric> is True.
        Must be a non-numeric value excluding 'e' and 'm' which are reserved characters.
        [Default: 'd']
    :param delimeter: The delimeter to use between data. [Default: -]
    :param group: The match index to return. If 'all' returns all matches. [Default: 0]
    :param full_path: Use the full file path instead of only the base name. [Default: False]
    :param abs_path: Use the absolute file path instead of only the base name. [Default: False]
    :param flags: Regular expression flags to use when matching. [Default: 0]
    
    :returns: The value of the found value, returned as int or float if numeric, string otherwise
    :raises RuntimeError: If no match is found
    """
    
    # get file name without path or file extension
    if abs_path:
        file = os.path.abspath( file )
        
    elif full_path:
        pass
        
    else:
        file = os.path.basename( file )
        file = os.path.splitext( file )[ 0 ]
    
    # modify search for delimeters and numeric types
    # if numeric, search for preceeding 'm' and trailing exponential
    if is_numeric:
        search = search.replace(
            '<>', 
            '(?<!e)(m)?(\d+)(?:{})?(\d+)?(?:e(m)?(\d+))?'.format( decimal )
        )
        
    
    # use non-matching groups to match hyphen delimeter or beginning or end of string
    search = '(?:^|(?<={}))'.format( delimeter ) + search + '(?={}|$)'.format( delimeter ) 
    
    match = re.findall( search, file, flags )
    if len( match ) == 0:
        # search pattern not found
        raise RuntimeError( 'Metadata not found. Searched for {} in {}'.format( search, file ) )

    if is_numeric:
        # numeric value
        # each element in match is a tuple of ( negative, base, decimal, negative exponent, exponent )
        for index, parts in enumerate( match ):
            # concat base and decimal
            val = '{}.{}'.format( parts[ 1 ], parts[ 2 ] )
            
            if parts[ 0 ]:
                # negative number
                val = '-' + val
            
            if parts[ 4 ]:
                # exponent
                val += 'e'
            
                if parts[ 3 ]:
                    # negative exponent
                    val += '-'
                
                val += parts[ 4 ]
            
        match[ index ] = float( val )
#             if not parts[ 4 ]:
#                 # standard notation
#                 val = float( parts[ 2 ] )

#                 if parts[ 1 ] == decimal:
#                     # decimal
#                     magnitude = math.floor( math.log10( val ) ) + 1
#                     val /= magnitude
#                 else:
#                     # int
#                     val = int( val )

#                 if parts[ 0 ] == 'm':
#                     # negative
#                     val *= -1

#                 match[ index ] = val

#             else:
#                 # scientific notation

#                 # decimal
#                 if parts[ 1 ] == decimal:
#                     # preceeding decimal
#                     val = float( parts[ 3 ] )
#                     magnitude = math.floor( math.log10( val ) ) + 1
#                     val /= magnitude

#                 else:
#                     # decimal in base
#                     try:
#                         dec = parts[ 2 ].find( decimal )

#                     except ValueError:
#                         # zero not in base, interpret as integer
#                         val = int( parts[ 3 ] )

#                     else:
#                         # zero in base, use as decimal
#                         val = parts[ 2 ]
#                         val = val[ :dec ] + '.' + val[ dec + 1: ]
#                         val = float( val )

#                     # multiply by exponent
#                     exp = int( parts[ 4 ] )
#                     if parts[ 3 ] == 'm':
#                         # negative exponent
#                         exp *= -1

#                     val *= 10** exp

#                 match[ index ] = float( val )
                    
    if group == 'all':
        return match
        
    else:
        return match[ group ]
    
    
    
def get_metadata_values( file, metadata ):
    """
    Gets metadata values from a file path.
    
    :param file: The file path to search.
    :param metadata: A dictionary, keys indicate level name, values are patterns to match.
    :returns: A dictionary of metadata values.
    """
    # key is name, value is regexp pattern
    headers = metadata.copy()
    for name, search in metadata.items():
        headers[ name ] = metadata_from_file_name( search, file )

    return headers


# In[6]:


def get_files( folder_path = None, file_pattern = None ):
    """
    Gets files from the specified path

    :param folder_path: The file path containing the data files [Default: Current Working Directory]
    :param file_pattern: A glob pattern to filter the imported files
    :returns: A list of file names
    """

    if folder_path is None:
        folder_path = os.getcwd()
        
    if file_pattern is None:
        file_pattern = '*'
    
    return glob.glob( os.path.join( folder_path, file_pattern ) )


def common_reindex( 
    dfs, 
    index = None, 
    how = 'linear', 
    fillna = 0, 
    add_values = None, 
    name = None,
    threshold = None,
    threshold_type = 'absolute'
):
    """
    Creates a common index across Pandas DataFrames.
    Does not work for MultiIndexed DataFrames
    
    :param dfs: An single or iterable collection of DataFrames
    :param index: The column to use as the index values [Default: index]
    :param how: How to interpolate data at new index values [Default: linear]
    :param fillna: Value to fill NaN values with [Default: 0]
    :param add_values: A list of index values to manually add [Default: None]
    :param name: The index name. If None uses the name of the first DataFrame. [Default: None]
    :returns: A copy of the DataFrames reindexed as prescribed
    """
    if len( dfs ) == 0:
        return
    
    name = dfs[ 0 ].index.name if name is None else name
    
    # set index to given
    if index is not None:
        # TODO: MultiIndexed columns
        dfs = [ df.copy().set_index( index ) for df in dfs ]
        
    # create common index
    combined_index = [ df.index.values for df in dfs ]
    if add_values is not None:
        combined_index.append( add_values )
    
    combined_index = np.unique( np.concatenate( combined_index ) )
    combined_index = pd.Index( combined_index, name = name )
    
    # reindex data
    dfs = [ df.reindex( combined_index ).interpolate( method = how, limit_area = 'inside' ) for df in dfs ]
    
    if fillna is not False:
        dfs = [ df.fillna( fillna ) for df in dfs ]
        
    return dfs



# TODO
def set_index_from_multicolumn( df, key, how = 'linear', fillna = 0, inplace = False ):
    """
    Sets the column from a MultiIndex of a Pandas DataFrame to be the index
    Automatically creates a common index on all found columns and interpoaltes
    
    :param df: The Pandas DataFrame to set the index on
    :param key: The column key to set the new index to
    :param how: How to interpolate data when creating the common index [Default: linear]
    :param fillna: The value to fill NaN values with after interpolation [Default: 0]
    :param inplace: Return a new DataFrame or replace the original [Default: False]
    :returns: A new DataFrame in not inplace, otherwise None
    """
 
    tdf = df if inplace else df.copy()
    
    tdf.index = tdf.xs( key, level = 'metrics', axis = 1 ).values.flatten()
    tdf.drop( 'wavelength', level = 'metrics', axis = 1, inplace = True )
    tdf.columns = tdf.columns.droplevel( 'metrics' )
    
    return ( None if inplace else tdf )



def import_data( 
    import_datum, 
    folder_paths, 
    file_pattern = '*', 
    interpolate = 'linear', 
    fillna = 0,
    **kwargs
):
    """
    Imports data from generic output files
    
    :param import_datum: The function to import a single data file
    :param folder_path: The file path, or list of file paths containing the data files.
    :param file_pattern: A glob pattern to filter the imported files [Default: '*']
    :param interpolate: How to interpolate data for a common index [Default: linear]
        Use None to prevent reindexing
    :param fillna: Value to fill NaN values with [Default: 0]
    :param kwargs: Additional arguments to pass to the import_datum function.
    :returns: A Pandas DataFrame with MultiIndexed columns
    :raises:
    """
    
    # get dataframes from files
    if type( folder_paths ) is str:
        # convert single folder path to list
        folder_paths = [ folder_paths ]
    
    files = []
    for folder in folder_paths:
        files += get_files( folder, file_pattern )
        
    if len( files ) == 0:
        # no files found
        raise RuntimeError( 'No files found.' )
        
    df = []
    for file in files:
        data = import_datum( file, **kwargs ) # run local import datum function
        df.append( data )
        
    if interpolate is not None:
        df = common_reindex( df, how = 'interpolate', fillna = fillna )
        
    df = pd.concat( df, axis = 1 ).sort_index( axis = 1 )
    return df


# In[7]:


def set_plot_defaults():
    """
    Set matplotlib plotting defautls
    """
    
    # set plot format defaults
    mpl.rc( 'font', size = 16 )
    mpl.rc( 'xtick', labelsize = 14 )
    mpl.rc( 'ytick', labelsize = 14 )
    mpl.rc( 'figure', figsize = ( 10, 8 ) )


# In[8]:


def get_level_index( df, level, axis = 0 ):
    """
    Returns the index of the given level name.
    
    :param df: The DataFrame to search.
    :param level: The name of the level.
    :param axis: The index axis to use. [Default: 0]
    :returns: The index of the level.
    """
    names = df.axes[ axis ].names
    return names.index( level )


def keep_levels( df, level = 1, axis = 1, inplace = False ):
    """
    Drops outer levels of a MultiIndex, keeping the inner indices.
    
    :param df: the DataFrame to modify.
    :param level: How many levels to keep. Can be an integer or a level name.
        [Default: 1]
    :param axis: The axis of the index to modify. [Defaut: 1]
    :param inplace: Modify the DataFrame in place. [Default: False]
    :returns: The modified DataFrame.
    :raises: RuntimeError if invalid level is passed.
    """
    if not inplace:
        df = df.copy()
        
    if type( level ) is str:
        # get level index of name
        level = get_level_index( df, level, axis = axis )
        
    if type( axis ) is not str:
        # get axis name
        if axis == 0:
            axis = 'index'
            
        elif axis == 1:
            axis = 'columns'
            
        else:
            # invalid axis
            raise RuntimeError( 'Invalid axis {}.'.format( axis ) )
    
    
    ax = getattr( df, axis )
    levels = len( ax.levels )
    if levels > level:
        new_index = ax.droplevel( list( range( level ) ) )
        setattr( df, axis, new_index )
        
    else:
        raise RuntimeError( 'Invalid level {}'.format( level ) )
        
    return df



def find_level_path( groups, key ):
    """
    Returns the path of a key in a nested dictionary structure with lists as leaves
    
    :param groups: A nested dictionary structure with lists as leaves.
    :param key: The key to search for in the leaves.
    :returns: A list of the path to the found key, or False if not found.
    """
    # base case
    if type( groups ) is list:
        if key in groups:
            # found key
            return []
        
        else:
            return False
        
    # traverse structure
    else:
        for name, child in groups.items():
            path = find_level_path( child, key )
            
            if path is not False:
                # found key
                path.insert( 0, name )
                return path
            
        # did not find key in any children    
        return False
    
    
# TODO
def insert_index_level( df, levels, key_level = 0, axis = 1 ):
    """
    [TODO]
    Insert levels into a MultIndex.
    
    :param df: The DataFrame to modify.
    :param levels: A dictionary of levels
    """
    df.columns = pd.MultiIndex.from_tuples(
        [ ( *levels, *name ) for name in df.columns.values ],
        names = [ *names, *df.columns.names ]
    )
    return df
    
    
    

def add_index_levels( df, groups, names, key_level = 0, axis = 1 ):
    """
    Adds addtional MultiIndex levels to a Pandas DataFrame
    
    :param df: The DataFrame to modify
    :param groups: A nested dictionary with keys as the group name and 
        values a list of current level values in that group.
        Multiple levels can be defined at once using nested dictionaries.
        If None, all current values under key_level are added. 
        [Default: None]
    :param names: A name or list of names for the new levels. [Default: None]
    :param key_level: The level of the current index which the grouping values exist [Default: Top Level]
    :param axis: The axis to group. 0 for index, 1 for columns [Default: 1]
    :returns: The grouped DataFrame
    """
    names = names if ( type( names ) is list ) else [ names ]
    grouped = []
    ax = df.axes[ axis ]
    old_names = ax.names

    if type( key_level ) is str:
        key_level = names.index( key_level )

    for index in ax:
        if type( index ) is tuple:
            key = index[ key_level ]
        
        else:
            key = index
        
        new_index = find_level_path( groups, key )
        if new_index is False: 
            # key not found
            raise RuntimeError( 'Key {} not found in groups'.format( key ) )

        new_index = tuple( new_index )
        new_index += index if ( type( index ) == tuple ) else ( index, )

        data = df.xs( index, axis = axis )
        data = data.rename( new_index )
        grouped.append( data )

    grouped = pd.concat( grouped, axis = 1 )
    grouped.columns = grouped.columns.set_names( names + old_names )
    grouped = grouped.sort_index( axis = axis )
    
    return grouped



def enumerate_duplicate_key( df, level = 0, axis = 1 ):
    """
    If multiple keys are the same in the given index, enumerate them, making them unique
    
    :param df: The Pandas DataFrame to modify
    :param level: If a MultiIndex, which level to examine [Default: Top level]
    :param axis: The axis to examine [Default: 1]
    :returns: A new DataFrame with enumerate indices
    """
    # TODO
    # get duplicates
    indices = [ i for i, n in enumerate( df.columns ) if n == '1ba' ]
    names = df.columns.values
    if len( indices ) > 1:
        # enumerate, starting at 1
        names[ indices[ 1 ] ] = names[ indices[ 1 ] ] + '-{}'.format( index )


# In[9]:


def import_dataframe( path ):
    """
    Creates a Pandas DataFrame from a csv file generated from the import_data() function
    Automatically detects header columns and index
    
    :param path: The path to the saved dataframe
    :returns: A Pandas DataFrame
    """
    
    # get header lines
    with open( path ) as file:
        header_search = '^[^\d]' # stop on digits, indicates data start
        line = file.readline()
        headers = 0
        while line is not None:
            match = re.match( header_search, line )
            if match:
                headers += 1
                line = file.readline()
                
            else:
                # end of headers
                line = None 
                
    headers = list( range( headers ) )
    return pd.read_csv( path, header = headers, index_col = 0 )


# In[10]:


def df_fit_function( fcn, param_names = None, guess = None, modify = None, **kwargs ):
    """
    Returns a function that fits a pandas DataFrame to a function
    
    :param fcn: The function to use for fitting
    :param param_names: Name of the parameters to use in the ultimately returned DataFrame 
        [Default: Names used in the passed function]
    :param guess: A function used to produce the inital parameters guess
        It should accept a Pandas Series containing the data and return a
        tuple of the parameter predictions [Default: All 1]
    :param modify: A function run before the fitting on the DataFrame.
    :param kwargs: Additional parameters to be passed to scipy.optimize.curve_fit()
    :returns: A function that accepts a Pandas DataFrame and fits the data to the provided function.
        The function returns a Pandas DataFrame with the fit parameter and error for each parameter
    """
    param_names = inspect.getfullargspec( fcn ).args[ 1: ] if param_names is None else param_names
    
    header = [ param_names, [ 'value', 'std' ] ]
    header = pd.MultiIndex.from_product( header, names = [ 'parameter', 'metric' ] )
    
    def fitter( df ):
        fits = pd.DataFrame( index = df.columns, columns = header )
        mdf = df if modify is None else modify( df )
        
        for col in mdf:
            data = mdf.xs( col, axis = 1 ).dropna()
            initial = guess( data ) if callable( guess ) else guess
            
            try:
                fit = curve_fit(
                    fcn,
                    xdata = data.index.values,
                    ydata = data.values,
                    p0 = initial,
                    **kwargs
                )
                
            except RuntimeError as err:
                print( col, err )
                continue

            # create dictionaries of parameter values and standard deviations 
            params = dict( zip( 
                [ ( param, 'value' ) for param in param_names ], 
                 fit[ 0 ] 
            ) )
            
            stds = dict( zip(
                [ ( param , 'std' ) for param in param_names ],
                np.sqrt( fit[ 1 ].diagonal() )
            ) )
            
            params.update( stds )
            fit = pd.Series( params, name = col )
            fits.loc[ col ] = fit
            
        return fits
    
    return fitter


def gradient_threshold( 
    df, 
    div = 'slope', 
    threshold = -1, 
    calc = 'error',
    window = 5
):
    """
    Thresholds data based on the local curvature.
    
    :param df: The DataFrame to threshold.
    :param div: The type of derivative to examine. Use 'slope' or 'curvature'. [Default: slope]
    :param threshold: [Default: -1]
    :param calc: The type of threshold to use. 
        Use 'absolute' or 'error'. [Default: error]
    :param window: Window width to calculate average gradient for error calculation.
        [Default: 5]
    :returns: Thresholded DataFrame
    """
    def compute_grads( df ):
        diffs = df.diff() 
    
        # calculate x-axis differences
        runs = diffs.index.values
        runs = np.reshape( runs, ( runs.shape[ 0 ], 1 ) )
        runs = np.repeat( runs, diffs.shape[ 1 ] , axis = 1 )
        grads = diffs/ runs
        
        return grads
    
    
    grads = compute_grads( df )
    
    if div == 'curvature':
        grads = compute_grads( grads )
    
    if calc == 'error':
        # compute error threshold
        threshold = threshold* grads.rolling( window = window ).mean().abs()
        grads = grads.abs()
        print( threshold )
        df = df.where( grads < threshold )
    
    else:
        # absolute threshold
        if threshold > 0:
            df = df.where( grads < threshold )

        if threshold < 0:
            df = df.where( grads > threshold )
        
    df = df.dropna()
    df = df.reset_index( drop = True )
    return df



def break_and_align( 
    df, 
    index = None, 
    threshold = 10, 
    level = None, 
    include_index = None,
    align_col = None
):
    """
    Aligns breaks in the index or column exceeding threshold, creating cycles.
    Indices with all missing data in each group are removed before splitting.
    
    E.g.
    If an index has values [ 1, 2, 3, 6, 7, 9, 12, 13, 15 ] with a threshold of 2
    The data would be broken after 3 and 9, creating three cycles:
    [ [ 1, 2, 3 ], [ 6, 7, 9 ], [ 12, 13, 15 ] ]
    
    :param df: DataFrame to align.
    :param index: The data to use as the index for splitting data.
        For MultiIndex, must pass full column name in reference to level.
        If None, uses the index. [Default: None]
    :param threshold: The minimum value of a break in index values to create a new cycle.
        [Default: 10]
    :param level: The level to group data. If None treats each column individually.
        [Default: None]
    :include_index: Include index values as column with given name.
        If None, index is called 'index'.
        Only used if index parameter is None.
        [Default: None]
    :align_col: Name of the aligned column. 
        This will be the index values of each cycle shifted to start at 0.
        If None, do not compute. [Default: None]
    :returns: DataFrame with new column level 'cycles' of aligned data.
    """ 
    df = df.copy()
    
    # column level names for re-aligned data
    level_names = [ name for name in df.columns.names ]
    level_names.insert( 1, 'cycle' )
    
    if index is None:
        if type( include_index ) is str:
            df.index = df.index.rename( include_index )

        index_col = include_index
      
    cycles = []  
    groups = df.items() if ( level is None ) else df.groupby( level = level, axis = 1 )
    for name, data in groups:
        if type( data ) is pd.Series:
            data = pd.DataFrame( data )
        
        data = data.dropna( how = 'all' )
        data_index = data.index if ( index is None ) else data.loc[ :, index_col ]
        
        breaks = np.where( # non continuous index values, before break
            np.diff( data_index.values ) >= threshold 
        )[ 0 ] 

        pib = 0 # previous index break
        for i in range( len( breaks ) ):
            index_name = ( *name[ :-1 ], index_col )
            
            ib = breaks[ i ] + 1; # index break, including break point
            cycle = data.iloc[ pib : ib ].copy() # cycle data
            if index is None:
                # move index values into data
                cycle[ index_name ] = cycle.index
            
            if align_col is not None:
                align_index = ( *name[ :-1 ], align_col )
                
                cycle_start = cycle.iloc[ 0 ][ index_name ]
                cycle[ align_index ] =  cycle.loc[ :, index_name ].copy() - cycle_start
                
            # set cycle columns
            header = [ ( head[ 0 ], i, *head[ 1: ] ) for head in cycle.columns.values ]
            header = pd.MultiIndex.from_tuples( header, names = level_names )
            cycle.columns = header
            
            cycle.reset_index( drop = True, inplace = True )
            cycles.append( cycle )
            pib = ib

    cycles =  pd.concat( cycles, axis = 1 )
    return cycles




#-------------------------------------- TODO ------------------------------------
# def furcate( df, model = std.gaussian_distribution, bins = 2, guess = None ):
#     """
#     Finds values to split the data at, creating a given number of segments.
#     Splitting values are found by creating a histogram of the data,
#         fitting (bins - 1) models to the histogram, and choosing the
#         intersection point of adjacent models as the splitting point.
#     Note: Automatic guessing only works with 2 parameter models, currently 
    
#     :param df: A one-dimensional data array.
#     :param model: The PDF model to fit each data bin with. 
#         First parameter is center, second is spread. 
#         [Default: Gaussian]
#     :param bins: The number of bins to create.
#     :param guess: A function that takes in the data and outputs inital parameter guess. 
#         If None, centers are distributed equally along range, and
#         spreads are the width of the bin.
#         [Default: None]
#     :returns: A list of (bins - 1) splitting points.
#     :raises RuntimeError: If guess is None and the number of parameters for the model
#         is not 2.
#     """
#     df = df/ df.max()
#     ( counts, edges ) = np.histogram( df, 100* bins )
#     counts = counts/ counts.max()
    
#     # create inital guess if not provided
#     if guess is None:
#         if len( signature( model ).parameters ) != 2:
#             raise RuntimeError( 'Can not use automatic guessing for models without 2 parameters.' )
        
#         centers = [ np.mean( edge[ i: i + 2 ] ) for i in range( edges.shape[ 0 ] - 1 ) ]
#         spreads = [ ( df.max() - df.min() )/ bins ]* bins
#         guess = list( zip( centers, spread ) )
        
#     # fit functions
    
        
    
#     return splits


# In[11]:


def idxnearest( df, val ):
    """
    Gets the index of the nearest value
    
    :param df: The Pandas DataFrame to search
    :param val: The value to match
    :returns: The index value of the nearest value
    """
    return abs( df - val ).idxmin()
    


# In[12]:


def wl_to_en( l ):
    """
    Converts a wavelength, given in nm, to an energy in eV
    
    :param l: The wavelength to convert, in nm
    :returns: The corresponding energy in eV
    """
    a = phys.physical_constants[ 'electron volt-joule relationship' ][ 0 ] # J
    return phys.Planck* phys.c/( a* l* 1e-9 )


def en_to_wl( e ):
    """
    Converts an energy, given in eV, to a wavelength
    
    :param e: The energy to convert, in eV
    :returns: The corresponding wavelength in nm
    """
    a = phys.physical_constants[ 'electron volt-joule relationship' ][ 0 ] # J
    return 1e9* phys.Planck* phys.c/( a* e )


def gaussian_distribution( mu = 0 , sigma = 1, x = None ):
    """
    A Normal or Gaussian distribution
    
    :param mu: The mean, or None [Default: 0]
    :param sigma: The standard deviation, or None [Default: None]
    """
    return np.exp( -np.power( ( x - mu ), 2 )/( 2* np.power( sigma, 2 ) ) )


def boltzmann_distribution( t = 300, e = None ):
    """
    The Boltzmann distribution at a given temperature
    
    :param t: The temperature in Kelvin, or None [Default: 300]
    :param e: The energy in eV, or None [Default: None]
    :returns: Returns a function describing the Boltmann distribution
        as a function of energy and or temperature, if either are None;
        or the value if both are not None
    """
    a = phys.physical_constants[ 'electron volt-joule relationship' ][ 0 ] # J
    k = phys.Boltzmann/ a
    
    if ( T is None ) and ( E is None ):
        # function of energy and temperature
        boltzmann = lambda E, T: np.exp( -E/( k* T ) )
        
    elif ( T is None ) and ( E is not None ):
        # function of temperature only
        boltzmann = lambda T: np.exp( -e/( k* T ) )
        
    elif ( T is not None ) and ( E is None ):
        # function of energy only
        boltzmann = lambda E: np.exp( -E/( k* t ) )
    
    else:
        # both values passed, return value
        boltzmann = np.exp( -e/( k* t ) )
    
    return boltzmann


def fermi_distribution( ef = 0, t = 300, e = None ):
    """
    The Fermi distribution at a given temperature and Fermi energy 
    
    :param ef: The Fermi energy of the system in eV, or None [Default: 0]
    :param t: The temperature at which to calculate the distribution in K, or None [Default: 300]
    :param e: The energy at whcih to calculate the distribution in eV, or None [Default: None]
    :returns: A function representing the Fermi distribution taking temperature and or energy
        as parameters, or returning a value if both are specified
    """
    a = phys.physical_constants[ 'electron volt-joule relationship' ][ 0 ] # J
    k = phys.Boltzmann/ a
    
    if ( ef is None ) and ( t is None ) and ( e is None ):
        # function of Ef, T, and E
        fermi = lambda Ef, T, E: np.power( 1 + np.exp( ( E - Ef )/( k* T ) ), -1 )
    
    elif ( ef is None ) and ( t is None ) and ( e is not None ):
        # function of Ef and T
        fermi = lambda Ef, T: np.power( 1 + np.exp( ( e - Ef )/( k* T ) ), -1 )
        
    elif ( ef is None ) and ( t is not None ) and ( e is None ):
        # function of Ef and E
        fermi = lambda Ef, E: np.power( 1 + np.exp( ( E - Ef )/( k* t ) ), -1 )
    
    elif ( ef is None ) and ( t is not None ) and ( e is not None ):
        # function of Ef
        fermi = lambda Ef: np.power( 1 + np.exp( ( e - Ef )/( k* t ) ), -1 )
    
    elif ( ef is not None ) and ( t is None ) and ( e is None ):
        # function of E and T
        fermi = lambda T, E: np.power( 1 + np.exp( ( E - Ef )/( k* T ) ), -1 )
        
    elif ( ef is not None ) and ( t is None ) and ( e is not None ):
        # function of T
        fermi = lambda T: np.power( 1 + np.exp( ( e - Ef )/( k* T ) ), -1 )
        
    elif ( ef is not None ) and ( t is not None ) and ( e is None ):
        # function of E
        fermi = lambda E: np.power( 1 + np.exp( ( E - Ef )/( k* t ) ), -1 )
    
    else:
        # value
        fermi = n.power( 1 + np.exp( ( e - Ef )/( k* t ) ), -1 )
    
    return fermi


def dos( e0 = 0, e = None ):
    """
    The density of states
    
    :param e0: An energy in eV, or None [Default: 0]
    :param e: An energy in eV, or None [Default: None]
    :returns: A function that takes in an energy and reference energy and 
        returns the density of states function or a value
    """
    
    if ( e0 is None ) and ( e is None ):
        # function of e and e0
        density = lambda E, E0: np.sqrt( E - E0 )
            
    elif ( e0 is None ) and ( e is not None ):
        # function of e0
        density = lambda E0: np.sqrt( e - E0 )
        
    elif ( e0 is not None ) and ( e is None ):
        # function of e
        density = lambda E: np.sqrt( E - e0 )
        
    else:
        # value
        density = np.sqrt( e - e0 )
        
    return density
    
# TODO: Multiply functions together
def population( ef = 0, t = 300, e0 = 0, e = None ):
    """
    Returns a function or value representing the distribution of the population at a given energy
        
    :param ef: The Fermi energy in eV, or None [Default: 0]
    :param t: The temperature in K, or None [Default: 300]
    :param e0: The density of states base energy in eV, or None [ Default: 0 ]
    :param e: The energy to evaluate the system at, or None [Default: None]
    :returns: A function representing the population that takes as parameters 
        the values passed as None.
        If all values are specified, returns the value directly.
    """
    
    return fermi_distribution( ef, t, e )* boltzmann_distribution( t, e )* dos( e0, e )
    


# In[13]:


def plot_levels( plot, df, show = True, level = 'metrics', axis = 1, **fig_args ):
    """
    Plots each element of a Pandas DataFrame in a separate subplot.
    
    :param plot: A function that receives a Pandas DataSeries and axis to plot it on ( ax, data, name ).
    :param df: The DataFrame to plot.
    :param show: Show the plot. [Defualt: True]
    :param level: Which level to iterate over. [Default: metrics]
    :param axis: The axis to iterate over. [Default: 'columns']
    :param fig_args: Keyword arguments passed to plt.subplot().
    :returns: The Figure and Axes of the plot as a tuple ( fig, axs ).
    """
    if axis == 'rows':
        axis = 0 
        
    elif axis == 'columns':
        axis = 1
        
    ax = df.axes[ axis ]
    
    levels = list( range( ax.names.index( level ) + 1 ) )
    groups = df.groupby( level = levels, axis = axis )
    
    num_plots = len( groups )
    cols = int( np.floor( np.sqrt( num_plots ) ) )
    rows = int( np.ceil( num_plots/ cols ) )
    fig, axs = plt.subplots( rows, cols, **fig_args )
    index = 0
    
    for name, data in groups:
        row = int( np.floor( index/ cols ) )
        col = int( index% cols )
        
        if len( axs.shape ) == 1:
            # only rows
            ax = ax[ row ]
            
        else:
            # rows and cols
            ax = axs[ row, col ]
        
        plot( ax, data, name )
        index += 1
        
    fig.tight_layout()
    
    if show:
        plt.show()
    
    return ( fig, axs )


def plot_df( plot, df, show = True, **fig_args ):
    """
    Plots each element of a Pandas DataFrame in a separate subplot.
    
    :param plot: A function that receives a Pandas DataSeries and axis to plot it on ( ax, data, name ).
    :param df: The DataFrame to plot.
    :param show: Show the plot. [Defualt: True]
    :param fig_args: Keyword arguments passed to plt.subplot().
    :returns: The Figure and Axes of the plot as a tuple ( fig, axs ).
    """
    num_plots = int( df.columns.shape[ 0 ] )
    cols = int( np.floor( np.sqrt( num_plots ) ) )
    rows = int( np.ceil( num_plots/ cols ) )
    fig, axs = plt.subplots( rows, cols, **fig_args )
    index = 0
    
    for name, data in df.items():
        row = int( np.floor( index/ cols ) )
        col = int( index% cols )
        
        if len( axs.shape ) == 1:
            # only rows
            ax = ax[ row ]
            
        else:
            # rows and cols
            ax = axs[ row, col ]
        
        plot( ax, data, name )
        index += 1
        
    fig.tight_layout()
    
    if show:
        plt.show()
    
    else:
        return ( fig, axs )



def boxplot_groups( df, groups, total = True, show = True ):
    """
    Creates a box plot of a grouped Pandas Series
    
    :param df: A Pandas Series containing the data to be plotted
    :param groups: A single or list of index levels to group by
    :param total: Whether to include a plot for all data [Default: True]
    :param show: Whether to show the plot or return the axis [Default: True]
    :returns: None if show is True, else the matplotlib Axis it is plotted on
    """
    fig, axs = plt.subplots()
    data = [ df.values ] if total else []
    labels = [ 'All' ] if total else []
    for name, group in df.groupby( groups ):
        labels.append( name )
        data.append( group.values )

    axs.boxplot( data, labels = labels )
    plt.xticks( rotation = 70 )
    
    if show:
        plt.show()
        
    else:
        return axs


# # Work

# In[ ]:




