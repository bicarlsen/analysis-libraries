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
import logging
import inspect
import warnings

import numpy as np
import pandas as pd

import scipy as sp
import scipy.constants as phys
from scipy.optimize import curve_fit

import matplotlib as mpl
import matplotlib.pyplot as plt


# In[2]:


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
    

def save_figure( path, kind = 'png', fig = None ):
    """
    Save a figure.
    
    :param path: Path to save file.
    :param kind: Format to save file. [Default: 'png']
    :param fig: Figure to save. If None, saves current figure.
        [Default: None]
    """
    
    if fig is None:
        fig = plt.gcf()
    
    fig.savefig( path, format = kind, bbox_inches = 'tight' )


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
    
    :param search: A RegEx string to search for with one group to extract.
        Delimeters are included automatically.
        For numeric values see the <is_numeric> argument for details on how to format the RegEx. 
    :param file: The file name to search in
    :param is_numeric: Is the extracted parameter numeric?
        Numeric values take the form of (\d+)?(<decimal>)?(\d+)? where 
            <decimal> is the decimal argument.
        <search> argument should take the form of a RegEx where the special token 
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
        
    elif not full_path:
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
    sep = os.path.sep
    start = '(?:^|{sep}|(?<={}))'.format( '{}', sep = sep ) if full_path else '(?:^|(?<={}))'
    end   = '(?={}|{sep}|$)'.format( '{}', sep = sep )      if full_path else '(?={}|$)'
    
    start = start.format( delimeter )
    end   = end.format( delimeter )
    
    search = start + search + end
    
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
        df = common_reindex( df, how = interpolate, fillna = fillna )
        
    df = pd.concat( df, axis = 1 ).sort_index( axis = 1 )
    return df


# In[ ]:


def downsample( df, method, value, how = 'linear' ):
    """
    Downsamples a DataFrame.
    
    :param method: Method to use for down sampling.
        + values: Down samples to the given values.
        + samples: Down samples to the given number of samples, evenly spaced.
        + resolution: Down samples to the given resoltuion.
    :param value: Values associated to the down sampling method.
    :param how: Grouping method. [Default: linear]
    :returns: Down sampled DataFrame.
    """
    df = df.copy()
    index = df.index
    
    if method == 'values':
        new_index = value
        
    elif method == 'samples':
        new_index = np.linspace( index.min(), index.max(), value )
        
    elif method == 'resolution':
        new_index = np.arange( index.min(), index.max() + value, value )
    
    # create common index
    combined_index = [ df.index.values, new_index ]
    combined_index = np.unique( np.concatenate( combined_index ) )
    combined_index = pd.Index( combined_index, name = index.name )
    
    # reindex data
    df = df.reindex( combined_index ).interpolate( method = how, limit_area = 'inside' )
    df = df.reindex( new_index )
    return df.dropna()


# TODO: Handle duplicate index values
def common_reindex( 
    dfs, 
    index = None, 
    how = 'linear', 
    fillna = 0, 
    add_values = None, 
    name = None,
    duplicates = None
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
    :param duplicates: (Not Implemented) Function to reduce duplicate index values, or None to raise Exception.
        [Default: None]
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


def keep_levels( df, levels, axis = 1, inplace = False ):
    """
    Keeps the given levels of the index.
    
    :param df: the DataFrame to modify.
    :param level: Value or list of levels to keep. Can be an integer or level names.
        [Default: 1]
    :param axis: The axis of the index to modify. [Defaut: 1]
    :param inplace: Modify the DataFrame in place. [Default: False]
    :returns: The modified DataFrame.
    :raises ValueError: If invalid level is passed.
    """
    
    def get_vals( elm, indices ):
        """
        Gets the values of an element at the given indices.
        
        :param elm: An iterable structure.
        :param indices: Value or list of index values to extract.
        :returns: Values at the given indices.
        """
        t = type( elm ) # remember type
        vals = [ elm[ ind ] for ind in indices  ] # extract values
        
        return t( vals ) # cast type
        
    
    if type( levels ) not in ( list, tuple ):
        levels = [ levels ]
    
    old = df.axes[ axis ]
    
    # get index values of levels
    for index, level in enumerate( levels ):
        if type( level ) is str:
            levels[ index ] = get_level_index( df, level, axis = axis )
    
    # creat new index, keeping only desired values
    new = pd.MultiIndex.from_tuples(
        [ get_vals( vals, levels ) for vals in old.values ],
        names = get_vals( old.names, levels )
    )
    
    if not inplace:
        df = df.copy()
        
    # replace axis
    if axis == 0:
        df.index = new
        
    elif axis == 1:
        df.columns = new
    
    return df
    

def drop_outer_levels( df, level = 1, axis = 1, inplace = False ):
    """
    Drops outer levels of a MultiIndex, keeping the inner indices.
    
    :param df: the DataFrame to modify.
    :param level: How many levels to keep. Can be an integer or a level name.
        [Default: 1]
    :param axis: The axis of the index to modify. [Defaut: 1]
    :param inplace: Modify the DataFrame in place. [Default: False]
    :returns: The modified DataFrame.
    :raises ValueError: If invalid level is passed.
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
        raise ValueError( 'Invalid level {}'.format( level ) )
        
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
    
    
def insert_index_levels( df, levels, names = None, key_level = 0, axis = 1, inplace = False  ):
    """
    Insert levels into a MultiIndexed DataFrame.
    
    :param df: The DataFrame to modify.
    :param levels: List of level values.
    :param names: List of level names. [Defualt: None]
    :param key_level: Index of insertion. [Default: 0]
    :param axis: Axis to insert on. [Default: 1]
    :param inplace: Transform DataFrame inplace. [Default: False]
    :returns: Modified DataFrame.
    """
    if not inplace:
        df = df.copy()
    
    ax = df.axes[ axis ]
    
    if not isinstance( levels, list ):
        levels = [ levels ]
    
    if names is None:
        names = [ None ]* len( levels )
        
    elif not isinstance( names, list ):
        names = [ names ]
    
    # create levels
    col_names = ax.values
    
    # convert all levels in to tuples, required for single level indexes
    if not isinstance( col_names[ 0 ], tuple ):
        col_names = [ ( name, ) for name in col_names ]

    levels = [ 
        ( *name[ 0: key_level ], *levels, *name[ key_level: ] ) 
        for name in col_names
    ]
    
    level_names = ax.names
    names = [ 
        *level_names[ :key_level ], 
        *names, 
        *level_names[ key_level: ] 
    ]
    
    # set index or columns
    new_index = pd.MultiIndex.from_tuples( levels, names = names )
    if axis == 0:
        df.index = new_index
    
    elif axis == 1:
        df.columns = new_index
        
    else:
        raise ValueError( 'Invalid axis {}'.format( axis ) )
    
    return df
    

def add_index_levels( df, groups, names = None, key_level = 0, axis = 1 ):
    """
    (Not Implemented)
    Adds addtional MultiIndex levels to a Pandas DataFrame
    
    :param df: The DataFrame to modify
    :param groups: A nested dictionary with keys as the group name and 
        values a list of current level values in that group.
        Multiple levels can be defined at once using nested dictionaries.
        If None, all current values under key_level are added. 
    :param names: A name or list of names for the new levels. [Default: None]
    :param key_level: The level of the current index which the grouping values exist [Default: 0]
    :param axis: The axis to group. 0 for index, 1 for columns [Default: 1]
    :returns: The grouped DataFrame
    """    
    grouped = []
    ax = df.axes[ axis ]
    old_names = ax.names
    names = names if isinstance( names, list ) else [ names ]

    if isinstance( key_level, str ):
        key_level = old_names.index( key_level )

    for index in ax:
        if isinstance( index, tuple ):
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
    
    if names is not None:
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


# In[ ]:


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
            
            with warnings.catch_warnings( record = True ) as w:
                warnings.filterwarnings( 'error' )
    
                try:
                    fit = curve_fit(
                        fcn,
                        xdata = data.index.values,
                        ydata = data.values,
                        p0 = initial,
                        **kwargs
                    )

                except RuntimeError as err:
                    logging.warning( f'{ col }: { err }' )
                    continue

                except TypeError as err:
                    logging.warning( f'{ col }: { err }' )
                    continue
                    
                except Exception as err:
                    logging.warning( f'{ col }: { err }' )
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


def fits_to_df( fcn, fits, index ):
    """
    Converts a Pandas DataFrame of fit parameters (output from #df_fit_function) to a
    DataFrame of values.
    
    :param fcn: The function used as the fit.
    :param fits: A DataFrame of fits, as returned from #df_fit_function).
    :param index: The index values to use.
    :returns: A DataFrame with each column for each row in the fits, 
        with values of the function evaluated on the provided index.
    """
    for index, fit in fits.iterrows:
        params = fit.xs( 'value', level = 'metric' )

        fits = inten.xs( 'value', level = 'metric' )
        i = pl.intensity_gaussian_population( fits.Eg0, fits.sigma, fits.t )

        idata = fits.A* np.array( list( map( i, xdata ) ) )
        inten = pd.DataFrame( idata, index = meas.index, columns = [ 'intensity' ] )
    


# In[1]:


def smooth_mask( mask, window = 10 ):
    """
    Smooths a mask of True/False values.
    
    :param mask: Mask to smooth.
    :param window: Smoothing window. [Default: 10]
    :returns: Smoothed mask.
    """
    # convert False/True to 0/1
    if isinstance( mask, pd.DataFrame ):
        if mask.shape[ 1 ] > 1:
            raise TypeError( 'Mask can not have more than one column.' )
        
        df = mask.squeeze( axis = 1 ).astype( int )
        
    elif isinstance( mask, pd.Series ):
        df = mask.apply( int )
        
    else:
        df = pd.Series( map( int, mask ) )
    
    # smooth values and convert back to False/True
    df = df.rolling( window = window ).mean()
    df = df.fillna( 0 )
    df = df.apply( round )
    df = df.apply( bool )
    
    return df.values


def mask_from_threshold( 
    df, 
    threshold = 3, 
    deviation = 'std',
    direction = 0,
    separation = 0,
    keep = 'first'
):
    """
    Create mask of indices breaking a threshold.
    
    :param df: The DataFrame to threshold.
    :param threshold: Deviations greater than threshold are masked.
        [Default: 3]
    :param deviation: The type of deviation to use. 
        Values [ 'std', 'error', 'value' ].
        'std' uses standard deviation.
        'error' uses deviation from the mean.
        'value' uses the raw values.
        [Default: 'std']
    :param direction: Direction values must pass threshold.
        +1 for more positive, -1 for more negative, 0 for absolute.
        [Default: 0]
    :param separation: Minimum separation between points.
        [Default: 0]
    :param keep: Point to keep if multiple are within spearation of eachother,
        or None to raise an Exception.
        Values are [ 'first', 'last', 'middle', None ]
        [Default: 'first']
    :returns: Indices of values breaking threshold.
    :raises: RuntimeException if multiple points are within separation of eachother
        and keep is None.
    """
    # setup dataframe and threshold
    if deviation == 'std':
        # use standard deviation
        threshold *= df.std()
        tdf = ( df - df.mean() ) # standardize data
        
    elif deviation == 'error':
        # use error relative to mean
        threshold *= df.mean()
        tdf = ( df - df.mean() ) # standardize data

    elif deviation == 'value':
        # value threshold 
        tdf = df.copy()
        
    else:
        raise ValueError( 'Invalid deivation type.' )
        
    # get mask from direction
    if direction == 0:
        mask = np.where( tdf.abs() > threshold )
        
    elif direction == -1:
        mask = np.where( tdf < threshold )
        
    elif direction == 1:
        mask = np.where( tdf > threshold )
        
    else:
        raise ValueError( 'Invalid direction.' )
        
    mask = mask[ 0 ]

    # check separation
    # find mask points with separation less than specified
    breaks = []
    for index in range( 1, len( mask ) ):
        if ( mask[ index ] - mask[ index - 1 ] ) > separation:
            # index is start of new mask group
            breaks.append( index )
            
    breaks.append( None ) # include final mask point
    
    # break mask into groups
    pbk = 0
    groups = []
    for brk in breaks:
        groups.append( mask[ pbk: brk ] )
        pbk = brk
    
    # keep group points
    if keep == 'first':
        mask = [ group[ 0 ] for group in groups ]
        
    elif keep == 'last':
        mask = [ group[ -1 ] for group in groups ]
        
    elif keep == 'middle':
        mask = [ group[ int( len( group )/ 2 ) ] for group in groups ]
            
    elif keep is None:
        raise RuntimeError( 'Separation violation in mask.' )
        
    else:
        raise ValueError( 'Invalid keep.' )
    
    return mask


def break_from_mask( df, mask, name = 'cycle', axis = 0, inplace = False ):
    """
    Breaks a DataFrame into cycles, given a mask.
    
    :param df: A Pandas DataFrame.
    :param mask: List of indices indicating break position.
    :param name: Name to assign to new index. [Default: 'cycle']
    :param axis: Axis to combine breaks. [Default: 0]
    :param inplace: Modify DataFrame inplace. [Default: False]
    :returns: Pandas DataFrame split into cycles.
    """
    if not inplace:
        df = df.copy()
    
    ax = df.axes[ axis ]
    names = ( name, *ax.names )
    
    mask = np.append( mask, None )
    if not ( 0 in mask ): 
        mask = np.insert( mask, 0, None )
    
    cycles = [
        df.iloc[ mask[ index ] : mask[ index + 1 ] ]
        for index in range( len( mask ) - 1 )
    ]
    
    # add cycle header
    for cycle, data in enumerate( cycles ):
        d_ax = data.axes[ axis ]
        
        headers = (
            [ ( cycle, *head_val ) for head_val in d_ax.values ]
            if isinstance( ax, pd.MultiIndex ) else
            [ ( cycle, head_val ) for head_val in d_ax.values ]
        )
        
        headers = pd.MultiIndex.from_tuples(
            headers, names = names
        )
        
        if axis == 0:
            data.index = headers
            
        else:
            data.columns = headers
    
    cycles = pd.concat( cycles, axis = axis )
    return cycles


def align_cycles( df, name = 'cycles' ):
    """
    Moves cycles from columns to index, adjusting times.
    
    :param df: DataFrame with cycles.
    :param name: Name of the index to align. [Default: 'cycles']
    :returns: DataFrame with time aligned in index by scan.
    """
    cycles = []
    time = 0
    for cycle, data in df.groupby( level = name, axis = 1 ):
        data.index = data.index + time
        time = data.index.max()

        data = data.dropna()
        data.columns = data.columns.droplevel( name )
        data = std.insert_index_levels( data, cycle, name, axis = 0 )

        cycles.append( data )

    cycles = pd.concat( cycles, axis = 0 ).sort_index( 0 )
    return cycles


def gradient_threshold( 
    df, 
    div = 'slope', 
    threshold = -1, 
    calc = 'error',
    window = 5,
    derivative = 1
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

    cycles = pd.concat( cycles, axis = 1 )
    return cycles


def break_from_gradient( 
    df, 
    threshold = -1, 
    calc = 'error',
    derivative = 1,
    window = 5,
    **kwargs
):
    """
    Thresholds data based on the local curvature.
    
    :param df: The DataFrame to threshold.
    :param div: The type of derivative to examine. Use 'slope' or 'curvature'. [Default: slope]
    :param threshold: [Default: -1]
    :param calc: The type of threshold to use. 
        Use 'absolute' or 'error'. [Default: error]
    :param derivative: Number of derivatives to compute. [Default: 1]
    :param window: Window width to calculate average gradient for error calculation.
        [Default: 5]
    :param kwargs: Parameters passed to #break_from_mask.
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
    
    
    grads = df
    for _ in range( derivative ):
        grads = compute_grads( grads )
    
    if calc == 'error':
        # compute error threshold
        threshold = threshold* grads.rolling( window = window ).mean().abs()
        grads = grads.abs()
        mask = df.where( grads > threshold )
    
    else:
        # absolute threshold
        if threshold > 0:
            mask = df.where( grads > threshold )

        if threshold < 0:
            mask = df.where( grads < threshold )
        
    df = break_from_mask( df, mask, **kwargs )
    return df


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
    
    
def df_grad( series ):
    """
    Gradient function for use of DataFrame#apply.
    """
    return np.gradient( series.values, series.index.values )


# ## Plots

# In[13]:


def index_from_counter( counter, rows, cols ):
    """
    Get the row and column of a matrix form a counter.
    
    :param counter: Counter.
    :param rows: Number of rows in matrix.
    :param cols: Number of columns in matrix.
    :returns: ( row, column ) of counter.
    """
    
    row = int( np.floor( counter / cols ) )
    col = int( counter % cols )
    
    return ( row, col )


def ax_from_counter( counter, axs ):
    """
    Gets an axis from an array of axes based on a counter.
    
    :param counter: Counter.
    :param axs: Matrix of axes.
    :returns: Axis.
    """
    row, col = index_from_counter( counter, *axs.shape )
        
    if len( axs.shape ) == 1:
        # only rows
        ax = ax[ row ]

    else:
        # rows and cols
        ax = axs[ row, col ]
        
    return ax


def plot_levels( plot, df, show = True, level = 'metrics', axis = 1, **fig_args ):
    """
    Plots each element of a Pandas DataFrame in a separate subplot.
    
    :param plot: A function that receives a Pandas DataSeries and axis to plot it on ( ax, data, name ).
    :param df: The DataFrame to plot.
    :param show: Show the plot. [Defualt: True]
    :param level: Which level to iterate over. [Default: 'metrics']
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
        ax = ax_from_counter( index, axs )
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
    
    
def temperature_plot_rainbow( df, colorbar = True, **kwargs ):
    """
    Plots a DataFrame by temperature.
    
    :param df: A DataFrame with temperature as the first colum index level.
    :param colorbar: Whether to include the color bar legend. [Default: True]
    :param **kwargs: Arguments passed to pandas.DataFrame#plot
    """
    
    fig, ax = plt.subplots()
    NUM_COLORS = df.shape[ 1 ]
    cm = plt.get_cmap( 'gist_rainbow' )
    ax.set_prop_cycle( color = [ cm( float( i / NUM_COLORS ) ) for i in range( NUM_COLORS ) ] )

    df.plot( ax = ax, **kwargs )
    
    if colorbar:
        temp_vals = df.columns.get_level_values( 0 ).values
        cax = fig.add_axes( [0.92, 0.15, 0.05, 0.7] )
        cbar = mpl.colorbar.ColorbarBase( 
            ax = cax, 
            cmap = cm,  
            norm = mpl.colors.Normalize( vmin = temp_vals.min(), vmax = temp_vals.max() ),
            orientation = 'vertical'
        )
        
        cbar_label = df.columns.names[ 0 ]
        cbar.set_label( cbar_label, labelpad = 15 )
    
    return ( fig, ax )


# # Work

# In[ ]:




