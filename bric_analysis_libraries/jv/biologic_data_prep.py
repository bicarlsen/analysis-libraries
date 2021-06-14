
# coding: utf-8

# # Biologic Data Prep

# ### Imports

import os
import logging

import numpy as np
import pandas as pd

from bric_analysis_libraries import standard_functions as std
from bric_analysis_libraries.jv import jv_data_prep as jvdp

# ## Data Prep

# ### Imports

# convenience methods
def channel_from_file_path( path, pattern = 'ch-<>' ):
    """
    Gets the channel from the file path.

    :param path: File path.
    :param pattern: RegEx pattern describing teh channel pattern.
        Passed to standard_functions#metadata_from_file_name.
        [Default: 'ch-<>']
    """
    ch = std.metadata_from_file_name(
        pattern,
        path,
        is_numeric = True,
        full_path = True
    )

    return int( ch )


# convenience methods
def cycle_from_file_path( path, pattern = 'cycle-<>' ):
    """
    Gets the cycle from the file path.

    :param path: File path.
    :param pattern: RegEx pattern describing teh channel pattern.
        Passed to standard_functions#metadata_from_file_name.
        [Default: 'cycle-<>']
    """
    cycle = std.metadata_from_file_name(
        pattern,
        path,
        is_numeric = True,
        full_path = True
    )

    return int( cycle )


def get_channel_folders( folder, pattern = 'ch-*' ):
    """
    Returns a list of folders in the folder matching the pattern.

    :param folder: Folder to search in.
    :param pattern: glob pattern to filter for. [Default: 'ch-*']
    """
    return glob.glob( os.path.join( folder, pattern ) )


def get_cycle_folders( folder, pattern = 'cycle-*' ):
    """
    Returns a list of folders in the folder matching the pattern.

    :param folder: Folder to search in.
    :param pattern: glob pattern to filter for. [Default: 'cycle-*']
    """
    return glob.glob( os.path.join( folder, pattern ) )


def import_datum( file, parameters ):
    """
    Imports a file indexed with channels.

    :param file: File path.
    :param parameters: Dictionary of parameter names keyed by raw with desired values.
    :returns: Pandas DataFrame.
    """
    df = pd.read_csv( file, header = [ 0, 1 ] )

    df.columns = pd.MultiIndex.from_tuples(
        [ ( int( ch ), parameters[ param.strip() ] ) for ch, param in df.columns.values ],
        names = [ 'channel', 'metrics' ]
    )

    return df


def import_voc_datum( file, reindex = True ):
    """
    Imports a Voc file with channel headers.

    :param file: Path to file.
    :param reindex: Reindex by time. [Default: True]
    :returns: Pandas DataFrame.
    """
    parameters = {
        'Time [s]': 'time',
        'Voltage [V]': 'voltage'
    }

    df = import_datum( file, parameters )

    if reindex:
        data = []
        for ch, datum in df.groupby( level = 'channel', axis = 1 ):
            datum = datum.dropna( subset = [ ( ch, 'time' ) ] )
            datum.set_index( ( ch, 'time' ), inplace = True )
            datum.index = datum.index.rename( 'time' )

            data.append( datum )

        data = std.common_reindex( data, fillna = np.NaN )
        df = pd.concat( data, axis = 1 )

        df.columns = df.columns.droplevel( 'metrics' )

    return df


def import_jv_datum( file, skip_rows = 0, reindex = True, split_direction = True  ):
    """
    Imports a JV file with channel headers.

    :param file: Path to file.
    :param skip_rows: Skip rows. [Default: 0]
    :param reindex: Reindex by voltage. [Default: True]
    :param split_direction: Split scans by direction. [Default: True]
    :returns: Pandas DataFrame.
    """
    parameters = {
        'Voltage [V]': 'voltage',
        'Current [A]': 'current',
        'Power [W]':   'power'
    }

    df = import_datum( file, parameters )
    df = df[ skip_rows: ]

    if reindex:
        data = []
        for ch, datum in df.groupby( level = 'channel', axis = 1 ):
            if split_direction:
                datum = jvdp.split_jv_scan( datum )

            data.append( datum )

        data = std.common_reindex( data, fillna = np.NaN )
        df = pd.concat( data, axis = 1 )

    return df


def import_mpp_tracking_datum( file, reindex = True, drop_cycle = True ):
    """
    Imports a MPP Tracking file with channel headers.

    :param file: Path to file.
    :param reindex: Reindex by time. [Default: True]
    :param drop_cycle: Drop cycle columns. [Default: True]
    :returns: Pandas DataFrame.
    """
    parameters = {
        'Time [s]':    'time',
        'Voltage [V]': 'voltage',
        'Current [A]': 'current',
        'Power [W]':   'power',
        'Cycle':       'cycle'
    }

    df = import_datum( file, parameters )

    if drop_cycle:
        df = df.drop( 'cycle', axis = 1, level = 'metrics' )

    if reindex:
        data = []
        for ch, datum in df.groupby( level = 'channel', axis = 1 ):
            datum = datum.dropna( subset = [ ( ch, 'time' ) ] )
            datum.set_index( ( ch, 'time' ), inplace = True )
            datum.index = datum.index.rename( 'time' )
            datum = datum.mean( level = 0 )  # take mean over duplicate index values

            data.append( datum )

        data = std.common_reindex( data, fillna = np.NaN )
        df = pd.concat( data, axis = 1 )

    return df


def import_mpp_datum( folder, voc_kwargs = {}, jv_kwargs = {}, mpp_kwargs = {} ):
    """
    Import Voc, JV scan, and MPP tracking data.

    :param folder: Path to folder containing files.
    :param voc_kwargs: Dictionary of keyword arguments passed to import_voc_datum().
        [Default: {}]
    :param jv_kwargs: Dictionary of keyword arguments passed to import_jv_datum().
        [Default: {}]
    :param mpp_kwargs: Dictionary of keyword arguments passed to import_mpp_datum().
        [Default: {}]
    :returns: Tuple of ( voc, jv, mpp ) DataFrames.
    """
    voc = import_voc_datum( os.path.join( folder, 'voc.csv' ), **voc_kwargs )
    jv  = import_jv_datum(  os.path.join( folder, 'jv.csv'  ), **jv_kwargs  )
    mpp = import_mpp_tracking_datum( os.path.join( folder, 'mpp.csv' ), **mpp_kwargs )

    return ( voc, jv, mpp )


def import_mpp_cycle_datum( folder, voc_kwargs = {}, jv_kwargs = {}, mpp_kwargs = {} ):
    """
    Import MPP data for a single cycle.

    :param folder: Path to folder containing cycle data.
    :param voc_kwargs: Dictionary of keyword arguments passed to import_voc_datum().
        [Default: {}]
    :param jv_kwargs: Dictionary of keyword arguments passed to import_jv_datum().
        [Default: {}]
    :param mpp_kwargs: Dictionary of keyword arguments passed to import_mpp_datum().
        [Default: {}]
    :returns: Tuple of ( voc, jv, mpp ) DataFrames.
    """
    cycle = cycle_from_file_path( folder )
    dfs = list(
        import_mpp_datum( folder, voc_kwargs, jv_kwargs, mpp_kwargs )
    )

    for index, df in enumerate( dfs ):
        df = std.insert_index_levels( # add cycle to index, below channel
            df,
            levels = [ cycle ],
            names  = [ 'cycle' ],
            key_level = 1
        )

        dfs[ index ] = df

    return tuple( dfs )


def import_mpp_cycle_data( folder, voc_kwargs = {}, jv_kwargs = {}, mpp_kwargs = {} ):
    """
    Import MPP data for multiple cycles.

    :param folder: Path to folder containing cycle data.
    :param voc_kwargs: Dictionary of keyword arguments passed to import_voc_datum().
        [Default: {}]
    :param jv_kwargs: Dictionary of keyword arguments passed to import_jv_datum().
        [Default: {}]
    :param mpp_kwargs: Dictionary of keyword arguments passed to import_mpp_datum().
        [Default: {}]
    :returns: Tuple of ( voc, jv, mpp ) DataFrames.
    """
    # get scan folders
    cycles = os.listdir( folder )

    # get data for each scan
    vocs = []
    jvs  = []
    mpps = []

    for cy_dir in cycles:
        cy_path = os.path.join( folder, cy_dir )

        voc, jv, mpp = import_mpp_cycle_datum(
            cy_path, voc_kwargs, jv_kwargs, mpp_kwargs ) # import cycle data

        vocs.append( voc )
        jvs.append( jv )
        mpps.append( mpp )

    vocs = std.common_reindex( vocs )
    jvs  = std.common_reindex( jvs )
    mpps = std.common_reindex( mpps )

    vocs = pd.concat( vocs, axis = 1 ).sort_index( axis = 1 )
    jvs  = pd.concat( jvs,  axis = 1 ).sort_index( axis = 1 )
    mpps = pd.concat( mpps, axis = 1 ).sort_index( axis = 1 )

    return ( vocs, jvs, mpps )



# ### Manipulation


def align_cycles( df ):
    """
    Moves cycles from columns to index, adjusting times.

    :param df: DataFrame with cycles.
    :returns: DataFrame with time aligned in index by scan.
    """
    cycles = []
    time = 0
    for cycle, data in df.groupby( level = 'cycle', axis = 1 ):
        data.index = data.index + time
        time = data.index.max()

        data = data.dropna()
        data.columns = data.columns.droplevel( 'cycle' )
        data = std.insert_index_levels( data, cycle, 'cycle', axis = 0 )

        cycles.append( data )

    cycles = pd.concat( cycles, axis = 0 ).sort_index( 0 )
    return cycles


def split_by_time( df, interval, inplace = False ):
    """
    Splits a DataFrame into cycles by time intervals.

    :param df: DataFrame to split.
    :param interval: Time interval to split.
    :param inplace: Manipulate DataFrame in place or create a copy. [Default: False]
    :returns: The DataFrame split into cycles by time interval.
    """
    pass


# ## Old Import Methods


def import_voc_datum_channel( file, channel_pattern = 'ch-<>', set_index = True, skiprows = 2 ):
    """
    Imports Voc datum from the given file.

    :param file: File path.
    :param channel_pattern: Add channel from file path as index level.
        Uses value as pattern in standard_functions#metadata_from_file_name.
        None if channel should be excluded.
        [Default: 'ch-<>']
    :param set_index: Sets the index to time. [Default: True]
    :param skiprows: Number of initial data points to drop. [Default: 2]
    :returns: Pandas DataFrame.
    """
    header = [ 'time', 'voltage' ]
    df = pd.read_csv(
        file,
        names = header,
        skiprows = ( 1 + skiprows ),
        engine = 'python'
    )

    if set_index:
        df.set_index( 'time', inplace = True )

    df.columns.rename( 'metrics', inplace = True )

    if channel_pattern is not None:
        ch = channel_from_file_path( file, channel_pattern )
        df = std.insert_index_levels( df, ch, 'channel' )

    # remove duplicate axis
    df = df.loc[ ~ df.index.duplicated() ]

    return df


def import_jv_datum_channel(
    file,
    channel_pattern = 'ch-<>',
    by_scan = True,
    skiprows = 2,
    skiprows_tail = 0
):
    """
    Imports JV datum from the given file.

    :param file: File path.
    :param channel_pattern: Add channel from file path as index level.
        Uses value as pattern in standard_functions#metadata_from_file_name.
        None if channel should be excluded.
        [Default: 'ch-<>']
    :param by_scan: Breaks data into forward and reverse scans, and sets the index to voltage.
        [Default: True]
    :param skiprows: Number of initial data points to drop. [Default: 2]
    :param skiprows_tail: Number of end data points to drop. [Default: 0]
    :returns: Pandas DataFrame.

    :raises ValueError: If multiple sign changes in the scan are detected.
    """
    header = [ 'voltage', 'current', 'power' ]
    df = pd.read_csv( file, names = header, skiprows = ( 1 + skiprows ) )

    # drop tail points
    if skiprows_tail:
        df = df.iloc[ : -skiprows_tail ]

    if by_scan:
        df = jvdp.split_jv_scan( df )


    if channel_pattern is not None:
        ch = channel_from_file_path( file, channel_pattern )
        df = std.insert_index_levels( df, ch, 'channel' )

    return df


def import_mpp_tracking_datum_channel(
    file,
    channel_pattern = 'ch-<>',
    set_index = True,
    drop_cycle = True,
    skiprows = 2
):
    """
    Imports MPP tracking datum from the given file.

    :param file: File path.
    :param channel_pattern: Add channel from file path as index level.
        Uses value as pattern in standard_functions#metadata_from_file_name.
        None if channel should be excluded.
        [Default: 'ch-<>']
    :param set_index: Sets the index to time. [Default: True]
    :param drop_cycle: Removes cycle information from the data. [Default: True]
    :param skiprows: Number of initial data points to drop. [Default: 2]
    :returns: Pandas DataFrame.
    """
    header = [ 'time', 'voltage', 'current', 'power', 'cycle' ]
    df = pd.read_csv( file, names = header, skiprows = ( skiprows + 1 ) )

    if drop_cycle:
        df.drop( 'cycle', axis = 1, inplace = True )

    if set_index:
        df.set_index( 'time', inplace = True )

    df.columns.rename( 'metrics', inplace = True )

    if channel_pattern is not None:
        ch = channel_from_file_path( file, channel_pattern )
        df = std.insert_index_levels( df, ch, 'channel' )

    return df


def import_mpp_datum_channel(
    folder,
    voc_kwargs = {},
    jv_kwargs  = {},
    mpp_kwargs = {}
):
    """
    Imports Voc, JV, and MPP data.

    :param folder: Folder path.
    :param voc_kwargs: Dictionary of keyword arguments passed to #import_mpp_voc_data.
    :param jv_kwargs:  Dictionary of keyword arguments passed to #import_mpp_jv_data.
    :param mpp_kwargs: Dictionary of keyword arguments passed to #import_mpp_mpp_data.
    :returns: Tuple of ( voc, jv, mpp ) Pandas DataFrames.
    """

    return (
        import_mpp_voc_data_channel( folder, **voc_kwargs ),
        import_mpp_jv_data_channel(  folder, **jv_kwargs  ),
        import_mpp_tracking_data_channel( folder, **mpp_kwargs )
    )


def import_mpp_cycle_datum_channel( folder, cycle_pattern = 'cycle-<>', channel_pattern = 'ch-<>' ):
    """
    Imports MPP tracking data from an MPP with JV program, broken in to cycles.

    :param folder: Folder path containing cycles.
    :param cycle_pattern: Pattern for cycle folders. [Default: 'cycle-<>']
    :param channel_pattern: Pattern for channel folders. [Default: 'ch-<>']
    :returns: Tuple of ( voc, jv, mpp ) Pandas DataFrames by cycle.
    """
    # get scan folders
    cycles = os.listdir( folder )

    # get data for each scan
    vocs = []
    jvs  = []
    mpps = []

    for cycle in cycles:
        cycle_path = os.path.join( folder, cycle )

        dfs = ( voc, jv, mpp ) = import_mpp_datum_channel( cycle_path ) # import scan data

        # add scan index
        cycle_id = int( std.metadata_from_file_name(
            cycle_pattern,
            cycle_path,
            full_path = True,
            is_numeric = True
        ) )

        for df in dfs:
            # channel already in headers
            std.insert_index_levels( df, cycle_id, 'cycle', key_level = 1 )

        vocs.append( voc )
        jvs.append( jv )
        mpps.append( mpp )

    vocs = std.common_reindex( vocs )
    jvs  = std.common_reindex( jvs )
    mpps = std.common_reindex( mpps )

    vocs = pd.concat( vocs, axis = 1 ).sort_index( axis = 1 )
    jvs  = pd.concat( jvs,  axis = 1 ).sort_index( axis = 1 )
    mpps = pd.concat( mpps, axis = 1 ).sort_index( axis = 1 )

    return ( vocs, jvs, mpps )


def import_jv_data( folder, file_pattern = 'ch*.csv', by_scan = True, **kwargs ):
    """
    Imports JV data.

    :param folder: Folder path containing data files.
    :param file_pattern: File pattern of data files, in glob format. [Default: ch*.csv]
    :param by_scan: Breaks data into forward and reverse scans, and sets the index to voltage.
        [Default: True]
    :param kwargs: Arguments passed to standard_functions#import_data
    :returns: DataFrame containg imported data.
    """
    jv = []
    for file in std.get_files( folder, file_pattern ):
        try:
            df = import_jv_datum_channel(
                file,
                channel_pattern = 'ch-<>',
                by_scan = by_scan
            )

        except ValueError as err:
            logging.warning( '{}: {}'.format( file, err ) )
            continue

        jv.append( df )

    jv = pd.concat( jv, axis = 1 )
    return jv


def import_mpp_voc_data_channel( folder, file_pattern = 'voc.csv', **kwargs ):
    """
    Imports Voc data from an MPP measuremnt.

    :param folder: Folder path containing data files.
    :param file_pattern: File pattern of data files, in glob format. [Default: voc.csv]
    :param kwargs: Arguments passed to #import_mpp_voc_datum.
    :returns: DataFrame containg imported data.
    """
    return std.import_data( import_voc_datum_channel, folder, file_pattern = file_pattern, **kwargs )


def import_mpp_jv_data_channel( folder, file_pattern = 'jv.csv', **kwargs ):
    """
    Imports JV data from an MPP measuremnt.

    :param folder: Folder path containing data files.
    :param file_pattern: File pattern of data files, in glob format. [Default: jv.csv]
    :param kwargs: Arguments passed to #import_mpp_jv_datum.
    :returns: DataFrame containg imported data.
    """
    return std.import_data( import_jv_datum_channel, folder, file_pattern = file_pattern, **kwargs )


def import_mpp_tracking_data_channel( folder, file_pattern = 'mpp.csv', **kwargs ):
    """
    Imports MPP data from an MPP measuremnt.

    :param folder: Folder path containing data files.
    :param file_pattern: File pattern of data files, in glob format. [Default: mpp.csv]
    :param kwargs: Arguments passed to #import_mpp_tracking_datum.
    :returns: DataFrame containg imported data.
    """
    return std.import_data(
        import_mpp_tracking_datum_channel,
        folder,
        file_pattern = file_pattern,
        **kwargs
    )


def import_mpp_data_channel( folder, channel_pattern = 'ch-<>' ):
    """
    Imports MPP data from multiple channels.

    :param folder: Folder containing channels.
    :param channel_pattern: Pattern for channel folders. [Default: 'ch-<>']
    :returns: tuple of ( voc, jv, mpp ) Pandas DataFrames by cycle and channel.
    """
    # get scan folders
    channels = os.listdir( folder )

    # get data for each scan
    vocs = []
    jvs  = []
    mpps = []

    for c_dir in channels:
        ch_path = os.path.join( folder, c_dir )

        voc, jv, mpp = import_mpp_datum_channel( # import scan data
            ch_path
        )

        vocs.append( voc )
        jvs.append( jv )
        mpps.append( mpp )

    vocs = std.common_reindex( vocs )
    jvs  = std.common_reindex( jvs )
    mpps = std.common_reindex( mpps )

    vocs = pd.concat( vocs, axis = 1 ).sort_index( axis = 1 )
    jvs  = pd.concat( jvs,  axis = 1 ).sort_index( axis = 1 )
    mpps = pd.concat( mpps, axis = 1 ).sort_index( axis = 1 )

    return ( vocs, jvs, mpps )


def import_mpp_cycle_data_channel( folder, channel_pattern = 'ch-<>', cycle_pattern = 'cycle-<>' ):
    """
    Imports MPP tracking data from an MPP with JV program from multiple channels, broken in to cycles.

    :param folder: Folder path containing channels with cycles.
    :param channel_pattern: Pattern for channel folders. [Default: 'ch-<>']
    :param cycle_pattern: Pattern for cycle folders. [Default: 'cycle-<>']
    :returns: tuple of ( voc, jv, mpp ) Pandas DataFrames by cycle and channel.
    """
    # get scan folders
    channels = os.listdir( folder )

    # get data for each scan
    vocs = []
    jvs  = []
    mpps = []

    for c_dir in channels:
        ch_path = os.path.join( folder, c_dir )

        voc, jv, mpp = import_mpp_cycle_datum_channel( # import scan data
            ch_path,
            channel_pattern = channel_pattern,
            cycle_pattern = cycle_pattern
        )

        vocs.append( voc )
        jvs.append( jv )
        mpps.append( mpp )

    vocs = std.common_reindex( vocs )
    jvs  = std.common_reindex( jvs )
    mpps = std.common_reindex( mpps )

    vocs = pd.concat( vocs, axis = 1 ).sort_index( axis = 1 )
    jvs  = pd.concat( jvs,  axis = 1 ).sort_index( axis = 1 )
    mpps = pd.concat( mpps, axis = 1 ).sort_index( axis = 1 )

    return ( vocs, jvs, mpps )
