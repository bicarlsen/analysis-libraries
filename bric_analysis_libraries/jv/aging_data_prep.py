#!/usr/bin/env python
# coding: utf-8

# Aging Setup Data Prep

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


# Convenience functions

def sample_from_file_name( file ):
    name_search = '(.+?)'
    return std.metadata_from_file_name( name_search, file, delimeter = '_', group = 1 )


def channel_from_file_name( file ):
    channel_search = '^Ch(\d+)'
    return std.metadata_from_file_name( channel_search, file, is_numeric = True, delimeter = '_' )


def sample_channel_index( file, metrics, sample_index, channel_index ):
    """
    Creates a standard column index with the sample name and channel.

    :param file: The file to optain the sample and channel from.
    :param metrics: A list of metric anmes as the bottom index level.
    :param sample_index: If True the sample name is used from the file name
        to create an index for the data.
    :param channel_index: If True the file channel is used from the file name
        to create an index for the data.
    :returns: A Pandas MultiIndex with levels [ 'channel', 'sample', 'metrics' ]
        as specified.
    """
    header = [  metrics ]
    names  = [ 'metrics' ]

    if sample_index:
        sample = sample_from_file_name( file )
        header.insert( 0, [ sample ] )
        names.insert( 0, 'sample' )

    if channel_index:
        channel = channel_from_file_name( file )
        header.insert( 0, [ channel ] )
        names.insert( 0, 'channel' )

    return pd.MultiIndex.from_product( header, names = names )



def import_aging_datum( file, sample_index = True, channel_index = False ):
    """
    Imports aging data from an _aging.txt file into a Pandas DataFrame.

    :param file: The file to import from.
    :param sample_index: If True the sample name is used from the file name
        to create an index for the data. [Default: True]
    :param channel_index: If True the file channel is used from the file name
        to create an index for the data. [Default: False]
    :returns: A Pandas DataFrame with the file's data.
    """
    header = [ 'time', 'power', 'voltage', 'current', 'intensity', 'temperature' ]
    df = pd.read_csv( file, sep = '\s+', skiprows = 1, names = header )

    header = sample_channel_index( file, header, sample_index, channel_index )
    df.columns = header

    return df


def import_metric_datum( file, reindex = True, sample_index = True, channel_index = False  ):
    """
    Imports JV metric data from a _JVmetrics.txt file into a Pandas DataFrame.

    :param file: The file to import from.
    :param reindex: Whether to reindex columns hierarchically or leave flat.
        If flat scan direction is indicated by '_rev' or '_for' trailing the metric.
        If hierarchical levels for data are [ 'direction', 'metric' ],
            where direction is [ 'forward', 'reverse', 'static' ], and
            metric is the standard abbereviation.
        [Default: True]
    :param sample_index: If True the sample name is used from the file name
        to create an index for the data. [Default: True]
    :param channel_index: If True the file channel is used from the file name
        to create an index for the data. [Default: False]
    :returns: A Pandas DataFrame with the file's data.
    """

    header = [
        'time',
        'voc_rev',
        'jsc_rev',
        'ff_rev',
        'power_rev',
        'vmpp_rev',
        'jmpp_rev',
        'hysteresis',
        'voc_for',
        'jsc_for',
        'ff_for',
        'power_for',
        'vmpp_for',
        'jmpp_for',
        'scan_rate',
        'intensity',
        'temperature'
    ]

    df = pd.read_csv( file, sep = '\s+', skiprows = 1, names = header )
    header = sample_channel_index( file, header, sample_index, channel_index )

    if not reindex:
        df.columns = header
        return df

    # create hierarchical index
    # level names
    names = [ 'direction', 'metric' ]
    if sample_index:
        names.insert( 0, 'sample' )

    if channel_index:
        names.insert( 0, 'channel' )

    # format tuples
    values = []
    for val in header.get_values():
        val = list( val )
        metric = val[ -1 ]

        # forward
        direction = metric.find( '_for' )
        if direction > -1:
            val[ -1 ] = metric[ :direction ]
            val.insert( -1, 'forward' )

            values.append( tuple( val ) )
            continue

        # reverse
        direction = metric.find( '_rev' )
        if direction > -1:
            val[ -1 ] = metric[ :direction ]
            val.insert( -1, 'reverse' )

            values.append( tuple( val ) )
            continue

        # static
        val[ -1 ] = metric
        val.insert( -1, 'static' )

        values.append( tuple( val ) )

    header = pd.MultiIndex.from_tuples( values, names = names )
    df.columns = header

    return df.sort_index( axis = 1 )


def import_jv_datum( file, sample_index = True, channel_index = False, sep = '\s+' ):
    """
    Imports aging data from a _JVs.txt file into a Pandas DataFrame.

    :param file: The file to import from.
    :param sample_index: If True the sample name is used from the file name
        to create an index for the data. [Default: True]
    :param channel_index: If True the file channel is used from the file name
        to create an index for the data. [Default: False]
    :param sep: The data separator. Can be a regular expression. [Default: \s+]
    :returns: A Pandas DataFrame with the file's data.
    """
    lines, cols = std.file_shape( file, sep = sep )
    num_scans = int( lines/ 3 )
    num_rows = cols

    names = [ 'index', 'direction', 'metric' ]
    header = [
        range( num_scans ),
        [ 'reverse', 'forward' ],
        [ 'voltage', 'current' ]
    ]

    if sample_index:
        header.insert( 0, [ sample_from_file_name( file ) ] )
        names.insert( 0, 'sample' )

    if channel_index:
        header.insert( 0, [ channel_from_file_name( file ) ] )
        names.insert( 0, 'channel' )

    header = pd.MultiIndex.from_product( header, names = names )
    data = pd.DataFrame( index = np.arange( num_rows ), columns = header )

    # read file in 3 line chunks ( time, voltage, current ) for transposition
    # scanned reverse then forward
    with open( file ) as f:
        splitter = re.compile( sep )
        index = 0
        for time in f:
            # get data
            voltage = splitter.split( f.readline() )
            current = splitter.split( f.readline() )

            # remove empty strings
            voltage = filter( ''.__ne__, voltage )
            current = filter( ''.__ne__, current )

            # convert to floats
            voltage = list( map( float, voltage ) )
            current = list( map( float, current ) )

            # first index where next voltage is larger than current
            direction_change = [ index for
                                index in ( range( len( voltage ) - 1 ) )
                                if voltage[ index ] < voltage[ index + 1 ] ]

            if len( direction_change ) == 0:
                # no forward scan
                logging.warn( 'Scan {} in file {} was not complete.'.format( index, file ) )

                v_rev = voltage
                j_rev = current
                v_for = []
                j_for = []

            else:
                direction_change = direction_change[ 0 ]

                v_rev = voltage[ : direction_change + 1  ]
                j_rev = current[ : direction_change + 1 ]
                v_for = voltage[ direction_change : ]
                j_for = current[ direction_change : ]


            # pad data for combining
            datum = [ v_rev, j_rev, v_for, j_for ]
            datum = list( map( np.array, datum ) )

            ref = np.empty(( num_rows ))
            ref[:] = np.nan

            n_datum = []
            for d in datum:
                nd = ref.copy()
                nd[ :d.shape[ 0 ] ] = d
                n_datum.append( nd )

            vals = np.stack( n_datum, axis = 1 )

            # create dataframe for scan
            df_header = [ h for h in header.get_values() if h[ 1 ] == index ]
            idh = df_header[ 0 ][ :-2 ]

            df_header = pd.MultiIndex.from_tuples( df_header, names = names )
            df = pd.DataFrame(
                data = vals,
                columns = df_header,
                dtype = np.float32
            )

            data.loc[ :, idh ] = df
            index += 1

    return data.dropna( how = 'all' )


def import_control_datum( file, sep = ',' ):
    """
    Imports temperature and intensity control data.

    :param file: File path of the program.
    :param sep: The column delimeter. [Default: ,]
    :returns: A pandas DataFrame with the program information.
    """
    names = [
        'duration',
        'intensity',
        'temperature_1',
        'temperature_2',
        'temperature_3',
        'temperature_4',
        'start',
        'pause',
        'stop'
    ]

    df = pd.read_csv( file, names = names, usecols = list( range( 9 ) ) )
    df.loc[ :, 'time' ] = df.duration.cumsum()

    return df.sort_index( axis = 1 )



def import_aging_data( folder, file_pattern = '*_aging.txt', **kwargs ):
    """
    Imports aging data.

    :param folder: Folder path containing data files.
    :param file_pattern: File pattern of data files, in glob format. [Default: *._aging.txt]
    :param kwargs: Arguments passed to standard_functions import_data()
    :returns: DataFrame containg imported data.
    """
    return std.import_data( import_aging_datum, folder, file_pattern = file_pattern, **kwargs )


def import_metric_data( folder, file_pattern = '*_JVmetrics.txt', **kwargs ):
    """
    Imports aging data.

    :param folder: Folder path containing data files.
    :param file_pattern: File pattern of data files, in glob format. [Default: *_JVmetrics.txt]
    :param kwargs: Arguments passed to standard_functions import_data()
    :returns: DataFrame containg imported data.
    """
    return std.import_data( import_metric_datum, folder, file_pattern = file_pattern, **kwargs )


def import_jv_data( folder, file_pattern = '*_JVs.txt', **kwargs ):
    """
    Imports aging data.

    :param folder: Folder path containing data files.
    :param file_pattern: File pattern of data files, in glob format. [Default: *.JVs.txt]
    :param kwargs: Arguments passed to standard_functions import_data()
    :returns: DataFrame containg imported data.
    """
    return std.import_data( import_jv_datum, folder, file_pattern = file_pattern, **kwargs )




def assign_temperatures_to_cycles( df, ctrl ):
    """
    Assigns temperature to cycles given a control DataFrame.
    Assumes all temperatures are the same.

    :param df: A DataFrame that has been split into cycles.
    :param ctrl: The control DataFrame.
    :returns: A Series of temperatures to assing to each cycle.
    """
    # temperature for all channels is the same
    temperatures = ctrl[[ 'time', 'temperature_1' ]].set_index( 'time' )
    temperatures.loc[ 0 ] = temperatures.iloc[ 0 ] # back fill to time 0
    temperatures.sort_index( inplace = True )

    temp_times = temperatures.index
    temps = []
    for name, data in df.groupby( level = [ 'channel', 'cycle' ], axis = 1 ):
        data = data.dropna()
        time = data.xs( 'time', axis = 1, level = 'metric' ).iloc[[ 0, -1 ]]/ 3600
        time = time.reset_index( drop = True )

        # temperatures, take end value
        start = time.loc[ 0 ].values[ 0 ]
        end   = time.loc[ 1 ].values[ 0 ]

        start = temperatures.loc[ temp_times <= start ].values[ -1, 0 ]
        end   = temperatures.loc[ temp_times <= end ].values[ -1, 0 ]

        if start != end:
            logging.warning( '{}: Temperature change.'.format( name ) )

        temp = pd.Series( { name: end } )
        temps.append( temp )

    temperatures = pd.concat( temps )

    return temperatures


# ## Work




