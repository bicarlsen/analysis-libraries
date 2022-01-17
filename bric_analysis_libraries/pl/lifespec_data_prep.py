#!/usr/bin/env python
# coding: utf-8

# LifeSpec TRPL Data Prep

import os
import sys

import pandas as pd

from bric_analysis_libraries import standard_functions as std
from bric_analysis_libraries.utils import metadata


# Data Prep


def sample_from_file_name( file ):
    name_search  = '^(.*?)'  # use lazy search
    return std.metadata_from_file_name( name_search, file, delimeter = '--' )


def period_from_file_name( file, base_unit = 'ns' ):
    """
    Retrieve pulse period from file name.

    :param file: File name to extract data from.
    :param base_unit: Base unit of time. [Default: 'ns']
    :returns: period in units of base_unit.
    """
    period_search = 'period(\d+[nu]s)'
    match = std.metadata_from_file_name( period_search, file )
    
    # convert to correct base
    val = float( match[ :-2 ] )
    unit = match[ -2: ]
    if unit == base_unit:
        # base unit and extracted unit match, no change needed.
        return val

    if unit == 'ns':
        if base_unit == 'us':
            return val* 1e3

    elif unit == 'us':
        if base_unit == 'ns':
            return val* 1e-3

    raise ValueError( 'Could not convert time unit to base unit.' )


def center_wavelength_from_file_name( file ):
    """
    Extract center wavelength from file name.

    :param file: File name.
    :returns: Center wavelength.
    """
    center_search = 'em<>nm'
    center = std.metadata_from_file_name( center_search, file, is_numeric = True )
    return center


def bandwidth_from_file_name( file ):
    """
    Extract bandwidth from file name.

    :parma file: File name.
    :returns: Bandwidth.
    """ 
    bandwidth_search = 'bandwidth<>nm'
    bandwidth = std.metadata_from_file_name( bandwidth_search, file, is_numeric = True )
    return bandwidth


# create standard metadata parser
metadata_dataframe_index = metadata.metadata_dataframe_index_parser( {
    # key is name of metadata, value is coorespoinding function
    ( module.replace( '_from_file_name', '' ) ): getattr( sys.modules[ __name__ ], module )
    for module in dir( sys.modules[ __name__ ] )
    if '_from_file_name' in module  # only retrieve functions with _from_file_name in their name
} )


def import_datum( path, metadata = None ):
    """
    Import data from a LifeSpec II experiment.

    :param path: Path of file with data.
    :param metadata: Dictionary of metadata to extract from the file name.
    :returns: Pandas DataFrame.
    """
    # find header data
    x_label = None
    y_label = None
    header_end = None
    with open( path ) as f:
        for index, line in enumerate( f ):
            fields = line.split( ',' )

            key = fields[ 0 ].lower()
            if key == '\n':
                # end of header
                header_end = index
                break

            value = fields[ 1 ].lower()
            if key == 'xaxis':
                x_label = value

            elif key == 'yaxis':
                y_label = value

    # import data
    df = pd.read_csv(
        path,
        usecols   = [ 0, 1 ],
        skiprows  = header_end,
        index_col = x_label,
        names     = ( x_label, y_label )
    )

    if metadata is not None:
        cols = metadata_dataframe_index( path, metadata )
        df.columns = cols 

    return df


def import_data(
    folder_path,
    file_pattern = '*.csv',
    metadata = None,
    interpolate = 'linear',
    fillna = 0
):
    """
    Imports data from TimeSpec II experiments output files.

    :param folder_path: The file path containing the data files
    :param file_pattern: A glob pattern to filter the imported files [Default: '*']
    :param metadata: Metadata from the file name is turned into MultiIndex columns.
       [Default: None]
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
        data = import_datum( file, metadata = metadata )  # run local import datum function
        df.append( data )

    if interpolate is not None:
        df = std.common_reindex( df, how = interpolate, fillna = fillna )

    df = pd.concat( df, axis = 1 ).sort_index( axis = 1 )
    return df
