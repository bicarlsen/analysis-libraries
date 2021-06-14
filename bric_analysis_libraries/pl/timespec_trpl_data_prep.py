#!/usr/bin/env python
# coding: utf-8

# # TimeSpec TRPL Data Prep


import pandas as pd

from bric_analysis_libraries import standard_functions as std


# ## Data Prep


def import_datum( path ):
    """
    Import data from a LifeSpec II TRPL experiment.

    :param path: Path of file with data.
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

    return df



def import_data(
    folder_path,
    file_pattern = '*.csv',
    metadata = None,
    interpolate = 'linear',
    fillna = 0
):
    """
    Imports data from TimeSpec II TRPL experiments output files.

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
        data = import_datum( file, metadata = metadata, cps = cps ) # run local import datum function
        df.append( data )

    if interpolate is not None:
        df = std.common_reindex( df, how = interpolate, fillna = fillna )

    df = pd.concat( df, axis = 1 )
    return df

