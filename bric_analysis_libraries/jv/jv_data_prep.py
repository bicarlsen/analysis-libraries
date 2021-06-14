# JV Data Prep

import numpy as np
import pandas as pd

import bric_analysis_libraries.standard_functions as std


def split_jv_scan( df ):
    """
    Split JV scan into forward and reverse components.
    :param df: Pandas DataFrame representing JV scan.
    :returns: Pandas Dataframe split by direciton.
    """
    multiindex = isinstance( df.columns, pd.MultiIndex )
    if multiindex:
        dv = df.xs( 'voltage', level = 'metrics', axis = 1 )

    else:
        dv = df.voltage

    dv = dv.diff().dropna()

    change = np.sign( dv ).diff().fillna( 0 ) # calculate sign changes
    change = np.where( change != 0 )[ 0 ] # get indices of sign changes
    if change.size > 1:
        # more than one sign change
        raise ValueError( 'Multiple sign changes detected in scan.' )

    elif change.size == 0:
        # no sign changes detected
        raise ValueError( 'No sign changes detected in scan.' )

    change = change[ 0 ]

    # break scans apart
    forward_first = ( dv.values[ 0 ] > 0 )
    df = (
        [
            df[ :( change + 1 ) ],
            df[ change: ]
        ]

        if forward_first else

        [
            df[ change: ],
            df[ :( change + 1 ) ]
        ]
    )

    for index, tdf in enumerate( df ):
        # set index
        if multiindex:
            tdf.index = pd.Index(
                tdf.xs( 'voltage', level = 'metrics', axis = 1 ).values.flatten(),
                name = 'voltage'
            )

            tdf = tdf.drop( 'voltage', level = 'metrics', axis = 1 )
            key_level = 1

        else:
            tdf.set_index( 'voltage', inplace = True )
            tdf.columns.rename( 'metrics', inplace = True )
            key_level = 0

        tdf = tdf[ tdf.index.notnull() ]

        # create multi-index
        name = 'forward' if ( index == 0 ) else 'reverse'
        tdf = std.insert_index_levels(
            tdf,
            levels = [ name ],
            names  = [ 'direction' ],
            key_level = key_level
        )

        df[ index ] = tdf # replace with modified


    # reindex for common voltage values
    df = std.common_reindex( df )

    # combine scan directions
    df = pd.concat( df, axis = 1 )

    return df