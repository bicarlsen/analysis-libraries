#!/usr/bin/env python
# coding: utf-8

# # Cary UV-Vis Absorption Data Prep


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


# ## Data Prep


def import_datum( path, **kwargs ):
    """
    Imports a Cary UV Vis Absorption data file into a Pandas DataFrame.

    :param path: Path to the file.
    :returns: Pandas DataFrame.
    """
    data_end = None
    fields   = None
    metrics  = None
    with open( path ) as f:
        for ( index, line ) in enumerate( f ):
            # search for end of data
            # occurs at first blank line

            # every line ends with trailing comma
            if index is 0:
                fields = line.split( ',' )[ :-1 ]

            elif index is 1:
                # get headers
                metrics = line.split( ',' )[ :-1 ]

            elif line is '\n':
                # no data in line
                data_end = index
                break

    # parse metrics for sample width
    # new samples begin with wavelength
    sample_indices = []
    for index, metric in enumerate( metrics ):
        if metric == 'Wavelength (nm)':
            sample_indices.append( index )

    # import samples individually
    df = []
    for ( i, sample_index ) in enumerate( sample_indices ):
        # get sample columns
        next_sample_index = ( # get next sample index, or end if last sample
            sample_indices[ i + 1 ]
            if ( i + 1 ) < len( sample_indices ) else
            len( metrics ) # final sample
        )
        use_cols = range( sample_index, next_sample_index )

        # get names of fields
        # ignore first as it is wavelength
        # will be set to index
        names = map(
            str.lower,
            metrics[ sample_index + 1 : next_sample_index ]
        )


        tdf = pd.read_csv(
            path,
            header = None,
            skiprows = 2,
            nrows = data_end - 2, # account for ignored headers
            usecols = use_cols,
            index_col = 0, # wavelength as index
            names = [ 'wavelength', *names ],
            **kwargs
        ).dropna()

        # add sample name to header
        tdf = std.insert_index_levels( tdf, fields[ sample_index ] )
        df.append( tdf )

    std.common_reindex( df )
    df = pd.concat( df, axis = 1 )
    return df



def import_data( folder, file_pattern = '*.csv' ):
    """
    Imports data from a Cary UV-Vis experiment.

    :param folder: Folder path containing the data files.
    :param file_pattern: File pattern of data files, in glob format. [Default: *.csv]
    :returns: DataFrame containing imported data.
    """
    return std.import_data( import_datum, folder, file_pattern = file_pattern )

