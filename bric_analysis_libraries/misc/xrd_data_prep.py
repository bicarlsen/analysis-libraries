
# coding: utf-8

# # XRD Data Prep


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



def import_datum( file, reindex = True ):
    """
    Imports a .csv file from XRD data.

    :param file: The file path to load.
    :param reindex: Set time as DataFrame Index. [Default: True]
    :returns: A Pandas DataFrame.
    """
    # parse header
    header_pattern = '\[Scan points\]'
    header_position = None
    with open( file ) as f:
        for num, line in enumerate( f ):
            if re.match( header_pattern, line ) is not None:
                header_position = num + 2 # account for next line, and header line
                break

    if header_position is None:
        raise RuntimeError( 'Could not find header.' )

    df = pd.read_csv( file, skiprows = header_position, names = [ 'angle', 'intensity' ] )

    if reindex:
        df = df.set_index( 'angle' )

    return df


def import_data( folder, file_pattern = '*.csv' ):
    """
    Imports XRD data.

    :param folder: Folder path containing the data files.
    :param file_pattern: File pattern of data files, in glob format. [Default: *.txt]
    :returns: DataFrame containing imported data.
    """
    return std.import_data( import_datum, folder, file_pattern = file_pattern )


# # Work
