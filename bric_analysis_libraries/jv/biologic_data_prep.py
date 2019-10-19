
# coding: utf-8

# # Biologic Data Prep

# ### Imports

# In[1]:


import logging

import numpy as np
import pandas as pd

from bric_analysis_libraries import standard_functions as std


# ## Data Prep

# In[ ]:


# convenience methods
def channel_from_file_path( path, pattern = 'ch<>' ):
    """
    Gets the channel from the file path.
    
    :param path: File path.
    :param pattern: RegEx pattern describing teh channel pattern.
        Passed to standard_functions#metadata_from_file_name.
        [Default: 'ch<>']
    """
    ch = std.metadata_from_file_name(
        pattern,
        path,
        is_numeric = True,
        full_path = True
    )
    
    return int( ch )


# In[2]:


def import_voc_datum( file, channel_index = 'ch<>', set_index = True ):
    """
    Imports Voc datum from the given file.
    
    :param file: File path.
    :param channel_index: Add channel from file path as index level.
        Uses value as pattern in standard_functions#metadata_from_file_name.
        None if channel should be excluded.
        [Default: 'ch<>']
    :param set_index: Sets the index to time. [Default: True]
    :returns: Pandas DataFrame.
    """
    header = [ 'time', 'voltage' ]
    df = pd.read_csv( file, names = header, skiprows = 1 )
    
    if set_index:
        df.set_index( 'time', inplace = True )
        
    df.columns.rename( 'metrics', inplace = True )
    
    if channel_index is not None:
        ch = channel_from_file_path( file, channel_index )
        df = std.insert_index_levels( df, ch, 'channel' )
    
    return df


def import_jv_datum( file, channel_index = 'ch<>', by_scan = True ):
    """
    Imports JV datum from the given file.
    
    :param file: File path.
    :param channel_index: Add channel from file path as index level.
        Uses value as pattern in standard_functions#metadata_from_file_name.
        None if channel should be excluded.
        [Default: 'ch<>']
    :param by_scan: Breaks data into forward and reverse scans, and sets the index to voltage. [Default: True]
    :returns: Pandas DataFrame.
    
    :raises ValueError: If multiple sign changes in the scan are detected.
    """
    header = [ 'voltage', 'current', 'power' ]
    df = pd.read_csv( file, names = header, skiprows = 1 )
    
    if by_scan:
        # detect direction change in votlage scan
        dv = df.voltage.diff()
        change = np.nan_to_num( np.diff( np.sign( dv ) ) ) # calculate sign changes
        change = np.where( change != 0 )[ 0 ] # get indices of sign changes
        if change.size > 1:
            # more than one sign change
            raise ValueError( 'Multiple sign changes detected in scan.' )
        
        change = change[ 0 ]
        
        # break scans apart
        forward_first = ( dv[ 0 ] > 0 )
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
            tdf.set_index( 'voltage', inplace = True )
            tdf.columns.rename( 'metrics', inplace = True )
       
            # create multi-index
            name = 'forward' if ( index == 0 ) else 'reverse'
            tdf = std.add_index_levels( 
                tdf, 
                { name: [ 'current', 'power' ] }, 
                names = [ 'direction' ] 
            )
            
            df[ index ] = tdf # replace with modified
        
        # reindex for common voltage values
        df = std.common_reindex( df )

        # combine scan directions
        df = pd.concat( df, axis = 1 )
        
    if channel_index is not None:
        ch = channel_from_file_path( file, channel_index )
        df = std.insert_index_levels( df, ch, 'channel' )
    
    return df


def import_mpp_datum( 
    file, 
    channel_index = 'ch<>', 
    set_index = True, 
    drop_cycle = True,
    skiprows = 2
):
    """
    Imports MPP datum from the given file.
    
    :param file: File path.
    :param channel_index:Add channel from file path as index level.
        Uses value as pattern in standard_functions#metadata_from_file_name.
        None if channel should be excluded.
        [Default: 'ch<>']
    :param set_index: Sets the index to time. [Default: True]
    :param drop_cycle: Removes cycle information from the data. [Defautl: True]
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
        
    if channel_index is not None:
        ch = channel_from_file_path( file, channel_index )
        df = std.insert_index_levels( df, ch, 'channel' )
        
    return df


# In[ ]:


def import_voc_data( folder, file_pattern = 'voc.csv', **kwargs ):
    """
    Imports Voc data.
    
    :param folder: Folder path containing data files.
    :param file_pattern: File pattern of data files, in glob format. [Default: voc.csv]
    :param kwargs: Arguments passed to standard_functions import_data()
    :returns: DataFrame containg imported data.
    """
    return std.import_data( import_voc_datum, folder, file_pattern = file_pattern, **kwargs )


def import_jv_data( folder, file_pattern = 'jv.csv', **kwargs ):
    """
    Imports JV data.
    
    :param folder: Folder path containing data files.
    :param file_pattern: File pattern of data files, in glob format. [Default: jv.csv]
    :param kwargs: Arguments passed to standard_functions import_data()
    :returns: DataFrame containg imported data.
    """
    return std.import_data( import_jv_datum, folder, file_pattern = file_pattern, **kwargs )


def import_mpp_data( folder, file_pattern = 'mpp.csv', **kwargs ):
    """
    Imports MPP data.
    
    :param folder: Folder path containing data files.
    :param file_pattern: File pattern of data files, in glob format. [Default: mpp.csv]
    :param kwargs: Arguments passed to standard_functions import_data()
    :returns: DataFrame containg imported data.
    """
    return std.import_data( import_mpp_datum, folder, file_pattern = file_pattern, **kwargs )


def import_all_mpp_data( folder ):
    """
    Imports Voc, JV, and MPP data.
    
    :param folder: Folder path.
    :returns: Tuple of ( voc, jv, mpp ).
    """
    
    return (
        import_voc_data( folder ),
        import_jv_data(  folder ),
        import_mpp_data( folder )
    )


# # Work
