# scaps/common

import re

import numpy as np
import pandas as pd

from .. import standard_functions as std
    
    
def split_content_header_data( content ):
    """
    Splits content into its header and data components.

    :param content: The content to split.
    :returns: Tuple of ( header, data ).
    :raises RuntimeError: If the data could not be split.
    """
    split_pattern = '(.*)\*{3} The recorded data \*{3}(.*)'
    match = re.search( split_pattern, content, re.DOTALL )
    if match is None:
        raise RuntimeError( 'Could not split content.' )
        
    return ( match.group( 1 ), match.group( 2 ) )


def batch_settings( content ):
    """
    Gets the batch settings.

    :param content: Content to search.
    :returns: Dictionary of settings with their index.
    :raises RuntimeError: If settings section could not be found.
    """
    section_pattern = '\*{3} Overview of batch settings \*{3}\n\n(.+?)\n\n'
    match = re.search( section_pattern, content, re.DOTALL )
    
    if match is None:
        raise RuntimeError( 'Settings could not be found.' )
        
    setting_pattern = 'batch parameter (\d+):(.*):'
    settings = {}
    for bp in match.group( 1 ).split( '\n' ):
        s_match = re.search( setting_pattern, bp )
        if s_match is None:
            continue
           
        s_num = s_match.group( 1 )
        s_val = s_match.group( 2 ).split( '>>' )
        s_val = tuple( sv.strip() for sv in s_val )
        settings[ s_num ] = s_val
    
    return settings