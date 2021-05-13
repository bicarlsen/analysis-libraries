# scaps/eb

import re

import numpy as np
import pandas as pd

from . import common
from .. import standard_functions as std


def import_eb_data( file, separator = '\t' ):
    """
    Import an energy band file from SCAPS.

    :param file: Path to the file, usually a .eb file.
    :returns: DataFrame representing the data. 
    """
    with open( file ) as f:
        h_content, d_content = common.split_content_header_data( f.read() )
    
    m_order = metrics_order( h_content )
    h_names = common.batch_settings( h_content )

    mesh = mesh_from_content( d_content, names = h_names, separator = separator )
    mesh = std.insert_index_levels( 
        mesh, 'x', names = 'metrics', key_level = mesh.columns.shape[ 0 ] 
    )
    
    df = [ mesh ]
    sections = metric_sections( d_content, names = h_names )
    for param, data in sections.items():
        tdf = data[ 2 ]
        tdf = std.insert_index_levels( 
            tdf, data[ 0 ], names = 'metrics', key_level = tdf.columns.shape[ 0 ]
        )
        
        df.append( tdf )
    
    df = pd.concat( df, axis = 1 ).sort_index( axis = 1 )
    return df
    

def metrics_order( content ):
    """
    Get order of metrics from content.

    :param content: Content to search.
    :returns: List of metrics in order.
    :raises RuntimeError: If the metrics section could not be found.
    """
    metrics_pattern = '\*{3} Overview of recorder settings \*{3}\n\n(.+?)\n\n'
    match = re.search( metrics_pattern, content, re.DOTALL )
    
    if match is None:
        raise RuntimeError( 'Metrics order could not be found.' )
        
    metrics = match.group( 1 ).split( '\n' )
    return metrics


def split_section_header_data( content ):
    """
    Splits a section into its header and data components.

    :param content: Section content.
    :returns: Tuple of ( header, data )
    """
    i_content = tuple( enumerate( content ) )
    
    header_lines = [ 
        index 
        for index, line in i_content 
        if line.startswith( 'bp' )
    ]
    
    headers = [ l for index, l in i_content if index in header_lines ]
    data = [ l for index, l in i_content if index not in header_lines ]
    
    return ( headers, data )


def headers_to_index( headers, names = None, separator = '\t' ):
    """
    Converts headers into a Pandas Index or MultiIndex.

    :param headers: List of headers.
    :param names: Level names to use for the index. [Default: None]
    :param separator: Data separator. [Default: '\t']
    :returns: Pandas Index or MultiIndex.
    """
    header_pattern = 'bp (\d+):(.+)'
    headers = [ re.match( header_pattern, h ) for h in headers ]
    headers = { 
        h.group( 1 ): map( float, h.group( 2 ).strip().split( separator ) )
        for h in headers 
        if h is not None 
    }
    
    if len( headers ) == 0:
        raise RuntimeError( 'No valid headers.' )
        
    elif len( headers ) == 1:
        for name, vals in headers.items():
            index = pd.Index( vals, name = name )

    else:
        values = []
        o_names = []
        for index, vals in headers.items():
            values.append( vals )
            o_names.append( index )
            
        values = tuple( zip( *values ) )
        index = pd.MultiIndex.from_tuples( values, names = o_names )
        
    if names is not None:
        rename = [ names[ h_name ] for h_name in index.names ]
        index = index.rename( rename )
    
    return index


def section_to_dataframe( section, separator = '\t', names = None ):
    """
    Converts a section into a DataFrame.

    :param section: Section to convert.
    :param separator: Data separator. [Default: '\t']
    :param names: Names to use for column levels. [Default: None]
    :return: DataFrame representing the section data.
    """
    if not isinstance( section, list ):
        section = section.split( '\n' )

    headers, data = split_section_header_data( section )
    cols = headers_to_index( headers, names = names, separator = separator )
    
    data = [ map( float, d.split( separator ) ) for d in data ]
    df = pd.DataFrame( data )
    df = df.set_index( 0 )
    df.index = df.index.astype( np.int64 )
    df.columns = cols
    
    return df


def mesh_from_content( content, names = None, section_break = '\n{4}', separator = '\t' ):
    """
    Creates a DataFrame representing the simulation mesh.

    :param content: Content to parse.
    :param names: Names to use for column levels. [Default: None]
    :param section_break: RegEx representing the section break. [Default: '\n{5}']
    :returns: DataFrame representing the mesh.
    """
    mesh_pattern = '\* The mesh.+?\*(.+?)' + section_break
    match = re.search( mesh_pattern, content, re.DOTALL )
    if match is None:
        raise RuntimeError( 'Mesh could not be found.' )
        
    mesh = match.group( 1 ).strip().split( '\n' )
    mesh = section_to_dataframe( mesh, names = names, separator = separator )
    return mesh
    
    
def metric_sections( content, names = None, section_break = '\n{4}', separator = '\t' ):
    """
    Creates DataFrames representing the simulation parameters.

    :param content: Content to parse.
    :param names: Names to use for column levels. [Default: None]
    :param section_break: RegEx representing the section break. [Default: '\n{5}']
    :returns: Dictionary of tuples of the form
        { index: ( name, code, DataFrame ) } representing the sections.
    """
    section_pattern = '\* Parameter (\d+):\s+(.+?)\*\n\*(.+?)\*\n\n(.+?)' + section_break
    matches = re.findall( section_pattern, content, re.DOTALL )
    sections = {}
    for match in matches:
        param = int( match[ 0 ].strip() )
        sections[ param ] = (
            match[ 1 ].strip(),
            match[ 2 ].strip(),
            section_to_dataframe( match[ 3 ], names = names, separator = separator )
        )
    
    return sections