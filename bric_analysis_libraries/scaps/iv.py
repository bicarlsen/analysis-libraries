# scaps/iv

import re

import numpy as numpy
import pandas as pd

from .. import standard_functions as std


def content_to_data_sections( content ):
    """
    :returns: List of data section each as a list of lines.
    """
    lines = content.split( '\n' )
    
    data_break_pattern = 'SCAPS [\d+\.]+'
    data_breaks = []
    for index, line in enumerate( lines ):
        if re.match( data_break_pattern, line ):
            data_breaks.append( index )

    data_breaks.append( len( lines ) )
    
    data_sections = [ 
        lines[ data_breaks[ i ] : data_breaks[ i + 1 ] - 1 ]
        for i in range( len( data_breaks ) - 1 )
    ]
    
    return data_sections


def section_positions( section ):
    """
    :returns: Dictionary of line numbers for section components.
    """
    parameters_pattern = '**Batch parameters**'
    params_start = section.index( parameters_pattern ) + 1
    params_end = section.index( '', params_start )
    
    header = params_end + 1 
    data_start = params_end + 3
    data_end = section.index( '', data_start )
    
    cp_pattern = 'solar cell parameters deduced from calculated IV-curve:'
    cp_start = section.index( cp_pattern )
    cp_end = section.index( '', cp_start )
    
    return {
        'params': ( params_start, params_end ),
        'header': header,
        'data':   ( data_start, data_end ),
        'cell parameters': ( cp_start, cp_end )
    }
    

def section_parameters( section ):
    """
    :returns: List of parameters from a data section.
    """
    pos = section_positions( section )[ 'params' ]
    params = [
        param.split( ':' )
        for param in section[ pos[ 0 ] : pos[ 1 ] ]
    ]
    
    params = [
        [ param[ 0 ].split( '>>' ), float( param[ 1 ] ) ]
        for param in params
    ]
    
    return params


def section_data( section, remove_header_units = True ):
    """
    :returns: pandas DataFrame representing the section.
    """
    pos = section_positions( section )

    # get data
    header = [ h.strip() for h in section[ pos[ 'header' ] ].split( '\t' ) ]
    v_index = 'v(V)'
    if remove_header_units:
        header = [ re.sub( '\(.*\)', '', h ) for h in header ]
        v_index = 'v'
        
    data = [ 
        [ float( v ) for v in d.split( '\t' ) ] 
        for d in section[ pos[ 'data' ][ 0 ] : pos[ 'data' ][ 1 ] ] 
    ]

    df = pd.DataFrame( data, columns = header )
    df = df.set_index( v_index )
    df.columns = df.columns.rename( 'metrics' )
    
    # get parameters
    params = section_parameters( section )
    p_names = [ tuple( p[ 0 ] ) for p in params ]
    p_vals =  [ p[ 1 ] for p in params ]
    df = std.insert_index_levels( df, p_vals, names = p_names )
    
    return df


def section_cell_parameters( section, remove_header_units = True ):
    """
    """
    pos = section_positions( section )[ 'cell parameters' ]
    param_pattern = '(.+)=\s*(\S+)\s+(.+)'  # <name> = <value> <unit>
    params = {}
    for line in section[ pos[ 0 ]: pos[ 1 ] ]:
        m = re.match( param_pattern, line )
        if m is None:
            continue
            
        name = m.group( 1 ).strip()
        val  = float( m.group( 2 ).strip() )
        unit = m.group( 3 ).strip()
        
        if not remove_header_units:
            name = f'{name} ({unit})'
        
        params[ name ] = [ val ]
    
    return params

    
def import_iv_data( file, **kwargs ):
    """
    :returns: Pandas DataFrame of IV data.
    """
    with open( file ) as f:
        content = f.read()
        
    df = []
    for section in content_to_data_sections( content ):
        tdf = section_data( section, **kwargs )
        df.append( tdf )

    df = std.common_reindex( df )
    df = pd.concat( df, axis = 1 ).sort_index( axis = 1 )
    return( df )


def import_cell_paramters( file, **kwargs ):
    """
    :returns: Pandas DataFrame of cell parameters.
    """
    with open( file ) as f:
        content = f.read()
        
    df = []
    for section in content_to_data_sections( content ):
        cp_params = section_cell_parameters( section, **kwargs )
        
        params = section_parameters( section )
        p_names = [ tuple( p[ 0 ] ) for p in params ]
        p_vals =  tuple( p[ 1 ] for p in params )

        tdf = pd.DataFrame( cp_params )
        tdf.index = pd.MultiIndex.from_tuples(
            ( p_vals, ), 
            names = p_names 
        )
    
        df.append( tdf )

    df = pd.concat( df, axis = 0 ).sort_index( axis = 0 )
    return( df )
