# Utils / Metadata

import sys
from functools import partial

import pandas as pd

import bric_analysis_libraries.standard_functions as std


def get_standard_metadata_value( md_map, file, metadata ):
    """
    Gets metadata values from a file path

    :param md_map: Dictionary of keys to callables to extract metadata.
        Callables should accept a single parameter which is the file name.
        :param file: The file path to search
    :param metadata: The key of a standard metadata to retrieve
        [ 'sample' ]
    :returns: A list of metadata values
    """
    return md_map[ metadata ]( file )


def get_standard_metadata_values( md_map, file, metadata ):
    """
    Gets metadata values from a file path

    :param md_map: Dictionary of keys to callables to extract metadata.
        Callables should accept a single parameter which is the file name.
    :param file: The file path to search
    :param metadata: A list of standard metadata to retrieve
        [ 'sample' ]
    :returns: Tuple of metadata values
    """
    return tuple( md_map[ meta ]( file ) for meta in metadata )


def get_metadata_values( md_map, file, metadata ):
    """
    Gets metadata values from a file path.

    :param md_map: Dictionary of keys to callables to extract metadata.
        Callables should accept a single parameter which is the file name.
    :param file: The file path to search.
    :param metadata: Metadata from the file name is turned into MultiIndex columns.
        + If list, use standard keywords to include in index [ 'sample' ]
        + If Dictionary, keys indicate level name.
            Value can be a regex pattern to match or another dictionary.
            If a dictionary, key 'search' is a pattern to match
            passed to standard_functions#metadata_from_filename, and key-value pairs
            are keyword arguments of standard_functions#metadata_from_filename.
            + Reseserved key 'standard' can be provided with a list
                of standard metadata names.
            [See also standard_functions#metadata_from_filename]
    :returns: Tuple of metadata values.
    """
    if isinstance( metadata, list ):
        # use standard metadata
        return get_standard_metadata_values( md_map, file, metadata )

    if isinstance( metadata, dict ):
        # use custom metadata
        # key is name, value is regexp pattern or dictionary with pattern and arguments

        header_names = list( metadata.keys() )

        # get number of values
        val_len = len( header_names )
        if 'standard' in header_names:
            val_len += len( metadata[ 'standard' ] ) - 1

        vals = header_names.copy()
        for name, search in metadata.items():
            index = header_names.index( name )

            if name == 'standard':
                # insert standard keys
                vals[ index ] = get_standard_metadata_values( md_map, file, search )

            else:
                # custom key
                if isinstance( search, dict ):
                    pattern = search[ 'search' ]
                    kwargs = search.copy()
                    del kwargs[ 'search' ]

                else:
                    pattern = search
                    kwargs = {}

                vals[ index ] = std.metadata_from_file_name( pattern, file, **kwargs )

        # flatten standard keys
        if 'standard' in header_names:
            index = header_names.index( 'standard' )
            vals = vals[ :index ] + list( vals[ index ] ) + vals[ index + 1: ]

        return tuple( vals )


def metadata_to_names( metadata ):
    """
    :param metadata: Dictionary of metadata as passed to #get_metadata_values.
    :returns: Tuple of names corresponding to the metadata.
    [See also: #get_metadata_values]
    """
    names = []
    for name in metadata:
        if name == 'standard':
            # flatten standard names
            names = [ *names, *metadata[ 'standard' ] ]

        else:
            names.append( name )

    return tuple( names )


def get_metadata( md_map, file, metadata ):
    """
    :param md_map: Dictionary of keys to callables to extract metadata.
        Callables should accept a single parameter which is the file name.
    :param file: The file path to search.
    :param metadata: Dictionary of metadata as passed to #get_metadata_values.
    :returns: Dictionary of name-value pairs representing the extracted metadata.
    [See also: #get_metadata_values]
    """
    values = get_metadata_values( md_map, file, metadata )
    names = metadata_to_names( metadata )
    if len( values ) != len( names ):
        raise ValueError( 'Values and names have different lengths.' )

    return {
        names[ index ]: values[ index ] 
        for index in range( len( values ) )
    }


def metadata_parser( md_map ):
    """
    :param md_map: Dictionary of keys to callables to extract metadata.
        The callables should accept a single parameter which is the file name.
    :returns: Partial of #get_metadata with `md_map` specified.
            [See also #get_metadata]
    """
    return partial( get_metadata, md_map )


def metadata_to_dataframe_index( md_map, file, metadata, force_multi = False ):
    """
    :param md_map: Dictionary of keys to callables to extract metadata.
        Callables should accept a single parameter which is the file name.
    :param file: The file path to search.
    :param metadata: Dictionary of metadata as passed to #get_metadata_values.
    :param force_multi: Return pandas.MultiIndex even if only
        one metadata value is present.
        [Default: False]
    :returns: Pandas Index or MultiIndex representing the extracted metadata.
        Returns pandas.Index if only one value is present, otherwise returns
        pandas.MultiIndex.
    [See also: #get_metadata_values]
    """
    md = get_metadata( md_map, file, metadata )
    names = tuple( md.keys() )
    if len( metadata ) == 1:
        return pd.Index(
            md.values(),
            name = names
        )

    else:
        return pd.MultiIndex.from_tuples(
            [ md.values() ],
            name = names
        )

def metadata_dataframe_index_parser( md_map ):
    """
    :param md_map: Dictionary of keys to callables to extract metadata.
        The callables should accept a single parameter which is the file name.
    :returns: Partial of #metadata_to_dataframe_index with `md_map` specified.
            [See also #metadata_to_dataframe_index]
    """
    return partial( metadata_to_dataframe_index, md_map )