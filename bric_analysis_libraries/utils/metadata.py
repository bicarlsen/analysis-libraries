# Utils / Metadata

import sys
from functools import partial

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
        + If Dictionary, keys indicate level name, value is pattern to match.
            or another dictionary with 'search' key being the pattern to match, and additional
            entries matching arguments passed to standard_functions#metadata_from_filename.
            + Reseserved key 'standard' can be provided with a list value to get standard metadata.
    :returns: A list of metadata values.
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
            vals = vals[ :index ] + vals[ index ] + vals[ index + 1: ]

        return vals


def metadata_parser( md_map ):
    """
    :param md_map: Dictionary of keys to callables to extract metadata.
        The callables should accept a single parameter which is the file name.
    :returns: Function that accepts a file name and metadata map, and returns the extracted metadata.
    """
    return partial( get_metadata_values, md_map )